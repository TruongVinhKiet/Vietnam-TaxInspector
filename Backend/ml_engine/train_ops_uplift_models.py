from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return float(d)


def train_ops_models() -> dict:
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    c.tax_code,
                    COALESCE(a.risk_score, c.risk_score, 0) AS fraud_score,
                    COALESCE(a.model_confidence, 0.55) AS fraud_confidence,
                    COALESCE(d.prob_90d, 0) AS delinquency_90d,
                    COALESCE(vr.risk_score, 0) AS vat_refund_score,
                    COALESCE((SELECT AVG(priority_score) FROM audit_selection_predictions asp WHERE asp.tax_code = c.tax_code), 0) AS prior_priority,
                    COALESCE((SELECT SUM(amount_collected) FROM collection_outcomes co WHERE co.tax_code = c.tax_code), 0) AS total_collected,
                    COALESCE((SELECT COUNT(*) FROM collection_actions ca WHERE ca.tax_code = c.tax_code), 0) AS n_actions,
                    COALESCE((SELECT AVG(CASE WHEN ca.result='success' THEN 1.0 WHEN ca.result='partial' THEN 0.5 ELSE 0.0 END) FROM collection_actions ca WHERE ca.tax_code = c.tax_code), 0) AS action_success
                FROM companies c
                LEFT JOIN LATERAL (
                    SELECT risk_score, model_confidence
                    FROM ai_risk_assessments
                    WHERE tax_code = c.tax_code
                    ORDER BY created_at DESC
                    LIMIT 1
                ) a ON TRUE
                LEFT JOIN LATERAL (
                    SELECT prob_90d
                    FROM delinquency_predictions
                    WHERE tax_code = c.tax_code
                    ORDER BY created_at DESC
                    LIMIT 1
                ) d ON TRUE
                LEFT JOIN LATERAL (
                    SELECT risk_score
                    FROM vat_refund_predictions
                    WHERE tax_code = c.tax_code
                    ORDER BY created_at DESC
                    LIMIT 1
                ) vr ON TRUE
                """
            )
        ).mappings().all()
        if len(rows) < 300:
            raise RuntimeError(f"Need >=300 rows for ops uplift models; got {len(rows)}")

        X = []
        y_audit = []
        y_collection = []
        for r in rows:
            fraud = _safe_float(r["fraud_score"])
            conf = _safe_float(r["fraud_confidence"], 0.55)
            delinq = _safe_float(r["delinquency_90d"])
            refund = _safe_float(r["vat_refund_score"])
            prior = _safe_float(r["prior_priority"])
            n_actions = _safe_float(r["n_actions"])
            action_success = _safe_float(r["action_success"])
            total_collected = _safe_float(r["total_collected"])
            features = [fraud, conf, delinq, refund, prior, n_actions, action_success]
            X.append(features)
            y_audit.append(1 if (total_collected > 0 and action_success >= 0.2) else 0)
            y_collection.append(total_collected)

        X = np.asarray(X, dtype=float)
        y_audit = np.asarray(y_audit, dtype=int)
        y_collection = np.asarray(y_collection, dtype=float)
        if len(np.unique(y_audit)) < 2:
            raise RuntimeError("Audit uplift labels have one class only.")

        Xa_train, Xa_test, ya_train, ya_test = train_test_split(X, y_audit, test_size=0.2, random_state=42, stratify=y_audit)
        audit_model = RandomForestClassifier(
            n_estimators=260,
            max_depth=9,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        audit_model.fit(Xa_train, ya_train)
        audit_prob = audit_model.predict_proba(Xa_test)[:, 1]
        audit_auc = float(roc_auc_score(ya_test, audit_prob))
        audit_pr_auc = float(average_precision_score(ya_test, audit_prob))

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_collection, test_size=0.2, random_state=42)
        collection_model = RandomForestRegressor(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        collection_model.fit(Xc_train, yc_train)
        yc_pred = collection_model.predict(Xc_test)
        collection_r2 = float(r2_score(yc_test, yc_pred))

        model_version = f"ops-learned-{date.today().isoformat()}"
        audit_model_path = MODEL_DIR / "audit_selection_learned_model.joblib"
        collection_model_path = MODEL_DIR / "collections_uplift_learned_model.joblib"
        meta_path = MODEL_DIR / "ops_uplift_model_meta.json"
        report_path = MODEL_DIR / "ops_uplift_quality_report.json"
        joblib.dump(audit_model, audit_model_path)
        joblib.dump(collection_model, collection_model_path)
        meta = {
            "model_version": model_version,
            "features": ["fraud_score", "fraud_confidence", "delinquency_90d", "vat_refund_score", "prior_priority", "n_actions", "action_success"],
            "trained_at": datetime.utcnow().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report = {
            "model_version": model_version,
            "metrics": {
                "audit_auc": round(audit_auc, 4),
                "audit_pr_auc": round(audit_pr_auc, 4),
                "collection_r2": round(collection_r2, 4),
            },
            "gates": {
                "audit_auc_min": {"threshold": 0.64, "actual": round(audit_auc, 4), "pass": bool(audit_auc >= 0.64)},
                "audit_pr_auc_min": {"threshold": 0.42, "actual": round(audit_pr_auc, 4), "pass": bool(audit_pr_auc >= 0.42)},
                "collection_r2_min": {"threshold": 0.20, "actual": round(collection_r2, 4), "pass": bool(collection_r2 >= 0.20)},
            },
        }
        report["gates"]["overall_pass"] = bool(
            report["gates"]["audit_auc_min"]["pass"]
            and report["gates"]["audit_pr_auc_min"]["pass"]
            and report["gates"]["collection_r2_min"]["pass"]
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        registry = ModelRegistryService(db)
        dataset_hash = str(abs(hash(tuple(tuple(row) for row in X[:1000]))))
        dataset_version_id = registry.register_dataset_version(
            dataset_key="ops_uplift_training_dataset",
            dataset_version=model_version,
            entity_type="company",
            row_count=len(X),
            source_tables=[
                "companies",
                "ai_risk_assessments",
                "delinquency_predictions",
                "vat_refund_predictions",
                "collection_actions",
                "collection_outcomes",
            ],
            filters={"date": date.today().isoformat()},
            data_hash=dataset_hash,
            created_by="train_ops_uplift_models",
        )
        label_version_id = registry.register_label_version(
            label_key="ops_uplift_labels",
            label_version=model_version,
            entity_type="company",
            label_source="collection_outcomes",
            positive_count=int(y_audit.sum()),
            negative_count=int((1 - y_audit).sum()),
            label_hash=dataset_hash,
            notes="Audit binary uplift + collection regression labels.",
        )
        experiment_id = registry.ensure_experiment(
            experiment_key="ops_uplift_learned",
            model_name="ops_uplift",
            objective="learned_collections_and_audit",
            owner="ml-platform",
            metadata={"pipeline": "rf_dual_track"},
        )
        run_id = registry.start_training_run(
            model_name="ops_uplift",
            experiment_id=experiment_id,
            model_version=model_version,
            dataset_version_id=dataset_version_id,
            label_version_id=label_version_id,
            hyperparams={"audit_model": "RandomForestClassifier", "collection_model": "RandomForestRegressor"},
        )
        registry.complete_training_run(
            run_id=run_id,
            status="completed",
            metrics=report["metrics"],
            artifacts={
                "audit_model_path": str(audit_model_path),
                "collection_model_path": str(collection_model_path),
                "meta_path": str(meta_path),
                "report_path": str(report_path),
            },
        )
        registry.register_rollout(
            model_name="ops_uplift",
            model_version=model_version,
            environment="staging",
            rollout_type="shadow",
            status="planned",
            notes="Learned uplift replacing heuristic formulas for audit/collections materializers.",
            metadata={"quality_report": report},
        )
        return report


if __name__ == "__main__":
    result = train_ops_models()
    print(json.dumps(result, indent=2))
