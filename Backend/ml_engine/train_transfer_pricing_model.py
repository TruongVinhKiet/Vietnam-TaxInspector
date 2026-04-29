from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_transfer_pricing_model() -> dict:
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    tr.record_id,
                    tr.unit_price,
                    pr.p10,
                    pr.p50,
                    pr.p90
                FROM trade_records tr
                JOIN pricing_reference_curves pr
                  ON pr.goods_key = tr.goods_category
                 AND pr.country_pair = ('VN-' || tr.counterparty_country)
                 AND pr.time_bucket = (EXTRACT(YEAR FROM tr.trade_date)::text || '-' || LPAD(EXTRACT(MONTH FROM tr.trade_date)::text, 2, '0'))
                """
            )
        ).mappings().all()
        if len(rows) < 200:
            raise RuntimeError(f"Need >=200 rows for transfer pricing model; got {len(rows)}")

        feats = []
        labels = []
        for row in rows:
            unit_price = float(row["unit_price"] or 0.0)
            p10 = float(row["p10"] or 0.0)
            p50 = float(row["p50"] or 0.0)
            p90 = float(row["p90"] or 0.0)
            spread = max(1.0, p90 - p10)
            z_score = (unit_price - p50) / spread
            feats.append([unit_price, p10, p50, p90, spread, z_score])
            labels.append(1 if (unit_price > p90 or unit_price < p10) else 0)

        X = np.asarray(feats, dtype=float)
        y = np.asarray(labels, dtype=int)
        if len(np.unique(y)) < 2:
            raise RuntimeError("Transfer pricing labels have one class only.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(
            n_estimators=240,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, prob))
        pr_auc = float(average_precision_score(y_test, prob))

        model_version = f"transfer-pricing-ml-{date.today().isoformat()}"
        model_path = MODEL_DIR / "transfer_pricing_model.joblib"
        meta_path = MODEL_DIR / "transfer_pricing_model_meta.json"
        report_path = MODEL_DIR / "transfer_pricing_quality_report.json"
        joblib.dump(model, model_path)
        meta = {
            "model_version": model_version,
            "features": ["unit_price", "p10", "p50", "p90", "spread", "z_score"],
            "trained_at": datetime.utcnow().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report = {
            "model_version": model_version,
            "metrics": {"auc": round(auc, 4), "pr_auc": round(pr_auc, 4)},
            "gates": {
                "auc_min": {"threshold": 0.68, "actual": round(auc, 4), "pass": bool(auc >= 0.68)},
                "pr_auc_min": {"threshold": 0.45, "actual": round(pr_auc, 4), "pass": bool(pr_auc >= 0.45)},
                "overall_pass": bool(auc >= 0.68 and pr_auc >= 0.45),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        registry = ModelRegistryService(db)
        dataset_hash = str(abs(hash(tuple(tuple(x) for x in feats[:1000]))))
        dataset_version_id = registry.register_dataset_version(
            dataset_key="transfer_pricing_training_dataset",
            dataset_version=model_version,
            entity_type="trade_record",
            row_count=len(rows),
            source_tables=["trade_records", "pricing_reference_curves"],
            filters={"label_rule": "outside_p10_p90"},
            data_hash=dataset_hash,
            created_by="train_transfer_pricing_model",
        )
        label_version_id = registry.register_label_version(
            label_key="transfer_pricing_label",
            label_version=model_version,
            entity_type="trade_record",
            label_source="pricing_curves_rule",
            positive_count=int(y.sum()),
            negative_count=int((1 - y).sum()),
            label_hash=dataset_hash,
            notes="Pseudo labels from p10/p90 envelopes.",
        )
        experiment_id = registry.ensure_experiment(
            experiment_key="transfer_pricing_learned",
            model_name="transfer_pricing",
            objective="mispricing_classification",
            owner="ml-platform",
            metadata={"pipeline": "random_forest_v1"},
        )
        run_id = registry.start_training_run(
            model_name="transfer_pricing",
            experiment_id=experiment_id,
            model_version=model_version,
            dataset_version_id=dataset_version_id,
            label_version_id=label_version_id,
            hyperparams={"n_estimators": 240, "max_depth": 8},
        )
        registry.complete_training_run(
            run_id=run_id,
            status="completed",
            metrics=report["metrics"],
            artifacts={
                "model_path": str(model_path),
                "meta_path": str(meta_path),
                "report_path": str(report_path),
            },
        )
        registry.register_rollout(
            model_name="transfer_pricing",
            model_version=model_version,
            environment="staging",
            rollout_type="shadow",
            status="planned",
            notes="Learned transfer pricing model replacing pure z-score baseline.",
            metadata={"quality_report": report},
        )
        return report


if __name__ == "__main__":
    output = train_transfer_pricing_model()
    print(json.dumps(output, indent=2))
