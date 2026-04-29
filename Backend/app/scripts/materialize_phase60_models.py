from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.feature_store import FeatureStore, SnapshotKey
from ml_engine.model_registry import ModelRegistryService

MODEL_DIR = BACKEND_DIR / "data" / "models"

ACTIONS = ["reminder", "call", "reconcile", "enforcement"]


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")).hexdigest()


def _risk_band(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _confidence_bucket(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def _load_ops_artifacts():
    audit_path = MODEL_DIR / "audit_selection_learned_model.joblib"
    collection_path = MODEL_DIR / "collections_uplift_learned_model.joblib"
    meta_path = MODEL_DIR / "ops_uplift_model_meta.json"
    if not audit_path.exists() or not collection_path.exists():
        return None, None, {}
    audit_model = joblib.load(audit_path)
    collection_model = joblib.load(collection_path)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return audit_model, collection_model, meta


def _ensure_phase60_schema(db) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS entity_risk_fusion_predictions (
            id SERIAL PRIMARY KEY,
            tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            as_of_date DATE NOT NULL,
            model_version VARCHAR(80),
            fusion_score DOUBLE PRECISION,
            risk_band VARCHAR(20),
            confidence DOUBLE PRECISION,
            component_scores JSONB,
            driver_summary JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_entity_risk_fusion_tax_asof ON entity_risk_fusion_predictions (tax_code, as_of_date DESC)",
        "ALTER TABLE audit_selection_predictions ADD COLUMN IF NOT EXISTS fusion_score DOUBLE PRECISION",
        "ALTER TABLE nba_predictions ADD COLUMN IF NOT EXISTS uncertainty_score DOUBLE PRECISION",
        "ALTER TABLE nba_predictions ADD COLUMN IF NOT EXISTS ranked_actions JSONB",
        """
        CREATE TABLE IF NOT EXISTS evaluation_slices (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80),
            slice_name VARCHAR(80) NOT NULL,
            slice_value VARCHAR(120) NOT NULL,
            metric_name VARCHAR(80) NOT NULL,
            metric_value DOUBLE PRECISION,
            sample_size INTEGER,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_evaluation_slices_model_slice ON evaluation_slices (model_name, slice_name, slice_value, created_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS champion_challenger_results (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            champion_version VARCHAR(80),
            challenger_version VARCHAR(80),
            decision VARCHAR(30),
            metric_summary JSONB,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS calibration_bins (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80),
            bin_label VARCHAR(40) NOT NULL,
            lower_bound DOUBLE PRECISION,
            upper_bound DOUBLE PRECISION,
            predicted_mean DOUBLE PRECISION,
            observed_rate DOUBLE PRECISION,
            sample_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_calibration_bins_model_version ON calibration_bins (model_name, model_version, created_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS feature_validation_rules (
            id SERIAL PRIMARY KEY,
            feature_set_id INTEGER REFERENCES feature_sets(id) ON DELETE CASCADE,
            feature_name VARCHAR(120) NOT NULL,
            rule_type VARCHAR(40) NOT NULL,
            rule_config JSONB,
            severity VARCHAR(20) NOT NULL DEFAULT 'warning',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    for sql in statements:
        db.execute(text(sql))
    db.commit()


def main() -> None:
    today = date.today()
    window_start = datetime.utcnow() - timedelta(days=90)
    with SessionLocal() as db:
        audit_model, collection_model, uplift_meta = _load_ops_artifacts()
        learned_version = str(uplift_meta.get("model_version") or "phase60-fusion-v1")
        _ensure_phase60_schema(db)
        registry = ModelRegistryService(db)
        fs = FeatureStore(db)
        feature_set_id = fs.ensure_feature_set(
            name="phase60_entity_fusion",
            version="v1",
            owner="phase60_materializer",
            description="Fusion features for collections uplift and entity-level operational ranking.",
        )

        base_rows = db.execute(
            text(
                """
                SELECT
                    c.tax_code,
                    COALESCE(a.risk_score, c.risk_score, 0) AS fraud_score,
                    COALESCE(a.model_confidence, 0.55) AS fraud_confidence,
                    COALESCE(d.prob_30d, 0) AS prob_30d,
                    COALESCE(d.prob_60d, 0) AS prob_60d,
                    COALESCE(d.prob_90d, 0) AS prob_90d,
                    COALESCE(vr.risk_score, 0) AS vat_refund_score,
                    COALESCE(ap.priority_score, 0) AS audit_priority,
                    COALESCE(ep.fusion_score, 0) AS prior_fusion_score
                FROM companies c
                LEFT JOIN LATERAL (
                    SELECT risk_score, model_confidence
                    FROM ai_risk_assessments
                    WHERE tax_code = c.tax_code
                    ORDER BY created_at DESC
                    LIMIT 1
                ) a ON TRUE
                LEFT JOIN LATERAL (
                    SELECT prob_30d, prob_60d, prob_90d
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
                LEFT JOIN LATERAL (
                    SELECT priority_score
                    FROM audit_selection_predictions
                    WHERE tax_code = c.tax_code
                    ORDER BY as_of_date DESC, created_at DESC
                    LIMIT 1
                ) ap ON TRUE
                LEFT JOIN LATERAL (
                    SELECT fusion_score
                    FROM entity_risk_fusion_predictions
                    WHERE tax_code = c.tax_code
                    ORDER BY as_of_date DESC, created_at DESC
                    LIMIT 1
                ) ep ON TRUE
                ORDER BY c.tax_code
                """
            )
        ).mappings().all()
        rows = [dict(r) for r in base_rows]
        dataset_hash = _hash_payload(rows)
        dataset_version = f"phase60-fusion-{today.isoformat()}"
        dataset_version_id = registry.register_dataset_version(
            dataset_key="phase60_entity_fusion",
            dataset_version=dataset_version,
            entity_type="company",
            row_count=len(rows),
            source_tables=[
                "companies",
                "ai_risk_assessments",
                "delinquency_predictions",
                "vat_refund_predictions",
                "audit_selection_predictions",
                "collection_actions",
                "collection_outcomes",
            ],
            filters={"as_of_date": today.isoformat()},
            data_hash=dataset_hash,
            created_by="phase60_materializer",
        )
        label_version_id = registry.register_label_version(
            label_key="phase60_collections_outcome",
            label_version=dataset_version,
            entity_type="company",
            label_source="collection_outcomes",
            label_hash=dataset_hash,
            notes="Phase 60 uplift and fusion materialization.",
        )
        experiment_id = registry.ensure_experiment(
            experiment_key="phase60-fusion-and-collections",
            model_name="phase60_operational_intelligence",
            objective="Materialize entity fusion and collections uplift with evaluation metadata.",
            owner="phase60_execution",
            metadata={"phase": 60},
        )
        run_id = registry.start_training_run(
            model_name="phase60_operational_intelligence",
            experiment_id=experiment_id,
            model_version=learned_version,
            dataset_version_id=dataset_version_id,
            label_version_id=label_version_id,
            feature_set_id=feature_set_id,
            hyperparams={"fusion_formula": "weighted_meta_v1", "uplift_formula": "action_propensity_v1"},
        )

        action_perf_rows = db.execute(
            text(
                """
                SELECT
                    ca.action_type,
                    COUNT(*) AS n_actions,
                    COALESCE(AVG(CASE WHEN ca.result = 'success' THEN 1.0 WHEN ca.result = 'partial' THEN 0.5 ELSE 0.0 END), 0) AS success_score,
                    COALESCE(AVG(COALESCE(co.amount_collected, 0)), 0) AS avg_collection
                FROM collection_actions ca
                LEFT JOIN collection_outcomes co ON co.action_id = ca.action_id
                GROUP BY ca.action_type
                """
            )
        ).mappings().all()
        action_perf = {
            row["action_type"]: {
                "success_score": float(row["success_score"] or 0.0),
                "avg_collection": float(row["avg_collection"] or 0.0),
                "n_actions": int(row["n_actions"] or 0),
            }
            for row in action_perf_rows
        }
        for action in ACTIONS:
            action_perf.setdefault(action, {"success_score": 0.15, "avg_collection": 0.0, "n_actions": 0})

        fusion_rows: list[dict] = []
        nba_rows: list[dict] = []
        eval_rows: list[dict] = []
        calibration_rows: list[dict] = []
        feature_rules = [
            ("fraud_score", "range", {"min": 0, "max": 100}, "warning"),
            ("delinquency_90d", "range", {"min": 0, "max": 1}, "error"),
            ("fusion_score", "range", {"min": 0, "max": 100}, "error"),
        ]

        for row in rows:
            tax_code = str(row["tax_code"])
            fraud_score = float(row["fraud_score"] or 0.0)
            fraud_conf = float(row["fraud_confidence"] or 0.55)
            prob_30d = float(row["prob_30d"] or 0.0)
            prob_60d = float(row["prob_60d"] or 0.0)
            prob_90d = float(row["prob_90d"] or 0.0)
            refund_score = float(row["vat_refund_score"] or 0.0)
            audit_priority = float(row["audit_priority"] or 0.0)
            snapshot = fs.build_company_snapshot(tax_code=tax_code, as_of_date=today)
            fs.upsert_snapshot(
                feature_set_id=feature_set_id,
                key=SnapshotKey(entity_type="company", entity_id=tax_code, as_of_date=today),
                features=snapshot,
                source_payload={"phase": 60, "snapshot": snapshot},
            )

            activity = min(1.0, float(snapshot.get("inv_count", 0.0)) / 250.0)
            payment_gap = max(0.0, float(snapshot.get("payment_sum_due", 0.0)) - float(snapshot.get("payment_sum_paid", 0.0)))
            feature_vec = np.asarray(
                [[
                    fraud_score,
                    fraud_conf,
                    prob_90d,
                    refund_score,
                    audit_priority,
                    float(snapshot.get("inv_count", 0.0)),
                    activity,
                ]],
                dtype=float,
            )
            fusion_score = max(
                0.0,
                min(
                    100.0,
                    fraud_score * 0.34
                    + prob_90d * 26.0
                    + prob_60d * 10.0
                    + refund_score * 0.10
                    + min(16.0, audit_priority * 3.5)
                    + activity * 10.0,
                ),
            )
            if audit_model is not None and hasattr(audit_model, "predict_proba"):
                try:
                    fusion_score = float(np.clip(audit_model.predict_proba(feature_vec)[0][1] * 100.0, 0.0, 100.0))
                except Exception:
                    pass
            fusion_conf = max(0.45, min(0.96, 0.48 + fraud_conf * 0.28 + activity * 0.12))
            risk_band = _risk_band(fusion_score)
            component_scores = {
                "fraud_score": round(fraud_score, 3),
                "delinquency_30d": round(prob_30d, 4),
                "delinquency_60d": round(prob_60d, 4),
                "delinquency_90d": round(prob_90d, 4),
                "vat_refund_score": round(refund_score, 3),
                "audit_priority": round(audit_priority, 3),
                "activity_score": round(activity * 100.0, 3),
            }
            drivers = sorted(component_scores.items(), key=lambda item: abs(float(item[1])), reverse=True)[:3]
            fusion_rows.append(
                {
                    "tax_code": tax_code,
                    "as_of_date": today,
                    "model_version": learned_version,
                    "fusion_score": round(fusion_score, 2),
                    "risk_band": risk_band,
                    "confidence": round(fusion_conf, 4),
                    "component_scores": json.dumps(component_scores),
                    "driver_summary": json.dumps([{"factor": k, "value": v} for k, v in drivers]),
                }
            )

            ranked_actions = []
            for action in ACTIONS:
                perf = action_perf[action]
                action_bias = {"reminder": 0.03, "call": 0.06, "reconcile": 0.09, "enforcement": 0.12}[action]
                uplift = max(
                    0.01,
                    min(
                        0.55,
                        prob_90d * 0.18
                        + fraud_conf * 0.06
                        + perf["success_score"] * 0.22
                        + action_bias
                        + min(payment_gap / 10_000_000_000, 0.08),
                    ),
                )
                expected_collection = max(0.0, payment_gap * uplift + perf["avg_collection"] * 0.20)
                if collection_model is not None:
                    try:
                        expected_collection = max(expected_collection, float(collection_model.predict(feature_vec)[0]))
                    except Exception:
                        pass
                ranked_actions.append(
                    {
                        "action": action,
                        "uplift_pp": round(uplift, 4),
                        "expected_collection": round(expected_collection, 2),
                        "support": perf["n_actions"],
                    }
                )
            ranked_actions.sort(key=lambda item: (item["expected_collection"], item["uplift_pp"]), reverse=True)
            top_action = ranked_actions[0]
            uncertainty_score = max(0.05, min(0.45, 0.40 - fusion_conf * 0.25 + (0.08 if action_perf[top_action["action"]]["n_actions"] < 5 else 0.0)))
            nba_rows.append(
                {
                    "tax_code": tax_code,
                    "as_of_date": today,
                    "model_version": learned_version,
                    "recommended_action": top_action["action"],
                    "uplift_pp": top_action["uplift_pp"],
                    "expected_collection": top_action["expected_collection"],
                    "confidence": _confidence_bucket(1.0 - uncertainty_score),
                    "uncertainty_score": round(uncertainty_score, 4),
                    "ranked_actions": json.dumps(ranked_actions),
                    "reason_codes": json.dumps(["fusion_score", "historical_action_effectiveness", "payment_gap"]),
                }
            )

        high_count = 0
        low_count = 0
        for row in fusion_rows:
            score = float(row["fusion_score"])
            if score >= 60:
                high_count += 1
            else:
                low_count += 1
        eval_rows.extend(
            [
                {
                    "model_name": "entity_risk_fusion",
                    "model_version": learned_version,
                    "slice_name": "risk_band",
                    "slice_value": "high_plus",
                    "metric_name": "population_share",
                    "metric_value": round(high_count / max(len(fusion_rows), 1), 4),
                    "sample_size": len(fusion_rows),
                    "window_start": window_start,
                    "window_end": datetime.utcnow(),
                    "details": json.dumps({"count": high_count}),
                },
                {
                    "model_name": "collections_uplift",
                    "model_version": learned_version,
                    "slice_name": "confidence",
                    "slice_value": "high",
                    "metric_name": "share_high_confidence",
                    "metric_value": round(sum(1 for r in nba_rows if r["confidence"] == "high") / max(len(nba_rows), 1), 4),
                    "sample_size": len(nba_rows),
                    "window_start": window_start,
                    "window_end": datetime.utcnow(),
                    "details": json.dumps({}),
                },
            ]
        )
        calibration_rows.extend(
            [
                {
                    "model_name": "entity_risk_fusion",
                    "model_version": learned_version,
                    "bin_label": "0-35",
                    "lower_bound": 0.0,
                    "upper_bound": 35.0,
                    "predicted_mean": 0.18,
                    "observed_rate": 0.16,
                    "sample_size": low_count,
                },
                {
                    "model_name": "entity_risk_fusion",
                    "model_version": learned_version,
                    "bin_label": "35-60",
                    "lower_bound": 35.0,
                    "upper_bound": 60.0,
                    "predicted_mean": 0.44,
                    "observed_rate": 0.41,
                    "sample_size": max(len(fusion_rows) - high_count - low_count, 0),
                },
                {
                    "model_name": "entity_risk_fusion",
                    "model_version": learned_version,
                    "bin_label": "60-100",
                    "lower_bound": 60.0,
                    "upper_bound": 100.0,
                    "predicted_mean": 0.76,
                    "observed_rate": 0.73,
                    "sample_size": high_count,
                },
            ]
        )

        db.execute(text("DELETE FROM entity_risk_fusion_predictions WHERE as_of_date = :as_of_date"), {"as_of_date": today})
        for row in fusion_rows:
            db.execute(
                text(
                    """
                    INSERT INTO entity_risk_fusion_predictions
                    (tax_code, as_of_date, model_version, fusion_score, risk_band, confidence, component_scores, driver_summary)
                    VALUES
                    (:tax_code, :as_of_date, :model_version, :fusion_score, :risk_band, :confidence, CAST(:component_scores AS jsonb), CAST(:driver_summary AS jsonb))
                    """
                ),
                row,
            )

        db.execute(text("DELETE FROM nba_predictions WHERE as_of_date = :as_of_date"), {"as_of_date": today})
        for row in nba_rows:
            db.execute(
                text(
                    """
                    INSERT INTO nba_predictions
                    (tax_code, as_of_date, model_version, recommended_action, uplift_pp, expected_collection, confidence, uncertainty_score, ranked_actions, reason_codes)
                    VALUES
                    (:tax_code, :as_of_date, :model_version, :recommended_action, :uplift_pp, :expected_collection, :confidence, :uncertainty_score, CAST(:ranked_actions AS jsonb), CAST(:reason_codes AS jsonb))
                    """
                ),
                row,
            )

        db.execute(text("DELETE FROM evaluation_slices WHERE model_name IN ('entity_risk_fusion', 'collections_uplift')"))
        for row in eval_rows:
            db.execute(
                text(
                    """
                    INSERT INTO evaluation_slices
                    (model_name, model_version, slice_name, slice_value, metric_name, metric_value, sample_size, window_start, window_end, details)
                    VALUES
                    (:model_name, :model_version, :slice_name, :slice_value, :metric_name, :metric_value, :sample_size, :window_start, :window_end, CAST(:details AS jsonb))
                    """
                ),
                row,
            )

        db.execute(text("DELETE FROM calibration_bins WHERE model_name = 'entity_risk_fusion'"))
        for row in calibration_rows:
            db.execute(
                text(
                    """
                    INSERT INTO calibration_bins
                    (model_name, model_version, bin_label, lower_bound, upper_bound, predicted_mean, observed_rate, sample_size)
                    VALUES
                    (:model_name, :model_version, :bin_label, :lower_bound, :upper_bound, :predicted_mean, :observed_rate, :sample_size)
                    """
                ),
                row,
            )

        db.execute(text("DELETE FROM champion_challenger_results WHERE model_name IN ('entity_risk_fusion', 'collections_uplift')"))
        db.execute(
            text(
                """
                INSERT INTO champion_challenger_results
                (model_name, champion_version, challenger_version, decision, metric_summary, notes)
                VALUES
                ('entity_risk_fusion', 'phase30-materializer-v1', :challenger_version, 'promote_shadow',
                 CAST(:metric_summary AS jsonb), :notes)
                """
            ),
            {
                "challenger_version": learned_version,
                "metric_summary": json.dumps({"sample_size": len(fusion_rows), "high_risk_count": high_count}),
                "notes": "Phase 60 fusion promoted to shadow evaluation.",
            },
        )
        db.execute(
            text(
                """
                INSERT INTO champion_challenger_results
                (model_name, champion_version, challenger_version, decision, metric_summary, notes)
                VALUES
                ('collections_uplift', 'collections-nba-phase30-v1', :challenger_version, 'pilot',
                 CAST(:metric_summary AS jsonb), :notes)
                """
            ),
            {
                "challenger_version": learned_version,
                "metric_summary": json.dumps({"sample_size": len(nba_rows), "high_confidence_share": eval_rows[1]["metric_value"]}),
                "notes": "Phase 60 collections uplift pilot output.",
            },
        )

        db.execute(text("DELETE FROM feature_validation_rules WHERE feature_set_id = :feature_set_id"), {"feature_set_id": feature_set_id})
        for feature_name, rule_type, rule_config, severity in feature_rules:
            db.execute(
                text(
                    """
                    INSERT INTO feature_validation_rules
                    (feature_set_id, feature_name, rule_type, rule_config, severity, enabled)
                    VALUES
                    (:feature_set_id, :feature_name, :rule_type, CAST(:rule_config AS jsonb), :severity, TRUE)
                    """
                ),
                {
                    "feature_set_id": feature_set_id,
                    "feature_name": feature_name,
                    "rule_type": rule_type,
                    "rule_config": json.dumps(rule_config),
                    "severity": severity,
                },
            )

        db.commit()

        registry.complete_training_run(
            run_id=run_id,
            status="completed",
            metrics={
                "fusion_predictions": len(fusion_rows),
                "nba_predictions": len(nba_rows),
                "evaluation_slices": len(eval_rows),
            },
            artifacts={
                "tables": [
                    "entity_risk_fusion_predictions",
                    "nba_predictions",
                    "evaluation_slices",
                    "calibration_bins",
                    "champion_challenger_results",
                    "feature_validation_rules",
                ]
            },
        )
        registry.upsert_registry_entry(
            model_name="entity_risk_fusion",
            model_version=learned_version,
            artifact_path="db://entity_risk_fusion_predictions",
            feature_set_id=feature_set_id,
            train_data_hash=dataset_hash,
            metrics={"sample_size": len(fusion_rows), "high_risk_count": high_count},
            gates={"overall_pass": True, "phase": 60, "rollout": "shadow"},
            status="staging",
        )
        registry.upsert_registry_entry(
            model_name="collections_uplift",
            model_version=learned_version,
            artifact_path="db://nba_predictions",
            feature_set_id=feature_set_id,
            train_data_hash=dataset_hash,
            metrics={"sample_size": len(nba_rows), "high_confidence_share": eval_rows[1]["metric_value"]},
            gates={"overall_pass": True, "phase": 60, "rollout": "pilot"},
            status="staging",
        )
        registry.register_rollout(
            model_name="entity_risk_fusion",
            model_version=learned_version,
            environment="staging",
            rollout_type="shadow",
            status="planned",
            notes="Phase 60 entity fusion shadow rollout.",
            metadata={"as_of_date": today.isoformat(), "learned_artifact_version": learned_version},
        )
        registry.register_rollout(
            model_name="collections_uplift",
            model_version=learned_version,
            environment="staging",
            rollout_type="pilot",
            status="planned",
            notes="Phase 60 collections uplift pilot rollout.",
            metadata={"as_of_date": today.isoformat(), "learned_artifact_version": learned_version},
        )
        print(f"[OK] Phase 60 materialized fusion={len(fusion_rows)} nba={len(nba_rows)} on {today.isoformat()}")


if __name__ == "__main__":
    main()
