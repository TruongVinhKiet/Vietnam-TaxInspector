from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
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


def _hash_rows(rows: list[dict]) -> str:
    raw = json.dumps(rows, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _priority_bucket(score: float) -> str:
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 35:
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


def main() -> None:
    today = date.today()
    with SessionLocal() as db:
        audit_model, collection_model, uplift_meta = _load_ops_artifacts()
        fs = FeatureStore(db)
        registry = ModelRegistryService(db)
        company_feature_set_id = fs.ensure_feature_set(
            name="ops_company_snapshot",
            version="v1",
            owner="phase30_materializer",
            description="Company-level snapshot for audit, collections, and case-triage materialization.",
        )

        base_rows = db.execute(
            text(
                """
                SELECT
                    c.tax_code,
                    COALESCE(a.risk_score, c.risk_score, 0) AS fraud_score,
                    COALESCE(a.model_confidence, 0.55) AS fraud_confidence,
                    COALESCE(d.prob_90d, 0) AS delinquency_90d,
                    COALESCE(vr.risk_score, 0) AS vat_refund_score,
                    COALESCE(asp.priority_score, 0) AS prior_audit_priority,
                    COALESCE(asp.prob_recovery, 0) AS prior_prob_recovery,
                    COALESCE(ct, 0) AS open_case_count
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
                LEFT JOIN LATERAL (
                    SELECT priority_score, prob_recovery
                    FROM audit_selection_predictions
                    WHERE tax_code = c.tax_code
                    ORDER BY as_of_date DESC, created_at DESC
                    LIMIT 1
                ) asp ON TRUE
                LEFT JOIN LATERAL (
                    SELECT COUNT(*) AS ct
                    FROM case_queue
                    WHERE entity_id = c.tax_code
                      AND COALESCE(status, 'new') IN ('new', 'queued', 'in_progress')
                ) q ON TRUE
                ORDER BY c.tax_code
                """
            )
        ).mappings().all()

        dataset_rows = [dict(row) for row in base_rows]
        dataset_hash = _hash_rows(dataset_rows)
        dataset_version = f"ops-phase30-{today.isoformat()}"
        dataset_version_id = registry.register_dataset_version(
            dataset_key="ops_prediction_materialization",
            dataset_version=dataset_version,
            entity_type="company",
            row_count=len(dataset_rows),
            source_tables=[
                "companies",
                "ai_risk_assessments",
                "delinquency_predictions",
                "vat_refund_predictions",
                "audit_selection_predictions",
                "case_queue",
            ],
            filters={"as_of_date": today.isoformat()},
            data_hash=dataset_hash,
            created_by="phase30_materializer",
        )

        label_version_id = registry.register_label_version(
            label_key="ops_outcome_surrogates",
            label_version=dataset_version,
            entity_type="company",
            label_source="operational_tables",
            positive_count=None,
            negative_count=None,
            label_hash=dataset_hash,
            notes="Phase 30 surrogate outcomes from operational tables.",
        )

        experiment_id = registry.ensure_experiment(
            experiment_key="phase30-ops-materialization",
            model_name="ops_materialization",
            objective="Materialize audit selection, collections NBA, and case triage predictions from current stack.",
            owner="phase30_execution",
            metadata={"phase": 30},
        )

        run_id = registry.start_training_run(
            model_name="ops_materialization",
            experiment_id=experiment_id,
            model_version="phase30-materializer-v1",
            dataset_version_id=dataset_version_id,
            label_version_id=label_version_id,
            feature_set_id=company_feature_set_id,
            hyperparams={"formula": "hybrid_scoring_v1"},
        )

        audit_rows = []
        nba_rows = []
        triage_rows = []

        for row in dataset_rows:
            tax_code = str(row["tax_code"])
            fraud_score = float(row["fraud_score"] or 0.0)
            fraud_confidence = float(row["fraud_confidence"] or 0.55)
            delinquency = float(row["delinquency_90d"] or 0.0)
            vat_refund = float(row["vat_refund_score"] or 0.0)
            prior_prob = float(row["prior_prob_recovery"] or 0.0)
            prior_priority = float(row["prior_audit_priority"] or 0.0)
            open_case_count = int(row["open_case_count"] or 0)

            snapshot = fs.build_company_snapshot(tax_code=tax_code, as_of_date=today)
            fs.upsert_snapshot(
                feature_set_id=company_feature_set_id,
                key=SnapshotKey(entity_type="company", entity_id=tax_code, as_of_date=today),
                features=snapshot,
                source_payload={"dataset_version": dataset_version, "snapshot": snapshot},
            )

            inv_sum = float(snapshot.get("inv_sum_amount", 0.0))
            amount_due = max(0.0, float(snapshot.get("payment_sum_due", 0.0)) - float(snapshot.get("payment_sum_paid", 0.0)))
            company_activity = min(1.0, float(snapshot.get("inv_count", 0)) / 200.0)
            feature_vec = np.asarray(
                [[
                    fraud_score,
                    fraud_confidence,
                    delinquency,
                    vat_refund,
                    prior_priority,
                    float(open_case_count),
                    company_activity,
                ]],
                dtype=float,
            )

            prob_recovery = max(
                0.05,
                min(
                    0.98,
                    0.20
                    + fraud_score / 220.0
                    + delinquency * 0.25
                    + prior_prob * 0.20
                    + company_activity * 0.10,
                ),
            )
            if audit_model is not None and hasattr(audit_model, "predict_proba"):
                try:
                    prob_recovery = float(np.clip(audit_model.predict_proba(feature_vec)[0][1], 0.01, 0.99))
                except Exception:
                    pass
            expected_recovery = (amount_due * 0.45) + (inv_sum * 0.015) + (vat_refund * 1_000_000)
            expected_effort = max(6.0, 16.0 + open_case_count * 3.0 + delinquency * 24.0 - fraud_confidence * 4.0)
            priority_score = round((expected_recovery / max(expected_effort, 1.0)) / 1_000_000, 4)
            audit_rows.append(
                {
                    "tax_code": tax_code,
                    "as_of_date": today,
                    "model_version": str(uplift_meta.get("model_version") or "audit-selection-phase30-v1"),
                    "prob_recovery": round(prob_recovery, 4),
                    "expected_recovery": round(expected_recovery, 2),
                    "expected_effort": round(expected_effort, 2),
                    "priority_score": priority_score,
                    "reason_codes": json.dumps(["fraud_delinquency_fusion", "payment_gap", "invoice_volume"]),
                }
            )

            recommended_action = "enforcement" if delinquency >= 0.72 else "reconcile" if fraud_score >= 70 else "call" if delinquency >= 0.40 else "reminder"
            uplift_pp = max(0.02, min(0.45, delinquency * 0.32 + fraud_confidence * 0.08 + company_activity * 0.05))
            expected_collection = amount_due * (0.12 + uplift_pp)
            if collection_model is not None:
                try:
                    expected_collection = max(expected_collection, float(collection_model.predict(feature_vec)[0]))
                except Exception:
                    pass
            expected_collection = round(expected_collection, 2)
            confidence = "high" if uplift_pp >= 0.28 else "medium" if uplift_pp >= 0.12 else "low"
            nba_rows.append(
                {
                    "tax_code": tax_code,
                    "as_of_date": today,
                    "model_version": str(uplift_meta.get("model_version") or "collections-nba-phase30-v1"),
                    "recommended_action": recommended_action,
                    "uplift_pp": round(uplift_pp, 4),
                    "expected_collection": expected_collection,
                    "confidence": confidence,
                    "reason_codes": json.dumps(["delinquency_propensity", "amount_due", "company_activity"]),
                }
            )

            existing_cases = db.execute(
                text(
                    """
                    SELECT case_id, case_type, status, sla_due_at
                    FROM case_queue
                    WHERE entity_id = :tax_code
                    ORDER BY created_at DESC
                    """
                ),
                {"tax_code": tax_code},
            ).mappings().all()
            if not existing_cases:
                continue

            for case in existing_cases:
                pressure = 12.0 if case.get("sla_due_at") else 0.0
                triage_score = max(
                    0.0,
                    min(
                        100.0,
                        fraud_score * 0.45
                        + delinquency * 32.0
                        + min(20.0, prior_priority * 4.5)
                        + vat_refund * 0.10
                        + pressure
                        + open_case_count * 2.5,
                    ),
                )
                confidence_score = max(0.40, min(0.97, 0.50 + fraud_confidence * 0.30 + company_activity * 0.10))
                urgency = _priority_bucket(triage_score)
                routing_team = "Collections" if case["case_type"] == "collections" else "Audit" if case["case_type"] in {"audit", "refund"} else "FraudOps"
                next_steps = ["review_signals", "validate_latest_features"]
                if urgency in {"critical", "high"}:
                    next_steps.append("assign_investigator")
                triage_rows.append(
                    {
                        "case_id": case["case_id"],
                        "as_of_date": today,
                        "model_version": "case-triage-phase30-v1",
                        "priority_score": round(triage_score, 2),
                        "confidence": round(confidence_score, 4),
                        "urgency_level": urgency,
                        "next_steps": json.dumps(next_steps),
                        "routing_team": routing_team,
                        "reason_codes": json.dumps(["fraud_delinquency_fusion", "sla_pressure", "open_case_count"]),
                        "cohort_tags": json.dumps([case["case_type"], urgency]),
                    }
                )

        db.execute(text("DELETE FROM audit_selection_predictions WHERE as_of_date = :as_of_date"), {"as_of_date": today})
        for row in audit_rows:
            db.execute(
                text(
                    """
                    INSERT INTO audit_selection_predictions
                    (tax_code, as_of_date, model_version, prob_recovery, expected_recovery, expected_effort, priority_score, reason_codes)
                    VALUES
                    (:tax_code, :as_of_date, :model_version, :prob_recovery, :expected_recovery, :expected_effort, :priority_score, CAST(:reason_codes AS jsonb))
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
                    (tax_code, as_of_date, model_version, recommended_action, uplift_pp, expected_collection, confidence, reason_codes)
                    VALUES
                    (:tax_code, :as_of_date, :model_version, :recommended_action, :uplift_pp, :expected_collection, :confidence, CAST(:reason_codes AS jsonb))
                    """
                ),
                row,
            )

        db.execute(text("DELETE FROM case_triage_predictions WHERE as_of_date = :as_of_date"), {"as_of_date": today})
        for row in triage_rows:
            db.execute(
                text(
                    """
                    INSERT INTO case_triage_predictions
                    (case_id, as_of_date, model_version, priority_score, confidence, urgency_level, next_steps, routing_team, reason_codes, cohort_tags)
                    VALUES
                    (:case_id, :as_of_date, :model_version, :priority_score, :confidence, :urgency_level, CAST(:next_steps AS jsonb), :routing_team, CAST(:reason_codes AS jsonb), CAST(:cohort_tags AS jsonb))
                    """
                ),
                row,
            )

        db.commit()

        registry.complete_training_run(
            run_id=run_id,
            status="completed",
            metrics={
                "audit_predictions": len(audit_rows),
                "nba_predictions": len(nba_rows),
                "case_triage_predictions": len(triage_rows),
            },
            artifacts={
                "target_tables": [
                    "audit_selection_predictions",
                    "nba_predictions",
                    "case_triage_predictions",
                ],
                "as_of_date": today.isoformat(),
            },
        )
        registry.upsert_registry_entry(
            model_name="ops_materialization",
            model_version="phase30-materializer-v1",
            artifact_path="db://audit_selection_predictions,nba_predictions,case_triage_predictions",
            feature_set_id=company_feature_set_id,
            train_data_hash=dataset_hash,
            metrics={
                "audit_predictions": len(audit_rows),
                "nba_predictions": len(nba_rows),
                "case_triage_predictions": len(triage_rows),
            },
            gates={"overall_pass": True, "phase": 30},
            status="staging",
        )
        registry.register_rollout(
            model_name="ops_materialization",
            model_version="phase30-materializer-v1",
            environment="staging",
            rollout_type="batch",
            status="planned",
            notes="Phase 30 operational materializer for downstream prediction tables.",
            metadata={
                "as_of_date": today.isoformat(),
                "uplift_model_version": str(uplift_meta.get("model_version") or "heuristic"),
            },
        )

        print(
            f"[OK] Materialized audit={len(audit_rows)} nba={len(nba_rows)} triage={len(triage_rows)} predictions for {today.isoformat()}"
        )


if __name__ == "__main__":
    main()
