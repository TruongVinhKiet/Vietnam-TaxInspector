from __future__ import annotations

import json
import random
import uuid
from datetime import date, datetime, timedelta

from sqlalchemy import text

from app.database import SessionLocal


def main() -> None:
    rng = random.Random(314)
    today = date.today()
    with SessionLocal() as db:
        tax_codes = [r[0] for r in db.execute(text("SELECT tax_code FROM companies ORDER BY tax_code LIMIT 4000")).fetchall()]

        for tax_code in tax_codes:
            case_id = f"AUD-{uuid.uuid4().hex[:10].upper()}"
            opened = datetime.utcnow() - timedelta(days=rng.randint(10, 400))
            effort = round(rng.uniform(4, 120), 2)
            db.execute(
                text(
                    "INSERT INTO audit_cases (case_id, tax_code, opened_at, case_type, status, auditor_team, effort_hours) "
                    "VALUES (:case_id, :tax_code, :opened_at, :case_type, :status, :auditor_team, :effort_hours)"
                ),
                {
                    "case_id": case_id,
                    "tax_code": tax_code,
                    "opened_at": opened,
                    "case_type": "targeted_audit",
                    "status": "closed" if rng.random() < 0.6 else "open",
                    "auditor_team": f"Team-{rng.randint(1,8)}",
                    "effort_hours": effort,
                },
            )
            recovered = round(rng.uniform(0, 5_000_000_000), 2)
            penalty = round(recovered * rng.uniform(0.05, 0.3), 2)
            db.execute(
                text(
                    "INSERT INTO audit_outcomes (case_id, recovered_amount, penalty_amount, dispute_flag, final_amount, closing_reason) "
                    "VALUES (:case_id, :recovered_amount, :penalty_amount, :dispute_flag, :final_amount, :closing_reason)"
                ),
                {
                    "case_id": case_id,
                    "recovered_amount": recovered,
                    "penalty_amount": penalty,
                    "dispute_flag": rng.random() < 0.2,
                    "final_amount": recovered + penalty,
                    "closing_reason": "seed_outcome",
                },
            )
            prob = rng.uniform(0.05, 0.95)
            expected_recovery = recovered * prob
            expected_effort = effort * rng.uniform(0.8, 1.4)
            priority_score = (expected_recovery / max(expected_effort, 1)) / 1_000_000
            db.execute(
                text(
                    "INSERT INTO audit_selection_predictions "
                    "(tax_code, as_of_date, model_version, prob_recovery, expected_recovery, expected_effort, priority_score, reason_codes) "
                    "VALUES (:tax_code, :as_of_date, :model_version, :prob_recovery, :expected_recovery, :expected_effort, :priority_score, CAST(:reason_codes AS jsonb))"
                ),
                {
                    "tax_code": tax_code,
                    "as_of_date": today,
                    "model_version": "audit-selection-v1",
                    "prob_recovery": prob,
                    "expected_recovery": expected_recovery,
                    "expected_effort": expected_effort,
                    "priority_score": priority_score,
                    "reason_codes": json.dumps(["expected_recovery_over_effort"]),
                },
            )

            action_id = f"COL-{uuid.uuid4().hex[:10].upper()}"
            rec_action = rng.choice(["reminder", "call", "reconcile", "enforcement"])
            db.execute(
                text(
                    "INSERT INTO collection_actions (action_id, tax_code, action_type, scheduled_at, executed_at, result, notes) "
                    "VALUES (:action_id, :tax_code, :action_type, :scheduled_at, :executed_at, :result, :notes)"
                ),
                {
                    "action_id": action_id,
                    "tax_code": tax_code,
                    "action_type": rec_action,
                    "scheduled_at": datetime.utcnow(),
                    "executed_at": datetime.utcnow(),
                    "result": rng.choice(["success", "partial", "no_response"]),
                    "notes": "seed_action",
                },
            )
            collected = round(rng.uniform(0, 1_000_000_000), 2)
            db.execute(
                text(
                    "INSERT INTO collection_outcomes (tax_code, tax_period, amount_collected, collected_at, action_id) "
                    "VALUES (:tax_code, :tax_period, :amount_collected, :collected_at, :action_id)"
                ),
                {
                    "tax_code": tax_code,
                    "tax_period": f"{today.year}-Q{((today.month - 1)//3)+1}",
                    "amount_collected": collected,
                    "collected_at": datetime.utcnow(),
                    "action_id": action_id,
                },
            )
            db.execute(
                text(
                    "INSERT INTO nba_predictions "
                    "(tax_code, as_of_date, model_version, recommended_action, uplift_pp, expected_collection, confidence, reason_codes) "
                    "VALUES (:tax_code, :as_of_date, :model_version, :recommended_action, :uplift_pp, :expected_collection, :confidence, CAST(:reason_codes AS jsonb))"
                ),
                {
                    "tax_code": tax_code,
                    "as_of_date": today,
                    "model_version": "collections-nba-v1",
                    "recommended_action": rec_action,
                    "uplift_pp": round(rng.uniform(0, 0.4), 4),
                    "expected_collection": collected * rng.uniform(0.8, 1.5),
                    "confidence": rng.choice(["low", "medium", "high"]),
                    "reason_codes": json.dumps(["response_propensity", "amount_due"]),
                },
            )

            queue_case_id = f"CASE-{uuid.uuid4().hex[:10].upper()}"
            db.execute(
                text(
                    "INSERT INTO case_queue (case_id, case_type, entity_id, created_at, sla_due_at, status, priority) "
                    "VALUES (:case_id, :case_type, :entity_id, :created_at, :sla_due_at, :status, :priority)"
                ),
                {
                    "case_id": queue_case_id,
                    "case_type": rng.choice(["invoice", "refund", "audit", "collections"]),
                    "entity_id": tax_code,
                    "created_at": datetime.utcnow(),
                    "sla_due_at": datetime.utcnow() + timedelta(days=rng.randint(1, 30)),
                    "status": rng.choice(["new", "in_progress", "queued"]),
                    "priority": rng.choice(["low", "medium", "high"]),
                },
            )
            db.execute(
                text(
                    "INSERT INTO case_events (case_id, event_type, actor, payload_json) "
                    "VALUES (:case_id, :event_type, :actor, CAST(:payload_json AS jsonb))"
                ),
                {
                    "case_id": queue_case_id,
                    "event_type": "created",
                    "actor": "system_seed",
                    "payload_json": json.dumps({"seed": True}),
                },
            )
            pr_score = round(rng.uniform(0, 100), 2)
            urgency = "critical" if pr_score > 85 else "high" if pr_score > 65 else "medium" if pr_score > 35 else "low"
            db.execute(
                text(
                    "INSERT INTO case_triage_predictions "
                    "(case_id, as_of_date, model_version, priority_score, urgency_level, next_steps, routing_team, reason_codes) "
                    "VALUES (:case_id, :as_of_date, :model_version, :priority_score, :urgency_level, CAST(:next_steps AS jsonb), :routing_team, CAST(:reason_codes AS jsonb))"
                ),
                {
                    "case_id": queue_case_id,
                    "as_of_date": today,
                    "model_version": "case-triage-v1",
                    "priority_score": pr_score,
                    "urgency_level": urgency,
                    "next_steps": json.dumps(["review_signals", "assign_investigator"]),
                    "routing_team": f"Ops-{rng.randint(1,6)}",
                    "reason_codes": json.dumps(["sla_pressure", "risk_score"]),
                },
            )

        db.commit()
        print("[OK] Generated audit/collections/case-triage seeds + predictions")


if __name__ == "__main__":
    main()

