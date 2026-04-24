from __future__ import annotations

import json
from datetime import date

from sqlalchemy import text

from app.database import SessionLocal


def _risk_level(score: float) -> str:
    if score >= 75:
        return "critical"
    if score >= 55:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def main() -> None:
    as_of = date.today()
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT c.case_id, c.requested_amount, c.documents_score, "
                "COALESCE(COUNT(l.id), 0) AS link_count "
                "FROM vat_refund_cases c "
                "LEFT JOIN vat_refund_case_links l ON l.case_id = c.case_id "
                "GROUP BY c.case_id, c.requested_amount, c.documents_score"
            )
        ).fetchall()

        inserted = 0
        for case_id, requested_amount, documents_score, link_count in rows:
            amount = float(requested_amount or 0)
            doc = float(documents_score or 0.0)
            links = int(link_count or 0)

            score = 0.0
            reasons = []
            if amount > 1_000_000_000:
                score += 35
                reasons.append("high_requested_amount")
            if links >= 4:
                score += 20
                reasons.append("dense_invoice_linkage")
            if doc < 0.5:
                score += 25
                reasons.append("low_documents_score")
            elif doc < 0.7:
                score += 10
                reasons.append("medium_documents_score")
            if links == 0:
                score += 15
                reasons.append("missing_invoice_links")
            score = min(100.0, score)
            expected_loss = round(amount * (score / 100.0) * 0.6, 2)

            db.execute(
                text(
                    "INSERT INTO vat_refund_predictions "
                    "(case_id, as_of_date, model_version, risk_score, expected_loss, reason_codes) "
                    "VALUES (:case_id, :as_of_date, :model_version, :risk_score, :expected_loss, CAST(:reason_codes AS jsonb))"
                ),
                {
                    "case_id": case_id,
                    "as_of_date": as_of,
                    "model_version": "vat-refund-case-risk-v1",
                    "risk_score": round(score, 2),
                    "expected_loss": expected_loss,
                    "reason_codes": json.dumps(reasons),
                },
            )
            inserted += 1

        db.commit()
        print(f"[OK] vat_refund_predictions inserted={inserted}")


if __name__ == "__main__":
    main()

