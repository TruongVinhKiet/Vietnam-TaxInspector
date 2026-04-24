from __future__ import annotations

from datetime import date
import json

from sqlalchemy import text

from app.database import SessionLocal
from ml_engine.invoice_risk_model import InvoiceRiskScorer


def main() -> None:
    scorer = InvoiceRiskScorer()
    today = date.today()
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT invoice_number, seller_tax_code, buyer_tax_code, amount, vat_rate, date, payment_status, is_adjustment "
                "FROM invoices WHERE invoice_number IS NOT NULL ORDER BY date DESC LIMIT 50000"
            )
        ).mappings().all()

        inserted = 0
        for inv in rows:
            inv_no = str(inv["invoice_number"])
            ctx = db.execute(
                text(
                    "SELECT "
                    "(SELECT COUNT(*) FROM invoice_events WHERE invoice_number=:inv_no) AS event_count, "
                    "(SELECT COUNT(*) FROM invoice_fingerprints f JOIN invoice_fingerprints f2 ON f.hash_near_dup = f2.hash_near_dup "
                    "  AND f2.invoice_number <> f.invoice_number WHERE f.invoice_number=:inv_no) AS near_dup_count, "
                    "(SELECT COUNT(*) FROM invoices WHERE seller_tax_code=:seller AND buyer_tax_code=:buyer AND date=:inv_date) AS same_day_pair_count"
                ),
                {"inv_no": inv_no, "seller": inv["seller_tax_code"], "buyer": inv["buyer_tax_code"], "inv_date": inv["date"]},
            ).mappings().first()

            result = scorer.score(dict(inv), dict(ctx or {}))
            db.execute(
                text(
                    "INSERT INTO invoice_risk_predictions "
                    "(invoice_number, as_of_date, model_version, risk_score, risk_level, reason_codes, explanations, linked_invoice_ids) "
                    "VALUES (:invoice_number, :as_of_date, :model_version, :risk_score, :risk_level, CAST(:reason_codes AS jsonb), CAST(:explanations AS jsonb), CAST(:linked_invoice_ids AS jsonb))"
                ),
                {
                    "invoice_number": inv_no,
                    "as_of_date": today,
                    "model_version": result.model_version,
                    "risk_score": result.risk_score,
                    "risk_level": result.risk_level,
                    "reason_codes": json.dumps(result.reason_codes),
                    "explanations": json.dumps(result.explanations),
                    "linked_invoice_ids": json.dumps(result.linked_invoice_ids),
                },
            )
            inserted += 1

        db.commit()
        print(f"[OK] invoice risk predictions inserted={inserted}")


if __name__ == "__main__":
    main()

