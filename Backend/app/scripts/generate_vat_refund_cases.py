from __future__ import annotations

import random
import uuid
from datetime import datetime

from sqlalchemy import text

from app.database import SessionLocal


def main() -> None:
    rng = random.Random(123)
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT seller_tax_code AS tax_code, date_trunc('quarter', date)::date AS q_start, COUNT(*) AS n_inv, COALESCE(SUM(amount), 0) AS total_amount "
                "FROM invoices GROUP BY seller_tax_code, date_trunc('quarter', date) "
                "ORDER BY total_amount DESC LIMIT 5000"
            )
        ).fetchall()

        case_count = 0
        link_count = 0
        for tax_code, q_start, n_inv, total_amount in rows:
            if rng.random() > 0.35:
                continue
            case_id = f"VR-{uuid.uuid4().hex[:12].upper()}"
            period = f"{q_start.year}-Q{((q_start.month - 1)//3)+1}"
            requested_amount = float(total_amount or 0) * rng.uniform(0.02, 0.18)
            documents_score = round(rng.uniform(0.35, 0.98), 3)
            db.execute(
                text(
                    "INSERT INTO vat_refund_cases (case_id, tax_code, period, requested_amount, submitted_at, status, channel, documents_score) "
                    "VALUES (:case_id, :tax_code, :period, :requested_amount, :submitted_at, :status, :channel, :documents_score) "
                    "ON CONFLICT (case_id) DO NOTHING"
                ),
                {
                    "case_id": case_id,
                    "tax_code": tax_code,
                    "period": period,
                    "requested_amount": requested_amount,
                    "submitted_at": datetime.utcnow(),
                    "status": "submitted",
                    "channel": "eportal",
                    "documents_score": documents_score,
                },
            )
            case_count += 1

            inv_rows = db.execute(
                text(
                    "SELECT invoice_number FROM invoices WHERE seller_tax_code=:tax_code OR buyer_tax_code=:tax_code "
                    "ORDER BY amount DESC LIMIT 6"
                ),
                {"tax_code": tax_code},
            ).fetchall()
            for (invoice_number,) in inv_rows:
                if not invoice_number:
                    continue
                db.execute(
                    text(
                        "INSERT INTO vat_refund_case_links (case_id, invoice_number, link_type) "
                        "VALUES (:case_id, :invoice_number, :link_type)"
                    ),
                    {"case_id": case_id, "invoice_number": invoice_number, "link_type": "supporting"},
                )
                link_count += 1

        db.commit()
        print(f"[OK] vat_refund_cases={case_count}, links={link_count}")


if __name__ == "__main__":
    main()

