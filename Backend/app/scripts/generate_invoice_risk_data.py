from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

from sqlalchemy import text

from app.database import SessionLocal


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:64]


def main() -> None:
    rng = random.Random(42)
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT invoice_number, seller_tax_code, buyer_tax_code, date, amount "
                "FROM invoices WHERE invoice_number IS NOT NULL ORDER BY date DESC LIMIT 50000"
            )
        ).fetchall()

        inserted_events = 0
        inserted_fp = 0
        for inv_no, seller, buyer, inv_date, amount in rows:
            if not inv_no:
                continue
            base_key = f"{seller}|{buyer}|{int(float(amount or 0)//1_000_000)}"
            near_dup_key = base_key if rng.random() < 0.12 else f"{base_key}|{inv_no}"

            db.execute(
                text(
                    "INSERT INTO invoice_fingerprints (invoice_number, hash_near_dup, hash_line_items, hash_counterparty) "
                    "VALUES (:invoice_number, :hash_near_dup, :hash_line_items, :hash_counterparty) "
                    "ON CONFLICT (invoice_number) DO NOTHING"
                ),
                {
                    "invoice_number": inv_no,
                    "hash_near_dup": _h(near_dup_key),
                    "hash_line_items": _h(f"{inv_no}|line"),
                    "hash_counterparty": _h(f"{seller}|{buyer}"),
                },
            )
            inserted_fp += 1

            issued_time = datetime.combine(inv_date, datetime.min.time()) + timedelta(hours=rng.randint(8, 18))
            db.execute(
                text(
                    "INSERT INTO invoice_events (invoice_number, event_type, event_time, reason) "
                    "VALUES (:invoice_number, 'issued', :event_time, 'seed_issued')"
                ),
                {"invoice_number": inv_no, "event_time": issued_time},
            )
            inserted_events += 1

            if rng.random() < 0.08:
                evt_type = "canceled" if rng.random() < 0.5 else "adjusted"
                db.execute(
                    text(
                        "INSERT INTO invoice_events (invoice_number, event_type, event_time, reason) "
                        "VALUES (:invoice_number, :event_type, :event_time, :reason)"
                    ),
                    {
                        "invoice_number": inv_no,
                        "event_type": evt_type,
                        "event_time": issued_time + timedelta(days=rng.randint(1, 20)),
                        "reason": "seed_lifecycle_pattern",
                    },
                )
                inserted_events += 1

        db.commit()
        print(f"[OK] invoice_fingerprints upserted={inserted_fp} invoice_events_inserted={inserted_events}")


if __name__ == "__main__":
    main()

