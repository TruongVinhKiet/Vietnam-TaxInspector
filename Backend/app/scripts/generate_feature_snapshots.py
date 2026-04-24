"""
Generate leakage-safe feature snapshots (point-in-time).

Usage:
  python -m app.scripts.generate_feature_snapshots --feature-set company_core_v1 --from 2024-01-31 --to 2024-12-31 --step-months 1
"""

from __future__ import annotations

import argparse
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import SessionLocal
from ml_engine.feature_store import FeatureStore, SnapshotKey


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # keep as last day-ish: clamp by trying backwards
    for day in (d.day, 31, 30, 29, 28):
        try:
            return date(y, m, day)
        except ValueError:
            continue
    return date(y, m, 28)


def generate_company_snapshots(db: Session, feature_set_id: int, as_of: date, limit: int) -> int:
    store = FeatureStore(db)
    rows = db.execute(
        text("SELECT tax_code FROM companies ORDER BY tax_code LIMIT :limit"),
        {"limit": limit},
    ).fetchall()
    n = 0
    for (tax_code,) in rows:
        features = store.build_company_snapshot(tax_code=str(tax_code), as_of_date=as_of)
        store.upsert_snapshot(
            feature_set_id=feature_set_id,
            key=SnapshotKey(entity_type="company", entity_id=str(tax_code), as_of_date=as_of),
            features=features,
            source_payload={"entity": str(tax_code), "as_of_date": str(as_of), "kind": "company"},
        )
        n += 1
    return n


def generate_invoice_snapshots(db: Session, feature_set_id: int, as_of: date, limit: int) -> int:
    store = FeatureStore(db)
    rows = db.execute(
        text("SELECT invoice_number FROM invoices WHERE invoice_number IS NOT NULL AND date <= :as_of ORDER BY date DESC LIMIT :limit"),
        {"as_of": as_of, "limit": limit},
    ).fetchall()
    n = 0
    for (inv_no,) in rows:
        inv_no = str(inv_no)
        features = store.build_invoice_snapshot(invoice_number=inv_no, as_of_date=as_of)
        store.upsert_snapshot(
            feature_set_id=feature_set_id,
            key=SnapshotKey(entity_type="invoice", entity_id=inv_no, as_of_date=as_of),
            features=features,
            source_payload={"entity": inv_no, "as_of_date": str(as_of), "kind": "invoice"},
        )
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature snapshots (point-in-time).")
    parser.add_argument("--feature-set", default="company_core_v1")
    parser.add_argument("--owner", default="system")
    parser.add_argument("--from", dest="from_date", default="2024-01-31")
    parser.add_argument("--to", dest="to_date", default="2024-12-31")
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--company-limit", type=int, default=10000)
    parser.add_argument("--invoice-limit", type=int, default=50000)
    args = parser.parse_args()

    from_dt = date.fromisoformat(args.from_date)
    to_dt = date.fromisoformat(args.to_date)

    with SessionLocal() as db:
        store = FeatureStore(db)
        company_fs_id = store.ensure_feature_set(
            name="company_core",
            version="v1",
            owner=args.owner,
            description="Company point-in-time core features (invoices + payments).",
        )
        invoice_fs_id = store.ensure_feature_set(
            name="invoice_core",
            version="v1",
            owner=args.owner,
            description="Invoice point-in-time core features (lifecycle + enrichment).",
        )

        cursor = from_dt
        while cursor <= to_dt:
            n_company = generate_company_snapshots(db, company_fs_id, cursor, args.company_limit)
            n_invoice = generate_invoice_snapshots(db, invoice_fs_id, cursor, args.invoice_limit)
            print(f"[OK] as_of={cursor} company={n_company} invoice={n_invoice}")
            cursor = _add_months(cursor, args.step_months)


if __name__ == "__main__":
    main()

