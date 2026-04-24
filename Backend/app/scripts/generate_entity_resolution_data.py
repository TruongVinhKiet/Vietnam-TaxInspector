from __future__ import annotations

import random
import json
from datetime import date

from sqlalchemy import text

from app.database import SessionLocal


def _norm_name(name: str) -> str:
    return " ".join((name or "").lower().split())


def main() -> None:
    rng = random.Random(99)
    with SessionLocal() as db:
        companies = db.execute(
            text("SELECT tax_code, name, industry, registration_date FROM companies ORDER BY tax_code LIMIT 10000")
        ).fetchall()

        for tax_code, name, industry, reg_date in companies:
            rep_id = f"REP{int(tax_code[-4:]):04d}" if tax_code else None
            address = f"{(industry or 'Business')} Hub, Ward {int(tax_code[-2:]) % 20 if tax_code else 1}"
            db.execute(
                text(
                    "INSERT INTO entity_identities "
                    "(tax_code, legal_name, normalized_name, address, phone, email, representative_name, representative_id) "
                    "VALUES (:tax_code, :legal_name, :normalized_name, :address, :phone, :email, :representative_name, :representative_id)"
                ),
                {
                    "tax_code": tax_code,
                    "legal_name": name,
                    "normalized_name": _norm_name(name),
                    "address": address,
                    "phone": f"09{int(tax_code[-8:]) % 100000000:08d}",
                    "email": f"{tax_code}@example.vn",
                    "representative_name": f"Đại diện {tax_code[-3:]}",
                    "representative_id": rep_id,
                },
            )

        # Build synthetic alias edges
        sample_codes = [row[0] for row in companies]
        for _ in range(3000):
            a = rng.choice(sample_codes)
            b = rng.choice(sample_codes)
            if a == b:
                continue
            edge_type = rng.choice(["name_sim", "address_sim", "rep_sim", "phone_sim"])
            score = round(rng.uniform(0.55, 0.96), 3)
            db.execute(
                text(
                    "INSERT INTO entity_alias_edges (src_tax_code, dst_tax_code, edge_type, score, evidence_json) "
                    "VALUES (:src, :dst, :edge_type, :score, CAST(:evidence_json AS jsonb))"
                ),
                {
                    "src": a,
                    "dst": b,
                    "edge_type": edge_type,
                    "score": score,
                    "evidence_json": json.dumps({"seed": True, "edge_type": edge_type, "score": score}),
                },
            )

        # Build phoenix candidates
        for _ in range(1200):
            old_code = rng.choice(sample_codes)
            new_code = rng.choice(sample_codes)
            if old_code == new_code:
                continue
            score = round(rng.uniform(0.45, 0.93), 3)
            db.execute(
                text(
                    "INSERT INTO phoenix_candidates (old_tax_code, new_tax_code, score, signals_json, as_of_date) "
                    "VALUES (:old_code, :new_code, :score, CAST(:signals_json AS jsonb), :as_of_date)"
                ),
                {
                    "old_code": old_code,
                    "new_code": new_code,
                    "score": score,
                    "signals_json": json.dumps(
                        {
                            "shared_rep_indicator": round(rng.uniform(0, 1), 3),
                            "shared_address_indicator": round(rng.uniform(0, 1), 3),
                            "customer_overlap_ratio": round(rng.uniform(0, 1), 3),
                        }
                    ),
                    "as_of_date": date.today(),
                },
            )

        db.commit()
        print("[OK] entity_identities + alias_edges + phoenix_candidates generated")


if __name__ == "__main__":
    main()

