from __future__ import annotations

import json
import random
import uuid
from datetime import date

from sqlalchemy import text

from app.database import SessionLocal


COUNTRIES = ["Singapore", "Hong Kong", "China", "Korea", "Japan", "Vietnam"]


def main() -> None:
    rng = random.Random(77)
    with SessionLocal() as db:
        companies = db.execute(text("SELECT tax_code FROM companies ORDER BY tax_code LIMIT 5000")).fetchall()
        records = 0
        curves = {}
        for (tax_code,) in companies:
            for _ in range(rng.randint(1, 4)):
                goods = rng.choice(["electronics", "steel", "agri", "software", "logistics"])
                country = rng.choice(COUNTRIES)
                qty = round(rng.uniform(10, 2000), 3)
                unit_price = round(rng.uniform(100, 50000), 2)
                trade_date = date(2025, rng.randint(1, 12), rng.randint(1, 28))
                rec_id = f"TR-{uuid.uuid4().hex[:12].upper()}"
                db.execute(
                    text(
                        "INSERT INTO trade_records (record_id, tax_code, counterparty_country, goods_category, hs_code, quantity, unit_price, trade_date, channel) "
                        "VALUES (:record_id, :tax_code, :counterparty_country, :goods_category, :hs_code, :quantity, :unit_price, :trade_date, :channel)"
                    ),
                    {
                        "record_id": rec_id,
                        "tax_code": tax_code,
                        "counterparty_country": country,
                        "goods_category": goods,
                        "hs_code": f"{rng.randint(1000,9999)}.{rng.randint(10,99)}",
                        "quantity": qty,
                        "unit_price": unit_price,
                        "trade_date": trade_date,
                        "channel": "invoice_stub",
                    },
                )
                records += 1
                key = (goods, f"VN-{country}", f"{trade_date.year}-{trade_date.month:02d}")
                curves.setdefault(key, []).append(unit_price)

        for (goods, pair, bucket), vals in curves.items():
            vals_sorted = sorted(vals)
            p10 = vals_sorted[max(0, int(len(vals_sorted) * 0.1) - 1)]
            p50 = vals_sorted[max(0, int(len(vals_sorted) * 0.5) - 1)]
            p90 = vals_sorted[max(0, int(len(vals_sorted) * 0.9) - 1)]
            curve_id = f"CURVE-{goods}-{pair}-{bucket}".replace(" ", "_")
            db.execute(
                text(
                    "INSERT INTO pricing_reference_curves (curve_id, goods_key, country_pair, time_bucket, p10, p50, p90, n_samples) "
                    "VALUES (:curve_id, :goods_key, :country_pair, :time_bucket, :p10, :p50, :p90, :n_samples) "
                    "ON CONFLICT (curve_id) DO NOTHING"
                ),
                {
                    "curve_id": curve_id,
                    "goods_key": goods,
                    "country_pair": pair,
                    "time_bucket": bucket,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "n_samples": len(vals),
                },
            )

        db.commit()
        print(json.dumps({"trade_records": records, "pricing_curves": len(curves)}))


if __name__ == "__main__":
    main()

