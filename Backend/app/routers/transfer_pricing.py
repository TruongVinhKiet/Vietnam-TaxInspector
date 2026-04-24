from __future__ import annotations

import json
from datetime import date
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/transfer-pricing", tags=["Transfer Pricing"])


@router.post("/score")
def score_mispricing(limit: int = Query(5000, ge=1, le=50000), db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            "SELECT record_id, goods_category, counterparty_country, unit_price, trade_date "
            "FROM trade_records ORDER BY trade_date DESC LIMIT :limit"
        ),
        {"limit": limit},
    ).mappings().all()

    inserted = 0
    for r in rows:
        bucket = f"{r['trade_date'].year}-{r['trade_date'].month:02d}"
        pair = f"VN-{r['counterparty_country']}"
        curve = db.execute(
            text(
                "SELECT p10, p50, p90 FROM pricing_reference_curves "
                "WHERE goods_key=:goods_key AND country_pair=:country_pair AND time_bucket=:time_bucket LIMIT 1"
            ),
            {"goods_key": r["goods_category"], "country_pair": pair, "time_bucket": bucket},
        ).fetchone()
        if not curve:
            continue
        p10, p50, p90 = [float(v or 0) for v in curve]
        unit_price = float(r["unit_price"] or 0)
        spread = max(1.0, p90 - p10)
        z_score = (unit_price - p50) / spread
        risk_score = min(100.0, abs(z_score) * 100.0)
        reasons = []
        if unit_price > p90:
            reasons.append("price_above_p90")
        elif unit_price < p10:
            reasons.append("price_below_p10")
        db.execute(
            text(
                "INSERT INTO mispricing_predictions (record_id, as_of_date, model_version, z_score, risk_score, reason_codes) "
                "VALUES (:record_id, :as_of_date, :model_version, :z_score, :risk_score, CAST(:reason_codes AS jsonb))"
            ),
            {
                "record_id": r["record_id"],
                "as_of_date": date.today(),
                "model_version": "transfer-pricing-baseline-v1",
                "z_score": z_score,
                "risk_score": risk_score,
                "reason_codes": json.dumps(reasons),
            },
        )
        inserted += 1
    db.commit()
    return {"inserted": inserted}


@router.get("/mispricing")
def list_mispricing(
    tax_code: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    params = {"limit": limit}
    where = []
    if tax_code:
        where.append("t.tax_code = :tax_code")
        params["tax_code"] = tax_code
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.execute(
        text(
            f"""
            SELECT m.record_id, t.tax_code, t.goods_category, t.counterparty_country, t.unit_price, m.z_score, m.risk_score, m.reason_codes
            FROM mispricing_predictions m
            JOIN trade_records t ON t.record_id = m.record_id
            {where_sql}
            ORDER BY m.risk_score DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}

