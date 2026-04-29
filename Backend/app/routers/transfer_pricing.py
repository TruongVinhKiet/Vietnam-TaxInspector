from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/transfer-pricing", tags=["Transfer Pricing"])
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"


@lru_cache(maxsize=1)
def _load_transfer_pricing_artifacts():
    model_path = MODEL_DIR / "transfer_pricing_model.joblib"
    meta_path = MODEL_DIR / "transfer_pricing_model_meta.json"
    if not model_path.exists():
        return None, {}
    model = joblib.load(model_path)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return model, meta


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
    model_obj, model_meta = _load_transfer_pricing_artifacts()
    learned_version = str(model_meta.get("model_version") or "transfer-pricing-ml-v1") if model_obj else None
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
        model_version = "transfer-pricing-baseline-v1"
        if model_obj is not None:
            feature_vec = np.asarray(
                [[
                    float(unit_price),
                    float(p10),
                    float(p50),
                    float(p90),
                    float(spread),
                    float(z_score),
                ]],
                dtype=float,
            )
            try:
                if hasattr(model_obj, "predict_proba"):
                    prob = float(model_obj.predict_proba(feature_vec)[0][1])
                else:
                    pred = float(model_obj.predict(feature_vec)[0])
                    prob = max(0.0, min(1.0, pred))
                risk_score = min(100.0, max(0.0, prob * 100.0))
                model_version = learned_version or "transfer-pricing-ml-v1"
            except Exception:
                model_version = "transfer-pricing-baseline-v1"
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
                "model_version": model_version,
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

