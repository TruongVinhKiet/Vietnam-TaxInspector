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


@router.get("/analytics")
def get_transfer_pricing_analytics(db: Session = Depends(get_db)):
    # Scatter Data: volume, price, zscore
    scatter_rows = db.execute(text(
        "SELECT t.volume, t.unit_price, m.z_score "
        "FROM trade_records t JOIN mispricing_predictions m ON t.record_id = m.record_id "
        "ORDER BY m.risk_score DESC LIMIT 500"
    )).fetchall()
    
    scatter_data = [[float(r[0] or 100), float(r[1]), float(r[2]), "Giao dịch"] for r in scatter_rows]

    # Boxplot Data
    box_rows = db.execute(text(
        "SELECT goods_key, MIN(p10), AVG(p50), MAX(p90) FROM pricing_reference_curves GROUP BY goods_key LIMIT 4"
    )).fetchall()
    
    boxplot_data = {}
    boxplot_data["categories"] = [r[0] for r in box_rows] if box_rows else ["Mặt hàng A", "Mặt hàng B", "Mặt hàng C", "Mặt hàng D"]
    boxplot_data["box_data"] = [
        [max(0, float(r[1])-10), float(r[1]), float(r[2]), float(r[3]), float(r[3])+10] for r in box_rows
    ] if box_rows else [[10,12,15,18,22], [1,1.2,1.5,1.7,2], [0.2,0.25,0.3,0.35,0.4], [40,45,50,55,65]]
    
    boxplot_data["outliers"] = [
        ["Mặt hàng A", 7], ["Mặt hàng B", 2.8], ["Mặt hàng C", 0.5]
    ]

    # Diverging Bar
    div_rows = db.execute(text(
        "SELECT t.counterparty_country, AVG(m.z_score) "
        "FROM trade_records t JOIN mispricing_predictions m ON t.record_id = m.record_id "
        "GROUP BY t.counterparty_country LIMIT 5"
    )).fetchall()
    diverging_data = {
        "categories": [r[0] for r in div_rows] if div_rows else ["Đối tác A", "Đối tác B", "Đối tác C", "Đối tác D", "Đối tác E"],
        "values": [float(r[1])*10 for r in div_rows] if div_rows else [-45.5, -30.2, 12.5, 68.9, -15.0]
    }

    # Sankey Data
    sankey_data = {
        "nodes": [
            {"name": "Công ty Mẹ (Holding)", "itemStyle": {"color": "#0f172a"}},
            {"name": "Công ty Con A (SX)", "itemStyle": {"color": "#334155"}},
            {"name": "Công ty Con B (TM)", "itemStyle": {"color": "#334155"}},
            {"name": "Đối tác Nước ngoài X", "itemStyle": {"color": "#e11d48"}},
            {"name": "Cty Sân sau (Shell)", "itemStyle": {"color": "#ea580c"}}
        ],
        "links": [
            {"source": "Công ty Con A (SX)", "target": "Công ty Mẹ (Holding)", "value": 50},
            {"source": "Công ty Con B (TM)", "target": "Công ty Mẹ (Holding)", "value": 30},
            {"source": "Công ty Mẹ (Holding)", "target": "Đối tác Nước ngoài X", "value": 45},
            {"source": "Công ty Mẹ (Holding)", "target": "Cty Sân sau (Shell)", "value": 25}
        ]
    }

    # Summary KPIs
    summary_row = db.execute(text(
        "SELECT COUNT(*) AS total, "
        "COUNT(*) FILTER (WHERE m.z_score > 2) AS anomalies, "
        "AVG(m.z_score), SUM(t.volume * t.unit_price) "
        "FROM trade_records t "
        "JOIN mispricing_predictions m ON t.record_id = m.record_id"
    )).first()
    total_vol = float(summary_row[3]) if summary_row and summary_row[3] else 0

    # Records table
    rec_rows = db.execute(text(
        "SELECT t.record_id, t.tax_code, t.goods_category, t.unit_price, m.z_score, m.risk_score "
        "FROM trade_records t JOIN mispricing_predictions m ON t.record_id = m.record_id "
        "ORDER BY m.risk_score DESC LIMIT 20"
    )).fetchall()
    records = []
    for r in rec_rows:
        records.append({
            "id": r[0], "mst": r[1] or "", "item": r[2] or "",
            "price": f"{int(r[3]):,}" if r[3] else "0",
            "zscore": round(float(r[4]), 1) if r[4] else 0,
            "risk": round(float(r[5])) if r[5] else 0
        })

    return {
        "summary": {
            "total_records": int(summary_row[0]) if summary_row else 0,
            "anomalies": int(summary_row[1]) if summary_row else 0,
            "avg_zscore": round(float(summary_row[2]), 2) if summary_row and summary_row[2] else 0.0,
            "risk_value": f"{total_vol / 1_000_000_000:.1f}T" if total_vol else "0"
        },
        "scatter": scatter_data,
        "sankey": sankey_data,
        "boxplot": boxplot_data,
        "diverging": diverging_data,
        "records": records
    }

