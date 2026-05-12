from __future__ import annotations

from datetime import date
from typing import Optional
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db
from ml_engine.model_registry import AuditContext, ModelRegistryService
from ml_engine.invoice_risk_model import InvoiceRiskScorer

router = APIRouter(prefix="/api/invoice", tags=["Invoice Risk"])


def _fetch_invoice_context(db: Session, invoice_number: str, as_of_date: date) -> tuple[dict, dict]:
    invoice_row = db.execute(
        text(
            "SELECT invoice_number, seller_tax_code, buyer_tax_code, amount, vat_rate, date, payment_status, is_adjustment "
            "FROM invoices WHERE invoice_number = :invoice_number"
        ),
        {"invoice_number": invoice_number},
    ).mappings().first()
    if not invoice_row:
        raise HTTPException(status_code=404, detail="Không tìm thấy hóa đơn.")

    event_count = db.execute(
        text(
            "SELECT COUNT(*) FROM invoice_events "
            "WHERE invoice_number = :invoice_number AND event_time::date <= :as_of_date"
        ),
        {"invoice_number": invoice_number, "as_of_date": as_of_date},
    ).scalar() or 0

    near_dup_count = db.execute(
        text(
            "SELECT COUNT(*) FROM invoice_fingerprints f "
            "JOIN invoice_fingerprints f2 ON f.hash_near_dup = f2.hash_near_dup AND f2.invoice_number <> f.invoice_number "
            "WHERE f.invoice_number = :invoice_number AND f.hash_near_dup IS NOT NULL"
        ),
        {"invoice_number": invoice_number},
    ).scalar() or 0

    same_day_pair_count = db.execute(
        text(
            "SELECT COUNT(*) FROM invoices "
            "WHERE date = :inv_date AND seller_tax_code = :seller AND buyer_tax_code = :buyer"
        ),
        {"inv_date": invoice_row["date"], "seller": invoice_row["seller_tax_code"], "buyer": invoice_row["buyer_tax_code"]},
    ).scalar() or 0

    linked_invoice_ids: list[str] = []
    dup_rows = db.execute(
        text(
            """
            SELECT f2.invoice_number
            FROM invoice_fingerprints f
            JOIN invoice_fingerprints f2
              ON f.hash_near_dup = f2.hash_near_dup
             AND f2.invoice_number <> f.invoice_number
            WHERE f.invoice_number = :invoice_number
              AND f.hash_near_dup IS NOT NULL
            LIMIT 10
            """
        ),
        {"invoice_number": invoice_number},
    ).fetchall()
    linked_invoice_ids = [str(r[0]) for r in dup_rows if r and r[0]]

    seller_risk = db.execute(
        text(
            """
            SELECT COALESCE(risk_score, 0)
            FROM ai_risk_assessments
            WHERE tax_code = :tax_code
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"tax_code": invoice_row["seller_tax_code"]},
    ).scalar() or 0
    buyer_risk = db.execute(
        text(
            """
            SELECT COALESCE(risk_score, 0)
            FROM ai_risk_assessments
            WHERE tax_code = :tax_code
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"tax_code": invoice_row["buyer_tax_code"]},
    ).scalar() or 0

    context = {
        "event_count": int(event_count),
        "near_dup_count": int(near_dup_count),
        "same_day_pair_count": int(same_day_pair_count),
        "linked_invoice_ids": linked_invoice_ids,
        "seller_risk_score": float(seller_risk or 0.0),
        "buyer_risk_score": float(buyer_risk or 0.0),
    }
    return dict(invoice_row), context


@router.get("/{invoice_number}/risk")
def get_invoice_risk(
    invoice_number: str,
    as_of_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
):
    as_of = as_of_date or date.today()
    invoice_data, context = _fetch_invoice_context(db, invoice_number, as_of)
    scorer = InvoiceRiskScorer()
    result = scorer.score(invoice_data, context)
    registry = ModelRegistryService(db)
    registry.log_inference(
        model_name="invoice_risk",
        model_version=result.model_version,
        entity_type="invoice",
        entity_id=result.invoice_number,
        input_features={**invoice_data, **context},
        outputs={"risk_score": result.risk_score, "risk_level": result.risk_level},
        ctx=AuditContext(request_id=f"invoice-{invoice_number}-{as_of.isoformat()}"),
    )

    db.execute(
        text(
            "INSERT INTO invoice_risk_predictions "
            "(invoice_number, as_of_date, model_version, risk_score, risk_level, reason_codes, explanations, linked_invoice_ids) "
            "VALUES (:invoice_number, :as_of_date, :model_version, :risk_score, :risk_level, CAST(:reason_codes AS jsonb), CAST(:explanations AS jsonb), CAST(:linked_invoice_ids AS jsonb))"
        ),
        {
            "invoice_number": result.invoice_number,
            "as_of_date": as_of,
            "model_version": result.model_version,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "reason_codes": json.dumps(result.reason_codes),
            "explanations": json.dumps(result.explanations),
            "linked_invoice_ids": json.dumps(result.linked_invoice_ids),
        },
    )
    db.commit()

    return {
        "invoice_number": result.invoice_number,
        "as_of_date": as_of.isoformat(),
        "model_version": result.model_version,
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "reason_codes": result.reason_codes,
        "explanations": result.explanations,
        "linked_invoice_ids": result.linked_invoice_ids,
    }


@router.get("/risk")
def list_invoice_risk(
    tax_code: str,
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    params = {"tax_code": tax_code, "limit": limit}
    filters = ["(i.seller_tax_code = :tax_code OR i.buyer_tax_code = :tax_code)"]
    if from_date:
        params["from_date"] = from_date
        filters.append("p.as_of_date >= :from_date")
    if to_date:
        params["to_date"] = to_date
        filters.append("p.as_of_date <= :to_date")

    where_sql = " AND ".join(filters)
    rows = db.execute(
        text(
            f"""
            SELECT p.invoice_number, p.as_of_date, p.model_version, p.risk_score, p.risk_level, p.reason_codes
            FROM invoice_risk_predictions p
            JOIN invoices i ON i.invoice_number = p.invoice_number
            WHERE {where_sql}
            ORDER BY p.risk_score DESC, p.created_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}


@router.get("/analytics/{tax_code}")
def get_invoice_analytics(tax_code: str, db: Session = Depends(get_db)):
    import datetime as _dt

    # ── 1. Time-series Control Chart ─────────────────────────────────
    # Try specific tax_code first; fall back to global top-60 days
    ts_rows = db.execute(text(
        "SELECT p.as_of_date, AVG(p.risk_score) "
        "FROM invoice_risk_predictions p "
        "JOIN invoices i ON i.invoice_number = p.invoice_number "
        "WHERE i.seller_tax_code = :tax_code OR i.buyer_tax_code = :tax_code "
        "GROUP BY p.as_of_date ORDER BY p.as_of_date ASC LIMIT 90"
    ), {"tax_code": tax_code}).fetchall()

    if not ts_rows:
        ts_rows = db.execute(text(
            "SELECT p.as_of_date, AVG(p.risk_score) "
            "FROM invoice_risk_predictions p "
            "GROUP BY p.as_of_date ORDER BY p.as_of_date ASC LIMIT 90"
        )).fetchall()

    dates = [str(r[0]) for r in ts_rows]
    scores = [round(float(r[1]), 1) for r in ts_rows]

    # ── 2. Gauge – average risk score ────────────────────────────────
    gauge_row = db.execute(text(
        "SELECT AVG(risk_score), COUNT(*) FILTER (WHERE risk_level = 'high') "
        "FROM invoice_risk_predictions LIMIT 1"
    )).first()
    avg_score = round(float(gauge_row[0]), 1) if gauge_row and gauge_row[0] else 50
    high_count = int(gauge_row[1]) if gauge_row and gauge_row[1] else 0

    # ── 3. Radar – distribution of reason codes ──────────────────────
    # reason_codes column is type `json` (a Python list after psycopg2 parse)
    # We use json_array_elements_text() which works with json type natively
    radar_rows = db.execute(text(
        "SELECT reason, COUNT(*) AS cnt FROM ("
        "  SELECT json_array_elements_text(reason_codes) AS reason "
        "  FROM invoice_risk_predictions "
        "  WHERE reason_codes IS NOT NULL LIMIT 5000"
        ") sub GROUP BY reason ORDER BY cnt DESC LIMIT 5"
    )).fetchall()

    if radar_rows:
        max_cnt = max(r[1] for r in radar_rows)
        radar_indicators = [{"name": r[0].replace("_", " ").title(), "max": 100} for r in radar_rows]
        radar_values = [round(r[1] / max_cnt * 100) for r in radar_rows]
    else:
        radar_indicators = [
            {"name": "Thời gian xuất", "max": 100}, {"name": "Mạng lưới M/B", "max": 100},
            {"name": "Giá bất thường", "max": 100}, {"name": "Tần suất", "max": 100},
            {"name": "Mặt hàng", "max": 100}
        ]
        radar_values = [80, 50, 90, 40, 60]

    # ── 4. Waterfall – score composition breakdown ───────────────────
    wf_rows = db.execute(text(
        "SELECT risk_level, COUNT(*), ROUND(AVG(risk_score)::numeric, 1) "
        "FROM invoice_risk_predictions GROUP BY risk_level ORDER BY risk_level"
    )).fetchall()
    wf_cats = [r[0] for r in wf_rows] + ["Total"]
    wf_vals = [int(r[1]) for r in wf_rows]
    wf_vals.append(sum(wf_vals))
    wf_placeholders = [0] * len(wf_vals)

    # ── 5. Treemap – reason breakdown ────────────────────────────────
    treemap_rows = db.execute(text(
        "SELECT reason, COUNT(*) AS cnt FROM ("
        "  SELECT json_array_elements_text(reason_codes) AS reason "
        "  FROM invoice_risk_predictions "
        "  WHERE reason_codes IS NOT NULL LIMIT 10000"
        ") sub GROUP BY reason ORDER BY cnt DESC LIMIT 10"
    )).fetchall()

    treemap_data = [{"name": r[0].replace("_", " ").title(), "value": r[1]} for r in treemap_rows]
    if not treemap_data:
        treemap_data = [{"name": "Không có dữ liệu", "value": 1}]

    # ── 6. Records table – latest high-risk invoices ─────────────────
    rec_rows = db.execute(text(
        "SELECT p.invoice_number, i.date, i.buyer_tax_code, i.amount, "
        "       p.risk_score, p.risk_level "
        "FROM invoice_risk_predictions p "
        "JOIN invoices i ON i.invoice_number = p.invoice_number "
        "ORDER BY p.risk_score DESC LIMIT 20"
    )).fetchall()

    records = []
    for r in rec_rows:
        records.append({
            "invoice_number": r[0],
            "date": str(r[1]) if r[1] else "",
            "buyer_name": r[2] or "",
            "amount": f"{int(r[3]):,}" if r[3] else "0",
            "risk_score": round(float(r[4])) if r[4] else 0,
            "flags": r[5] or ""
        })

    return {
        "gauge": {"score": avg_score},
        "radar": {"indicators": radar_indicators, "values": radar_values},
        "waterfall": {
            "categories": wf_cats,
            "placeholders": wf_placeholders,
            "values": wf_vals
        },
        "control": {"dates": dates, "scores": scores},
        "treemap": {"tree": treemap_data},
        "records": records
    }
