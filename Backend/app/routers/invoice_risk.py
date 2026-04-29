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

