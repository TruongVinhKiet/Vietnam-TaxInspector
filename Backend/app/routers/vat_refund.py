from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/vat-refund", tags=["VAT Refund Risk"])


@router.get("/cases")
def list_refund_cases(
    tax_code: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    params = {"limit": limit}
    where = []
    if tax_code:
        where.append("c.tax_code = :tax_code")
        params["tax_code"] = tax_code
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    rows = db.execute(
        text(
            f"""
            SELECT c.case_id, c.tax_code, c.period, c.requested_amount, c.status, c.documents_score
            FROM vat_refund_cases c
            {where_sql}
            ORDER BY c.submitted_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}


@router.get("/cases/{case_id}/risk")
def get_refund_case_risk(
    case_id: str,
    as_of_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
):
    params = {"case_id": case_id}
    date_filter = ""
    if as_of_date:
        params["as_of_date"] = as_of_date
        date_filter = "AND p.as_of_date <= :as_of_date"

    row = db.execute(
        text(
            f"""
            SELECT p.case_id, p.as_of_date, p.model_version, p.risk_score, p.expected_loss, p.reason_codes
            FROM vat_refund_predictions p
            WHERE p.case_id = :case_id
            {date_filter}
            ORDER BY p.as_of_date DESC, p.created_at DESC
            LIMIT 1
            """
        ),
        params,
    ).mappings().first()
    if not row:
        return {"case_id": case_id, "available": False, "reason": "prediction_not_found"}
    return {"available": True, **dict(row)}

