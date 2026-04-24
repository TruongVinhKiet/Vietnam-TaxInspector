from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/collections", tags=["Collections NBA"])


@router.get("/next-best-action")
def get_next_best_actions(
    tax_code: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    params = {"limit": limit}
    where = []
    if tax_code:
        params["tax_code"] = tax_code
        where.append("tax_code = :tax_code")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.execute(
        text(
            f"""
            SELECT tax_code, as_of_date, model_version, recommended_action, uplift_pp, expected_collection, confidence, reason_codes
            FROM nba_predictions
            {where_sql}
            ORDER BY expected_collection DESC NULLS LAST
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}

