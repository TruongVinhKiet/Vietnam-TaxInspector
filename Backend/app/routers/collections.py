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
            SELECT n.tax_code, n.as_of_date, n.model_version, n.recommended_action, n.uplift_pp, n.expected_collection,
                   n.confidence, n.uncertainty_score, n.ranked_actions, n.reason_codes,
                   f.fusion_score, f.risk_band, f.driver_summary
            FROM nba_predictions n
            LEFT JOIN LATERAL (
                SELECT fusion_score, risk_band, driver_summary
                FROM entity_risk_fusion_predictions
                WHERE tax_code = n.tax_code
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT 1
            ) f ON TRUE
            {where_sql}
            ORDER BY n.expected_collection DESC NULLS LAST
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}


@router.get("/fusion-overview")
def get_fusion_overview(
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
            SELECT tax_code, as_of_date, model_version, fusion_score, risk_band, confidence, component_scores, driver_summary
            FROM entity_risk_fusion_predictions
            {where_sql}
            ORDER BY fusion_score DESC NULLS LAST
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}

