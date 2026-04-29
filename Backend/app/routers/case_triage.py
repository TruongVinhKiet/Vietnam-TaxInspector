from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/case-triage", tags=["Case Triage"])


@router.get("/queue")
def get_case_queue(
    status: str | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    params = {"limit": limit}
    where = []
    if status:
        params["status"] = status
        where.append("q.status = :status")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.execute(
        text(
            f"""
            SELECT q.case_id, q.case_type, q.entity_id, q.status, q.priority, q.sla_due_at,
                   p.priority_score, p.confidence, p.urgency_level, p.next_steps, p.routing_team, p.reason_codes, p.cohort_tags,
                   f.fusion_score, f.risk_band
            FROM case_queue q
            LEFT JOIN case_triage_predictions p ON p.case_id = q.case_id
            LEFT JOIN LATERAL (
                SELECT fusion_score, risk_band
                FROM entity_risk_fusion_predictions
                WHERE tax_code = q.entity_id
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT 1
            ) f ON TRUE
            {where_sql}
            ORDER BY p.priority_score DESC NULLS LAST, q.created_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": len(rows)}

