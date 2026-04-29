from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/audit", tags=["Audit Selection"])


@router.get("/shortlist")
def get_audit_shortlist(
    budget_hours: float = Query(400.0, gt=0),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        text(
            "SELECT a.tax_code, a.expected_recovery, a.expected_effort, a.priority_score, a.prob_recovery, "
            "a.fusion_score, f.risk_band, f.confidence "
            "FROM audit_selection_predictions a "
            "LEFT JOIN LATERAL ("
            "    SELECT risk_band, confidence "
            "    FROM entity_risk_fusion_predictions "
            "    WHERE tax_code = a.tax_code "
            "    ORDER BY as_of_date DESC, created_at DESC "
            "    LIMIT 1"
            ") f ON TRUE "
            "ORDER BY a.priority_score DESC NULLS LAST LIMIT :limit"
        ),
        {"limit": limit},
    ).mappings().all()
    selected = []
    used_hours = 0.0
    for row in rows:
        effort = float(row.get("expected_effort") or 0.0)
        if used_hours + effort > budget_hours:
            continue
        selected.append(dict(row))
        used_hours += effort
    return {
        "budget_hours": budget_hours,
        "used_hours": round(used_hours, 2),
        "items": selected,
        "total": len(selected),
    }

