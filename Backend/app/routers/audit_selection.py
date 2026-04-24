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
            "SELECT tax_code, expected_recovery, expected_effort, priority_score, prob_recovery "
            "FROM audit_selection_predictions "
            "ORDER BY priority_score DESC NULLS LAST LIMIT :limit"
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

