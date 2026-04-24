from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db

router = APIRouter(prefix="/api/entity", tags=["Entity Resolution"])


@router.get("/resolve")
def resolve_entity_network(
    tax_code: str = Query(...),
    min_score: float = Query(0.6, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    identity = db.execute(
        text("SELECT * FROM entity_identities WHERE tax_code = :tax_code"),
        {"tax_code": tax_code},
    ).mappings().first()
    edges = db.execute(
        text(
            "SELECT src_tax_code, dst_tax_code, edge_type, score, evidence_json "
            "FROM entity_alias_edges "
            "WHERE (src_tax_code = :tax_code OR dst_tax_code = :tax_code) "
            "AND score >= :min_score "
            "ORDER BY score DESC LIMIT :limit"
        ),
        {"tax_code": tax_code, "min_score": min_score, "limit": limit},
    ).mappings().all()
    return {
        "tax_code": tax_code,
        "identity": dict(identity) if identity else None,
        "alias_edges": [dict(e) for e in edges],
        "total_edges": len(edges),
    }


@router.get("/phoenix")
def get_phoenix_candidates(
    tax_code: str = Query(...),
    min_score: float = Query(0.5, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        text(
            "SELECT old_tax_code, new_tax_code, score, signals_json, as_of_date "
            "FROM phoenix_candidates "
            "WHERE (old_tax_code = :tax_code OR new_tax_code = :tax_code) "
            "AND score >= :min_score "
            "ORDER BY score DESC LIMIT :limit"
        ),
        {"tax_code": tax_code, "min_score": min_score, "limit": limit},
    ).mappings().all()
    return {"tax_code": tax_code, "items": [dict(r) for r in rows], "total": len(rows)}

