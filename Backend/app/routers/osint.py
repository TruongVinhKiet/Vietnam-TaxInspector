"""
osint.py – Open-Source Intelligence & Global UBO Discovery (DB-backed)
=======================================================================
Queries real PostgreSQL tables: ownership_links, companies, offshore_entities
to provide cross-border ownership graph, UBO discovery, and shell company analysis.

Endpoints:
    GET  /api/osint/graph/{tax_code}    – Get UBO network for a specific taxpayer
    GET  /api/osint/high-risk-ubo       – List top offshore entities by risk
    GET  /api/osint/search              – Search across OSINT graph
    GET  /api/osint/stats               – Graph-level statistics
    GET  /api/osint/countries           – List offshore jurisdictions and risk weights
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(prefix="/api/osint", tags=["OSINT & UBO Intelligence"])
TAX_CODE_PATTERN = re.compile(r"^\d{10}$")


# ────────────────────────────────────────────────────────────
#  Schemas
# ────────────────────────────────────────────────────────────

class OsintNode(BaseModel):
    id: str
    label: str
    type: str
    country: Optional[str] = None
    risk_score: Optional[float] = None
    tax_code: Optional[str] = None


class OsintEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    weight: float = 0.5


class OsintGraphResponse(BaseModel):
    center_node: Optional[OsintNode] = None
    nodes: List[OsintNode]
    edges: List[OsintEdge]
    total_connections: int
    offshore_jurisdictions: List[str]
    max_risk_score: Optional[float] = None
    data_source: str = "postgresql"


class HighRiskUBOItem(BaseModel):
    offshore_id: str
    label: str
    country: str
    risk_score: float
    connected_domestic_count: int
    relation_types: List[str]
    top_domestic_tax_codes: List[str]


class HighRiskUBOResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[HighRiskUBOItem]


class OsintStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    total_offshore_entities: int
    total_domestic_entities: int
    jurisdiction_distribution: Dict[str, int]
    relation_type_distribution: Dict[str, int]
    avg_risk_score: float
    high_risk_count: int


class JurisdictionInfo(BaseModel):
    country: str
    entity_count: int
    avg_risk_score: float
    total_connections: int


class SearchResult(BaseModel):
    node: OsintNode
    connections: int
    match_type: str


# ────────────────────────────────────────────────────────────
#  Helper: check if offshore_entities table exists
# ────────────────────────────────────────────────────────────

def _has_offshore_table(db: Session) -> bool:
    try:
        result = db.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'offshore_entities')"
        )).scalar()
        return bool(result)
    except Exception:
        return False


def _is_numeric_tax_code(value: Optional[str]) -> bool:
    if value is None:
        return False
    return bool(TAX_CODE_PATTERN.fullmatch(str(value).strip()))


def _resolve_graph_tax_code(input_code: str, db: Session) -> str:
    normalized = str(input_code).strip()
    if _is_numeric_tax_code(normalized):
        return normalized

    if _has_offshore_table(db):
        proxy = db.execute(
            text("SELECT proxy_tax_code FROM offshore_entities WHERE entity_code = :code LIMIT 1"),
            {"code": normalized},
        ).scalar()
        if _is_numeric_tax_code(proxy):
            return str(proxy)

    raise HTTPException(
        status_code=422,
        detail="tax_code phải gồm đúng 10 chữ số; hoặc cung cấp offshore entity_code hợp lệ đã có mapping proxy.",
    )


# ────────────────────────────────────────────────────────────
#  API Endpoints
# ────────────────────────────────────────────────────────────

@router.get("/stats", response_model=OsintStatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Return global statistics from ownership_links + offshore_entities tables."""
    
    total_links = db.execute(text("SELECT COUNT(*) FROM ownership_links")).scalar() or 0
    total_companies = db.execute(text("SELECT COUNT(*) FROM companies")).scalar() or 0

    # Offshore entities from table or from companies with offshore industry
    has_offshore = _has_offshore_table(db)
    if has_offshore:
        offshore_count = db.execute(text("SELECT COUNT(*) FROM offshore_entities")).scalar() or 0
        avg_risk = db.execute(text("SELECT COALESCE(AVG(risk_score), 0) FROM offshore_entities")).scalar() or 0
        high_risk = db.execute(text("SELECT COUNT(*) FROM offshore_entities WHERE risk_score >= 70")).scalar() or 0
        jur_rows = db.execute(text(
            "SELECT country, COUNT(*) as cnt FROM offshore_entities GROUP BY country ORDER BY cnt DESC"
        )).fetchall()
    else:
        offshore_count = db.execute(text(
            "SELECT COUNT(*) FROM companies WHERE industry = 'Offshore Entity'"
        )).scalar() or 0
        avg_risk = db.execute(text(
            "SELECT COALESCE(AVG(risk_score), 0) FROM companies WHERE industry = 'Offshore Entity'"
        )).scalar() or 0
        high_risk = db.execute(text(
            "SELECT COUNT(*) FROM companies WHERE industry = 'Offshore Entity' AND risk_score >= 70"
        )).scalar() or 0
        jur_rows = db.execute(text(
            "SELECT province, COUNT(*) as cnt FROM companies WHERE industry = 'Offshore Entity' GROUP BY province ORDER BY cnt DESC"
        )).fetchall()

    domestic_count = total_companies - offshore_count

    rel_rows = db.execute(text(
        "SELECT relationship_type, COUNT(*) as cnt FROM ownership_links GROUP BY relationship_type ORDER BY cnt DESC"
    )).fetchall()

    return OsintStatsResponse(
        total_nodes=total_companies,
        total_edges=total_links,
        total_offshore_entities=offshore_count,
        total_domestic_entities=domestic_count,
        jurisdiction_distribution={r[0]: r[1] for r in jur_rows if r[0]},
        relation_type_distribution={r[0]: r[1] for r in rel_rows if r[0]},
        avg_risk_score=round(float(avg_risk), 2),
        high_risk_count=high_risk,
    )


@router.get("/countries", response_model=List[JurisdictionInfo])
def list_jurisdictions(db: Session = Depends(get_db)):
    """List offshore jurisdictions with aggregated statistics from DB."""
    has_offshore = _has_offshore_table(db)
    
    if has_offshore:
        rows = db.execute(text("""
            SELECT
                oe.country,
                COUNT(DISTINCT oe.id) as entity_count,
                COALESCE(AVG(oe.risk_score), 0) as avg_risk,
                COUNT(ol.id) as total_conns
            FROM offshore_entities oe
            LEFT JOIN ownership_links ol ON ol.parent_tax_code = oe.proxy_tax_code
            GROUP BY oe.country
            ORDER BY entity_count DESC
        """)).fetchall()
    else:
        rows = db.execute(text("""
            SELECT
                c.province as country,
                COUNT(DISTINCT c.tax_code) as entity_count,
                COALESCE(AVG(c.risk_score), 0) as avg_risk,
                COUNT(ol.id) as total_conns
            FROM companies c
            LEFT JOIN ownership_links ol ON ol.parent_tax_code = c.tax_code
            WHERE c.industry = 'Offshore Entity'
            GROUP BY c.province
            ORDER BY entity_count DESC
        """)).fetchall()

    return [
        JurisdictionInfo(
            country=r[0] or "Unknown",
            entity_count=r[1],
            avg_risk_score=round(float(r[2]), 2),
            total_connections=r[3],
        )
        for r in rows
    ]


@router.get("/graph/{tax_code}", response_model=OsintGraphResponse)
def get_graph_for_tax_code(tax_code: str, depth: int = Query(default=2, ge=1, le=3), db: Session = Depends(get_db)):
    """Get the OSINT ownership graph centered on a specific tax code, queried from DB."""
    resolved_tax_code = _resolve_graph_tax_code(tax_code, db)

    # Check if company exists
    company = db.execute(text(
        "SELECT tax_code, name, industry, province, risk_score FROM companies WHERE tax_code = :tc"
    ), {"tc": resolved_tax_code}).fetchone()

    if not company:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy doanh nghiệp MST={resolved_tax_code}")

    # BFS through ownership_links
    visited = {resolved_tax_code}
    all_edges = []
    frontier = {resolved_tax_code}

    for _ in range(depth):
        if not frontier:
            break
        placeholders = ", ".join(f":tc_{i}" for i in range(len(frontier)))
        params = {f"tc_{i}": tc for i, tc in enumerate(frontier)}
        
        # Parent links (this tax_code is child)
        parent_rows = db.execute(text(f"""
            SELECT parent_tax_code, child_tax_code, relationship_type, ownership_percent
            FROM ownership_links
            WHERE child_tax_code IN ({placeholders})
        """), params).fetchall()

        # Child links (this tax_code is parent)
        child_rows = db.execute(text(f"""
            SELECT parent_tax_code, child_tax_code, relationship_type, ownership_percent
            FROM ownership_links
            WHERE parent_tax_code IN ({placeholders})
        """), params).fetchall()

        next_frontier = set()
        for row in parent_rows + child_rows:
            all_edges.append(row)
            for tc in [row[0], row[1]]:
                if tc not in visited:
                    visited.add(tc)
                    next_frontier.add(tc)
        frontier = next_frontier

    # Build node data from companies table
    nodes = []
    jurisdictions = set()
    max_risk = 0.0

    if visited:
        placeholders = ", ".join(f":tc_{i}" for i in range(len(visited)))
        params = {f"tc_{i}": tc for i, tc in enumerate(visited)}
        
        company_rows = db.execute(text(f"""
            SELECT c.tax_code, c.name, c.industry, c.province, c.risk_score, oe.entity_code
            FROM companies
            c
            LEFT JOIN offshore_entities oe ON oe.proxy_tax_code = c.tax_code
            WHERE c.tax_code IN ({placeholders})
        """), params).fetchall()

        for r in company_rows:
            is_offshore = r[2] == "Offshore Entity"
            offshore_display = r[5] if len(r) > 5 else None
            node = OsintNode(
                id=r[0],
                label=offshore_display or r[1] or r[0],
                type="offshore_entity" if is_offshore else "domestic_entity",
                country=r[3] if is_offshore else "Việt Nam",
                risk_score=float(r[4]) if r[4] else None,
                tax_code=r[0],
            )
            nodes.append(node)
            if is_offshore and r[3]:
                jurisdictions.add(r[3])
            if r[4] and float(r[4]) > max_risk:
                max_risk = float(r[4])

    # Deduplicate edges
    edge_set = set()
    unique_edges = []
    for e in all_edges:
        key = (e[0], e[1], e[2])
        if key not in edge_set:
            edge_set.add(key)
            weight = float(e[3]) / 100.0 if e[3] else 0.5
            unique_edges.append(OsintEdge(
                source=e[0], target=e[1],
                relation_type=e[2] or "shareholder",
                weight=min(1.0, weight),
            ))

    center = OsintNode(
        id=company[0], label=company[1] or company[0],
        type="domestic_entity", country="Việt Nam",
        risk_score=float(company[4]) if company[4] else None,
        tax_code=company[0],
    )

    return OsintGraphResponse(
        center_node=center,
        nodes=nodes,
        edges=unique_edges,
        total_connections=len(unique_edges),
        offshore_jurisdictions=sorted(jurisdictions),
        max_risk_score=round(max_risk, 2) if max_risk > 0 else None,
        data_source="postgresql",
    )


@router.get("/high-risk-ubo", response_model=HighRiskUBOResponse)
def list_high_risk_ubo(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=100),
    min_risk: float = Query(default=60.0, ge=0.0, le=100.0),
    country: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List offshore entities ranked by risk score from DB."""

    offset = (page - 1) * page_size
    has_offshore = _has_offshore_table(db)

    if has_offshore:
        params: Dict[str, Any] = {"min_risk": min_risk}
        where_clauses = ["COALESCE(oe.risk_score, c.risk_score, 0) >= :min_risk"]
        if country:
            where_clauses.append("oe.country = :country")
            params["country"] = country

        where_sql = " AND ".join(where_clauses)

        total = db.execute(text(f"""
            SELECT COUNT(*)
            FROM offshore_entities oe
            LEFT JOIN companies c ON c.tax_code = oe.proxy_tax_code
            WHERE {where_sql}
        """), params).scalar() or 0

        rows = db.execute(text(f"""
            SELECT
                oe.entity_code,
                oe.proxy_tax_code,
                COALESCE(oe.name, c.name, oe.entity_code) AS display_name,
                COALESCE(oe.country, c.province, 'Unknown') AS country,
                COALESCE(oe.risk_score, c.risk_score, 0) AS risk_score,
                COUNT(DISTINCT CASE WHEN ol.child_tax_code ~ '^\\d{{10}}$' THEN ol.child_tax_code END) AS domestic_count,
                ARRAY_AGG(DISTINCT ol.relationship_type) AS rel_types,
                ARRAY_AGG(DISTINCT ol.child_tax_code) AS child_codes
            FROM offshore_entities oe
            LEFT JOIN companies c ON c.tax_code = oe.proxy_tax_code
            LEFT JOIN ownership_links ol ON ol.parent_tax_code = oe.proxy_tax_code
            WHERE {where_sql}
            GROUP BY
                oe.entity_code,
                oe.proxy_tax_code,
                COALESCE(oe.name, c.name, oe.entity_code),
                COALESCE(oe.country, c.province, 'Unknown'),
                COALESCE(oe.risk_score, c.risk_score, 0)
            ORDER BY COALESCE(oe.risk_score, c.risk_score, 0) DESC, oe.entity_code ASC
            LIMIT :limit OFFSET :offset
        """), {**params, "limit": page_size, "offset": offset}).fetchall()
    else:
        params = {"min_risk": min_risk}
        where_clauses = ["c.industry = 'Offshore Entity'", "COALESCE(c.risk_score, 0) >= :min_risk"]
        if country:
            where_clauses.append("c.province = :country")
            params["country"] = country

        where_sql = " AND ".join(where_clauses)
        total = db.execute(text(f"SELECT COUNT(*) FROM companies c WHERE {where_sql}"), params).scalar() or 0

        rows = db.execute(text(f"""
            SELECT 
                c.tax_code,
                c.tax_code,
                c.name,
                c.province,
                c.risk_score,
                COUNT(DISTINCT CASE WHEN ol.child_tax_code ~ '^\\d{{10}}$' THEN ol.child_tax_code END) as domestic_count,
                ARRAY_AGG(DISTINCT ol.relationship_type) as rel_types,
                ARRAY_AGG(DISTINCT ol.child_tax_code) as child_codes
            FROM companies c
            LEFT JOIN ownership_links ol ON ol.parent_tax_code = c.tax_code
            WHERE {where_sql}
            GROUP BY c.tax_code, c.name, c.province, c.risk_score
            ORDER BY c.risk_score DESC
            LIMIT :limit OFFSET :offset
        """), {**params, "limit": page_size, "offset": offset}).fetchall()

    items = []
    for r in rows:
        rel_types = [x for x in (r[6] or []) if x] if has_offshore else [x for x in (r[5] or []) if x]
        child_raw = r[7] if has_offshore else r[6]
        child_codes = [x for x in (child_raw or []) if _is_numeric_tax_code(x)]
        items.append(HighRiskUBOItem(
            offshore_id=r[0],
            label=r[2] or r[0],
            country=r[3] or "Unknown",
            risk_score=round(float(r[4] or 0), 2),
            connected_domestic_count=r[5] or 0,
            relation_types=rel_types[:5],
            top_domestic_tax_codes=child_codes[:5],
        ))

    return HighRiskUBOResponse(total=total, page=page, page_size=page_size, items=items)


@router.get("/search", response_model=List[SearchResult])
def search_osint(
    q: str = Query(..., min_length=2, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Search companies (domestic + offshore) by name or tax_code."""
    query_like = f"%{q}%"

    has_offshore = _has_offshore_table(db)
    if has_offshore:
        rows = db.execute(text("""
            SELECT
                c.tax_code as proxy_tax_code,
                COALESCE(oe.entity_code, c.tax_code) as display_id,
                COALESCE(oe.name, c.name, c.tax_code) as display_name,
                CASE WHEN oe.id IS NOT NULL THEN 'Offshore Entity' ELSE c.industry END as industry,
                COALESCE(oe.country, c.province, 'Việt Nam') as country,
                COALESCE(oe.risk_score, c.risk_score, 0) as risk_score,
                (SELECT COUNT(*) FROM ownership_links ol
                 WHERE ol.parent_tax_code = c.tax_code OR ol.child_tax_code = c.tax_code) as conns
            FROM companies c
            LEFT JOIN offshore_entities oe ON oe.proxy_tax_code = c.tax_code
            WHERE c.tax_code ILIKE :q
               OR c.name ILIKE :q
               OR oe.entity_code ILIKE :q
               OR oe.name ILIKE :q
            ORDER BY COALESCE(oe.risk_score, c.risk_score, 0) DESC NULLS LAST
            LIMIT :lim
        """), {"q": query_like, "lim": limit}).fetchall()
    else:
        rows = db.execute(text("""
            SELECT
                c.tax_code as proxy_tax_code,
                c.tax_code as display_id,
                c.name as display_name,
                c.industry,
                c.province as country,
                c.risk_score,
                (SELECT COUNT(*) FROM ownership_links ol 
                 WHERE ol.parent_tax_code = c.tax_code OR ol.child_tax_code = c.tax_code) as conns
            FROM companies c
            WHERE c.tax_code ILIKE :q OR c.name ILIKE :q
            ORDER BY c.risk_score DESC NULLS LAST
            LIMIT :lim
        """), {"q": query_like, "lim": limit}).fetchall()

    results = []
    for r in rows:
        is_offshore = r[3] == "Offshore Entity"
        match_type = "tax_code" if q.lower() in (r[1] or "").lower() else "label"
        results.append(SearchResult(
            node=OsintNode(
                id=r[1], label=r[2] or r[1],
                type="offshore_entity" if is_offshore else "domestic_entity",
                country=r[4] if is_offshore else "Việt Nam",
                risk_score=float(r[5]) if r[5] else None,
                tax_code=r[0],
            ),
            connections=r[6] or 0,
            match_type=match_type,
        ))

    return results
