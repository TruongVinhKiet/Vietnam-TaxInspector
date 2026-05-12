"""
tax_agent_legal_graph_reasoner.py – Legal GraphRAG Multi-hop Reasoner (Enterprise v2)
======================================================================================
Traverses the Knowledge Graph (kg_entities, kg_relations) to perform
multi-hop legal reasoning: Luật → Nghị định → Thông tư → Công văn.

Architecture:
    1. Extract anchor entities from query (tax type, document ref, article, situation)
    2. Traverse KG via relation types: contains, implements, interprets, amends,
       replaces, conflicts_with
    3. Authority ranking: Luật > Nghị định > Thông tư > Quyết định > Công văn
    4. Validate effective_from/effective_to, status, scope
    5. If citation weak → ReAct rewrite query and re-traverse

Returns:
    - relation_path: full traversal path through KG
    - authority_path: ranked chain of legal documents
    - effective_status: current validity of each document
    - citation_spans: exact article/clause references
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Authority Hierarchy ──────────────────────────────────────────────────────
AUTHORITY_RANK = {
    "luat": 1,
    "bo_luat": 1,
    "phap_lenh": 2,
    "nghi_quyet": 2,
    "nghi_dinh": 3,
    "quyet_dinh_ttg": 4,
    "thong_tu": 5,
    "thong_tu_lien_tich": 5,
    "quyet_dinh": 6,
    "cong_van": 7,
    "chi_thi": 7,
}

TRAVERSAL_RELATIONS = [
    "contains", "implements", "interprets", "amends",
    "replaces", "conflicts_with", "references", "supplements",
]

MAX_HOPS = 4
MAX_RESULTS_PER_HOP = 10


@dataclass
class LegalEntity:
    """A node in the legal knowledge graph."""
    entity_id: int
    entity_type: str      # "luat", "nghi_dinh", "thong_tu", "cong_van", "dieu_khoan"
    name: str
    doc_number: str = ""
    effective_from: date | None = None
    effective_to: date | None = None
    status: str = "active"  # active, amended, repealed, expired
    authority_rank: int = 99
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_effective(self) -> bool:
        today = date.today()
        if self.status in ("repealed", "expired"):
            return False
        if self.effective_from and today < self.effective_from:
            return False
        if self.effective_to and today > self.effective_to:
            return False
        return True


@dataclass
class LegalRelation:
    """An edge in the legal knowledge graph."""
    source_id: int
    target_id: int
    relation_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningHop:
    """A single hop in the multi-hop reasoning chain."""
    hop_number: int
    source_entity: LegalEntity
    relation: str
    target_entity: LegalEntity
    relevance_score: float = 0.0


@dataclass
class GraphReasoningResult:
    """Complete result of a GraphRAG reasoning session."""
    query: str
    anchor_entities: list[dict[str, Any]]
    reasoning_path: list[ReasoningHop] = field(default_factory=list)
    authority_chain: list[LegalEntity] = field(default_factory=list)
    citation_spans: list[dict[str, Any]] = field(default_factory=list)
    effective_status: dict[str, str] = field(default_factory=dict)
    total_hops: int = 0
    total_latency_ms: float = 0.0
    fallback_used: bool = False
    rewrite_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "anchor_entities": self.anchor_entities,
            "reasoning_path": [
                {
                    "hop": h.hop_number,
                    "source": h.source_entity.name,
                    "relation": h.relation,
                    "target": h.target_entity.name,
                    "relevance": round(h.relevance_score, 3),
                }
                for h in self.reasoning_path
            ],
            "authority_chain": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "doc_number": e.doc_number,
                    "rank": e.authority_rank,
                    "is_effective": e.is_effective,
                    "status": e.status,
                }
                for e in self.authority_chain
            ],
            "citation_spans": self.citation_spans,
            "effective_status": self.effective_status,
            "total_hops": self.total_hops,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "fallback_used": self.fallback_used,
            "rewrite_count": self.rewrite_count,
        }


class LegalGraphReasoner:
    """
    Multi-hop legal reasoning engine using the Knowledge Graph.
    
    Designed to work with PostgreSQL tables:
      - kg_entities (id, entity_type, name, properties)
      - kg_relations (id, source_id, target_id, relation_type, weight)
    
    Falls back to vector/BM25 retrieval when KG traversal yields insufficient results.
    """

    def __init__(self, db_session=None):
        self._db = db_session
        self._kg_loaded = False
        self._entity_cache: dict[int, LegalEntity] = {}
        self._relation_index: dict[int, list[LegalRelation]] = {}

    def reason(
        self,
        query: str,
        *,
        db_session=None,
        max_hops: int = MAX_HOPS,
        require_effective: bool = True,
    ) -> GraphReasoningResult:
        """
        Perform multi-hop legal reasoning for the given query.
        
        Steps:
            1. Extract anchor entities (tax type, doc reference, article)
            2. Find matching KG nodes
            3. Traverse relations up to max_hops
            4. Rank by authority hierarchy
            5. Validate effectiveness
            6. If weak → rewrite query and retry
        """
        t0 = time.perf_counter()
        db = db_session or self._db

        result = GraphReasoningResult(query=query, anchor_entities=[])

        # Step 1: Extract anchors from query
        anchors = self._extract_anchors(query)
        result.anchor_entities = anchors

        if not anchors:
            result.fallback_used = True
            result.total_latency_ms = (time.perf_counter() - t0) * 1000.0
            return result

        if not db:
            result.fallback_used = True
            result.total_latency_ms = (time.perf_counter() - t0) * 1000.0
            return result

        # Step 2: Find seed entities in KG
        seed_entities = self._find_seed_entities(db, anchors)

        if not seed_entities:
            # Try rewrite
            rewritten = self._rewrite_query(query, anchors)
            if rewritten != query:
                result.rewrite_count += 1
                new_anchors = self._extract_anchors(rewritten)
                seed_entities = self._find_seed_entities(db, new_anchors)

        if not seed_entities:
            result.fallback_used = True
            result.total_latency_ms = (time.perf_counter() - t0) * 1000.0
            return result

        # Step 3: Multi-hop traversal
        visited: set[int] = set()
        current_entities = seed_entities
        all_entities: list[LegalEntity] = list(seed_entities)

        for hop in range(max_hops):
            if not current_entities:
                break
            next_entities: list[LegalEntity] = []
            for entity in current_entities:
                if entity.entity_id in visited:
                    continue
                visited.add(entity.entity_id)

                neighbors = self._get_neighbors(db, entity.entity_id)
                for rel, target in neighbors[:MAX_RESULTS_PER_HOP]:
                    if target.entity_id in visited:
                        continue
                    if require_effective and not target.is_effective:
                        result.effective_status[target.name] = target.status
                        continue

                    relevance = self._compute_relevance(query, target, rel)
                    result.reasoning_path.append(ReasoningHop(
                        hop_number=hop + 1,
                        source_entity=entity,
                        relation=rel.relation_type,
                        target_entity=target,
                        relevance_score=relevance,
                    ))
                    next_entities.append(target)
                    all_entities.append(target)
                    result.effective_status[target.name] = target.status

            current_entities = next_entities

        result.total_hops = len(result.reasoning_path)

        # Step 4: Build authority chain (sorted by rank)
        unique_entities = {e.entity_id: e for e in all_entities}
        result.authority_chain = sorted(
            unique_entities.values(),
            key=lambda e: (e.authority_rank, e.name),
        )

        # Step 5: Extract citation spans
        result.citation_spans = self._extract_citations(result.authority_chain, query)

        result.total_latency_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "[LegalGraphReasoner] Completed: %d hops, %d entities, %d citations, %.1fms",
            result.total_hops,
            len(result.authority_chain),
            len(result.citation_spans),
            result.total_latency_ms,
        )

        return result

    # ─── Anchor Extraction ────────────────────────────────────────────────────

    def _extract_anchors(self, query: str) -> list[dict[str, Any]]:
        """Extract legal anchor entities from the query."""
        anchors: list[dict[str, Any]] = []
        q = query.lower()

        # Tax type anchors
        tax_types = {
            "thuế gtgt": "VAT", "thuế giá trị gia tăng": "VAT",
            "vat": "VAT", "gtgt": "VAT",
            "thuế tncn": "PIT", "thuế thu nhập cá nhân": "PIT",
            "tncn": "PIT", "pit": "PIT",
            "thuế tndn": "CIT", "thuế thu nhập doanh nghiệp": "CIT",
            "tndn": "CIT", "cit": "CIT",
            "thuế môn bài": "LICENSE_TAX",
            "thuế tiêu thụ đặc biệt": "SPECIAL_CONSUMPTION",
            "thuế xuất nhập khẩu": "IMPORT_EXPORT",
        }
        for pattern, tax_type in tax_types.items():
            if pattern in q:
                anchors.append({"type": "tax_type", "value": tax_type, "text": pattern})

        # Document reference anchors (e.g., "Luật 38/2019", "NĐ 126/2020")
        doc_patterns = [
            (r"luật\s*(?:số\s*)?(\d+/\d{4}(?:/QH\d+)?)", "luat"),
            (r"(?:nghị\s*định|NĐ)\s*(?:số\s*)?(\d+/\d{4}(?:/NĐ-CP)?)", "nghi_dinh"),
            (r"(?:thông\s*tư|TT)\s*(?:số\s*)?(\d+/\d{4}(?:/TT-BTC)?)", "thong_tu"),
            (r"(?:công\s*văn|CV)\s*(?:số\s*)?(\d+/\w+)", "cong_van"),
            (r"(?:quyết\s*định|QĐ)\s*(?:số\s*)?(\d+/\d{4})", "quyet_dinh"),
        ]
        for pattern, doc_type in doc_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                anchors.append({
                    "type": "document_ref",
                    "value": match.group(1),
                    "doc_type": doc_type,
                    "text": match.group(0),
                })

        # Article/Clause references (e.g., "Điều 13", "Khoản 2")
        article_patterns = [
            (r"[Đđ]iều\s*(\d+)", "article"),
            (r"[Kk]hoản\s*(\d+)", "clause"),
            (r"[Đđ]iểm\s*([a-záàảãạăắằẳẵặâấầẩẫậ])", "point"),
        ]
        for pattern, ref_type in article_patterns:
            for match in re.finditer(pattern, query):
                anchors.append({
                    "type": ref_type,
                    "value": match.group(1),
                    "text": match.group(0),
                })

        # Situation anchors
        situations = {
            "hoàn thuế": "tax_refund",
            "miễn thuế": "tax_exemption",
            "giảm thuế": "tax_reduction",
            "chuyển giá": "transfer_pricing",
            "nợ thuế": "tax_delinquency",
            "chậm nộp": "late_payment",
            "xử phạt": "penalty",
            "khiếu nại": "complaint",
            "kê khai": "declaration",
            "giảm trừ gia cảnh": "personal_deduction",
            "người phụ thuộc": "dependent",
            "hóa đơn điện tử": "e_invoice",
            "hộ kinh doanh": "household_business",
        }
        for pattern, situation in situations.items():
            if pattern in q:
                anchors.append({"type": "situation", "value": situation, "text": pattern})

        return anchors

    # ─── KG Query Methods ─────────────────────────────────────────────────────

    def _find_seed_entities(
        self, db, anchors: list[dict]
    ) -> list[LegalEntity]:
        """Find KG entities matching the extracted anchors."""
        try:
            from sqlalchemy import text

            entities: list[LegalEntity] = []
            seen_ids: set[int] = set()

            for anchor in anchors:
                if anchor["type"] == "document_ref":
                    rows = db.execute(
                        text("""
                            SELECT id, entity_type, display_name as name,
                                   attributes_json->>'doc_number' as doc_number,
                                   effective_from as eff_from,
                                   effective_to as eff_to,
                                   status
                            FROM kg_entities
                            WHERE (attributes_json->>'doc_number' ILIKE :ref
                                   OR display_name ILIKE :name_pattern)
                            LIMIT :limit
                        """),
                        {
                            "ref": f"%{anchor['value']}%",
                            "name_pattern": f"%{anchor['value']}%",
                            "limit": MAX_RESULTS_PER_HOP,
                        },
                    ).fetchall()
                elif anchor["type"] == "tax_type":
                    rows = db.execute(
                        text("""
                            SELECT id, entity_type, display_name as name,
                                   attributes_json->>'doc_number' as doc_number,
                                   effective_from as eff_from,
                                   effective_to as eff_to,
                                   status
                            FROM kg_entities
                            WHERE (display_name ILIKE :pattern
                                   OR attributes_json->>'tax_type' = :tax_type)
                            LIMIT :limit
                        """),
                        {
                            "pattern": f"%{anchor['text']}%",
                            "tax_type": anchor["value"],
                            "limit": MAX_RESULTS_PER_HOP,
                        },
                    ).fetchall()
                elif anchor["type"] == "situation":
                    rows = db.execute(
                        text("""
                            SELECT id, entity_type, display_name as name,
                                   attributes_json->>'doc_number' as doc_number,
                                   effective_from as eff_from,
                                   effective_to as eff_to,
                                   status
                            FROM kg_entities
                            WHERE display_name ILIKE :pattern
                            LIMIT :limit
                        """),
                        {"pattern": f"%{anchor['text']}%", "limit": MAX_RESULTS_PER_HOP},
                    ).fetchall()
                else:
                    continue

                for row in rows:
                    eid = row[0]
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    etype = row[1] or "unknown"
                    entities.append(LegalEntity(
                        entity_id=eid,
                        entity_type=etype,
                        name=row[2] or "",
                        doc_number=row[3] or "",
                        effective_from=self._parse_date(row[4]),
                        effective_to=self._parse_date(row[5]),
                        status=row[6] or "active",
                        authority_rank=AUTHORITY_RANK.get(etype, 99),
                    ))

            return entities

        except Exception as exc:
            if hasattr(db, "rollback"):
                try: db.rollback()
                except: pass
            logger.warning("[LegalGraphReasoner] Seed entity lookup failed: %s", exc)
            return []

    def _get_neighbors(
        self, db, entity_id: int
    ) -> list[tuple[LegalRelation, LegalEntity]]:
        """Get neighboring entities via KG relations."""
        try:
            from sqlalchemy import text

            rows = db.execute(
                text("""
                    SELECT r.id, r.source_entity_id, r.target_entity_id, r.relation_type, r.weight,
                           e.id as eid, e.entity_type, e.display_name as name,
                           e.attributes_json->>'doc_number' as doc_number,
                           e.effective_from as eff_from,
                           e.effective_to as eff_to,
                           e.status
                    FROM kg_relations r
                    JOIN kg_entities e ON (
                        CASE WHEN r.source_entity_id = :eid THEN r.target_entity_id
                             ELSE r.source_entity_id END
                    ) = e.id
                    WHERE (r.source_entity_id = :eid OR r.target_entity_id = :eid)
                      AND r.relation_type = ANY(:rel_types)
                    ORDER BY r.weight DESC
                    LIMIT :limit
                """),
                {
                    "eid": entity_id,
                    "rel_types": TRAVERSAL_RELATIONS,
                    "limit": MAX_RESULTS_PER_HOP,
                },
            ).fetchall()

            results = []
            for row in rows:
                rel = LegalRelation(
                    source_id=row[1],
                    target_id=row[2],
                    relation_type=row[3],
                    weight=float(row[4] or 1.0),
                )
                etype = row[6] or "unknown"
                entity = LegalEntity(
                    entity_id=row[5],
                    entity_type=etype,
                    name=row[7] or "",
                    doc_number=row[8] or "",
                    effective_from=self._parse_date(row[9]),
                    effective_to=self._parse_date(row[10]),
                    status=row[11] or "active",
                    authority_rank=AUTHORITY_RANK.get(etype, 99),
                )
                results.append((rel, entity))

            return results

        except Exception as exc:
            if hasattr(db, "rollback"):
                try: db.rollback()
                except: pass
            logger.warning("[LegalGraphReasoner] Neighbor lookup failed: %s", exc)
            return []

    # ─── Relevance & Citations ────────────────────────────────────────────────

    def _compute_relevance(
        self, query: str, entity: LegalEntity, relation: LegalRelation
    ) -> float:
        """Compute relevance score for a KG hop."""
        score = 0.5
        q_lower = query.lower()

        # Name match bonus
        entity_name_lower = entity.name.lower()
        if any(word in entity_name_lower for word in q_lower.split() if len(word) > 2):
            score += 0.2

        # Authority rank bonus (higher authority = more relevant)
        if entity.authority_rank <= 3:
            score += 0.15
        elif entity.authority_rank <= 5:
            score += 0.1

        # Relation weight
        score += min(0.15, relation.weight * 0.1)

        # Effectiveness bonus
        if entity.is_effective:
            score += 0.1

        return min(1.0, score)

    def _extract_citations(
        self, entities: list[LegalEntity], query: str
    ) -> list[dict[str, Any]]:
        """Extract citation spans from authority chain."""
        citations: list[dict[str, Any]] = []
        for entity in entities:
            if not entity.doc_number and not entity.name:
                continue
            citation = {
                "document": entity.name,
                "doc_number": entity.doc_number,
                "type": entity.entity_type,
                "authority_rank": entity.authority_rank,
                "is_effective": entity.is_effective,
                "status": entity.status,
            }
            if entity.effective_from:
                citation["effective_from"] = entity.effective_from.isoformat()
            if entity.effective_to:
                citation["effective_to"] = entity.effective_to.isoformat()
            citations.append(citation)

        return citations

    # ─── Query Rewrite ────────────────────────────────────────────────────────

    def _rewrite_query(self, query: str, anchors: list[dict]) -> str:
        """Rewrite query to improve KG matching when initial search fails."""
        expansions = {
            "VAT": "thuế giá trị gia tăng Luật thuế GTGT",
            "PIT": "thuế thu nhập cá nhân Luật thuế TNCN",
            "CIT": "thuế thu nhập doanh nghiệp Luật thuế TNDN",
            "tax_refund": "hoàn thuế điều kiện hoàn thuế quy trình hoàn",
            "tax_exemption": "miễn thuế ưu đãi thuế",
            "transfer_pricing": "chuyển giá giao dịch liên kết Nghị định 132",
            "late_payment": "chậm nộp thuế tiền chậm nộp lãi suất chậm nộp",
            "penalty": "xử phạt vi phạm hành chính thuế Nghị định 125",
            "personal_deduction": "giảm trừ gia cảnh người phụ thuộc 11 triệu",
            "e_invoice": "hóa đơn điện tử Nghị định 123 Thông tư 78",
        }

        added_parts = [query]
        for anchor in anchors:
            val = anchor.get("value", "")
            if val in expansions:
                added_parts.append(expansions[val])

        return " ".join(added_parts)

    # ─── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if not value:
            return None
        if isinstance(value, date):
            return value
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except Exception:
            return None
