"""
tax_agent_graphrag.py – GraphRAG Engine for Legal Knowledge Graph
==================================================================
Hybrid Graph + Vector retrieval for Vietnamese tax law reasoning.

Architecture:
    1. Knowledge Graph stored in PostgreSQL (kg_entities, kg_relations)
    2. In-memory NetworkX DiGraph for fast traversal
    3. Anchor search via embedding similarity (JSON fallback, no pgvector needed)
    4. Multi-hop graph expansion (1-3 hops along legal relations)
    5. Merged candidates fed into existing BM25 + Cross-Encoder reranker

Designed for: Core i7-8th gen, 12GB RAM, CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np

from ml_engine.tax_agent_legal_intelligence import (
    AUTHORITY_RANKS,
    canonical_doc_type,
    effective_status,
    official_letter_scope,
)

logger = logging.getLogger(__name__)

# ─── Relation type weights for graph scoring ──────────────────────────────────
# Higher weight = stronger semantic relevance during traversal
RELATION_WEIGHTS = {
    "implements":    1.0,   # NĐ hướng dẫn Luật
    "interprets":    0.95,  # TT giải thích NĐ
    "amends":        0.9,   # Sửa đổi bổ sung
    "supplements":   0.85,  # Bổ sung
    "replaces":      0.8,   # Thay thế
    "contains":      0.75,  # Luật chứa Điều
    "cites":         0.7,   # Trích dẫn
    "requires":      0.65,  # Yêu cầu tuân thủ
    "conflicts_with": 0.5,  # Mâu thuẫn
    "related_to":    0.6,   # Liên quan chung
}

@dataclass
class KGEntity:
    """A node in the knowledge graph."""
    entity_key: str
    entity_type: str
    display_name: str
    description: str = ""
    authority_rank: int = 50
    effective_from: str | None = None
    effective_to: str | None = None
    status: str = "active"
    chunk_ids: list[int] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    db_id: int | None = None


@dataclass
class KGRelation:
    """An edge in the knowledge graph."""
    source_key: str
    target_key: str
    relation_type: str
    weight: float = 1.0
    confidence: float = 0.8
    evidence_text: str = ""


@dataclass
class GraphRAGResult:
    """Result from GraphRAG retrieval."""
    chunks: list[dict]               # Enriched chunks with graph context
    subgraph: dict                   # Extracted subgraph for visualization
    traversal_path: list[str]        # Entity keys visited
    anchor_entities: list[str]       # Starting entities from vector search
    expansion_depth: int             # How many hops were traversed
    method: str = "graphrag"         # "graphrag" | "vector_fallback"
    total_entities: int = 0
    total_relations: int = 0
    communities_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    authority_path: list[dict[str, Any]] = field(default_factory=list)
    effective_status: dict[str, Any] = field(default_factory=dict)
    official_letter_scope: dict[str, Any] = field(default_factory=dict)
    relation_path: list[dict[str, Any]] = field(default_factory=list)


class KnowledgeGraphManager:
    """
    Manages the in-memory Knowledge Graph backed by PostgreSQL.

    Loads kg_entities + kg_relations into a NetworkX DiGraph.
    Provides traversal, community detection, and persistence.
    """

    def __init__(self):
        self._graph: nx.DiGraph | None = None
        self._entity_map: dict[str, dict] = {}   # entity_key → entity attrs
        self._loaded = False
        self._load_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._graph is not None

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes() if self._graph else 0

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges() if self._graph else 0

    def load_from_db(self, db) -> bool:
        """Load the knowledge graph from PostgreSQL into memory."""
        from sqlalchemy import text as sql_text

        t0 = time.perf_counter()
        try:
            # Load entities
            rows = db.execute(sql_text("""
                SELECT id, entity_key, entity_type, display_name, description,
                       authority_rank, effective_from, effective_to, status,
                       chunk_ids, attributes_json, embedding_json
                FROM kg_entities
                WHERE status != 'deleted'
                ORDER BY id
            """)).mappings().all()

            if not rows:
                logger.info("[GraphRAG] No entities found in kg_entities table")
                self._loaded = False
                return False

            G = nx.DiGraph()
            self._entity_map = {}

            for row in rows:
                key = str(row["entity_key"])
                attrs = {
                    "db_id": int(row["id"]),
                    "entity_type": canonical_doc_type(str(row["entity_type"])),
                    "display_name": str(row["display_name"]),
                    "description": str(row.get("description") or ""),
                    "authority_rank": int(row.get("authority_rank") or 50),
                    "effective_from": row.get("effective_from"),
                    "effective_to": row.get("effective_to"),
                    "status": str(row.get("status") or "active"),
                    "chunk_ids": list(row.get("chunk_ids") or []),
                    "attributes": dict(row.get("attributes_json") or {}),
                }
                attrs["effective_status"] = effective_status(
                    effective_from=attrs.get("effective_from"),
                    effective_to=attrs.get("effective_to"),
                    status=attrs.get("status"),
                )
                attrs["official_letter_scope"] = official_letter_scope(
                    doc_type=attrs.get("entity_type"),
                    title=attrs.get("display_name", ""),
                    text=attrs.get("description", ""),
                    attributes=attrs.get("attributes", {}),
                )

                # Parse embedding if available
                emb_json = row.get("embedding_json")
                if emb_json:
                    try:
                        if isinstance(emb_json, str):
                            emb_json = json.loads(emb_json)
                        attrs["embedding"] = np.array(emb_json, dtype=np.float32)
                    except Exception:
                        pass

                G.add_node(key, **attrs)
                self._entity_map[key] = attrs

            # Load relations
            rel_rows = db.execute(sql_text("""
                SELECT r.id, s.entity_key AS source_key, t.entity_key AS target_key,
                       r.relation_type, r.weight, r.confidence, r.evidence_text
                FROM kg_relations r
                JOIN kg_entities s ON s.id = r.source_entity_id
                JOIN kg_entities t ON t.id = r.target_entity_id
            """)).mappings().all()

            for rr in rel_rows:
                src = str(rr["source_key"])
                tgt = str(rr["target_key"])
                if G.has_node(src) and G.has_node(tgt):
                    G.add_edge(
                        src, tgt,
                        relation_type=str(rr["relation_type"]),
                        weight=float(rr.get("weight") or 1.0),
                        confidence=float(rr.get("confidence") or 0.8),
                        evidence_text=str(rr.get("evidence_text") or ""),
                    )

            self._graph = G
            self._loaded = True
            self._load_time = (time.perf_counter() - t0) * 1000.0

            logger.info(
                "[GraphRAG] ✓ Loaded knowledge graph: %d entities, %d relations in %.0fms",
                G.number_of_nodes(), G.number_of_edges(), self._load_time,
            )
            return True

        except Exception as exc:
            logger.warning("[GraphRAG] Failed to load graph: %s", exc)
            self._loaded = False
            return False

    def get_neighbors(
        self,
        entity_key: str,
        max_hops: int = 2,
        relation_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Get neighboring entities up to max_hops away.
        Returns list of {entity_key, entity_type, display_name, distance, path}.
        """
        if not self.is_loaded or entity_key not in self._graph:
            return []

        visited: dict[str, dict] = {}
        queue = [(entity_key, 0, [entity_key])]

        while queue:
            current, depth, path = queue.pop(0)
            if current in visited:
                continue
            if depth > max_hops:
                continue

            visited[current] = {
                "entity_key": current,
                "distance": depth,
                "path": list(path),
                **self._entity_map.get(current, {}),
            }

            if depth < max_hops:
                # Outgoing edges
                for _, neighbor, data in self._graph.out_edges(current, data=True):
                    rel_type = data.get("relation_type", "")
                    if relation_types and rel_type not in relation_types:
                        continue
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1, path + [neighbor]))

                # Incoming edges (reverse traversal)
                for neighbor, _, data in self._graph.in_edges(current, data=True):
                    rel_type = data.get("relation_type", "")
                    if relation_types and rel_type not in relation_types:
                        continue
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1, path + [neighbor]))

        # Remove the starting node itself
        visited.pop(entity_key, None)
        return sorted(visited.values(), key=lambda x: x["distance"])

    def get_subgraph(self, entity_keys: list[str]) -> dict:
        """Extract a subgraph around the given entities for visualization."""
        if not self.is_loaded:
            return {"nodes": [], "edges": []}

        nodes_set = set()
        for key in entity_keys:
            if key in self._graph:
                nodes_set.add(key)
                # Add 1-hop neighbors
                for _, n in self._graph.out_edges(key):
                    nodes_set.add(n)
                for n, _ in self._graph.in_edges(key):
                    nodes_set.add(n)

        nodes = []
        for key in nodes_set:
            attrs = self._entity_map.get(key, {})
            nodes.append({
                "id": key,
                "label": attrs.get("display_name", key),
                "type": attrs.get("entity_type", "unknown"),
                "authority_rank": attrs.get("authority_rank", 50),
                "effective_status": attrs.get("effective_status", {}),
                "official_letter_scope": attrs.get("official_letter_scope", {}),
                "status": attrs.get("status", "active"),
                "is_anchor": key in entity_keys,
            })

        edges = []
        for u, v, data in self._graph.edges(data=True):
            if u in nodes_set and v in nodes_set:
                edges.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation_type", "related_to"),
                    "weight": data.get("weight", 1.0),
                    "confidence": data.get("confidence", 0.8),
                    "evidence_text": data.get("evidence_text", ""),
                })

        return {"nodes": nodes, "edges": edges}

    def detect_communities(self) -> list[dict]:
        """Run Louvain community detection on the graph."""
        if not self.is_loaded or self.node_count < 3:
            return []

        try:
            undirected = self._graph.to_undirected()
            communities = nx.community.louvain_communities(
                undirected, resolution=1.0, seed=42,
            )

            results = []
            for i, community_nodes in enumerate(communities):
                entity_keys = list(community_nodes)
                # Generate community title from highest-authority node
                members = [
                    (k, self._entity_map.get(k, {}).get("authority_rank", 0))
                    for k in entity_keys
                ]
                members.sort(key=lambda x: x[1], reverse=True)
                top_name = self._entity_map.get(members[0][0], {}).get("display_name", "")

                results.append({
                    "community_key": f"community_{i:03d}",
                    "level": 0,
                    "title": f"Nhóm: {top_name}" if top_name else f"Nhóm {i+1}",
                    "entity_keys": entity_keys,
                    "size": len(entity_keys),
                })

            return results

        except Exception as exc:
            logger.warning("[GraphRAG] Community detection failed: %s", exc)
            return []

    def status(self) -> dict[str, Any]:
        """Return graph status."""
        return {
            "loaded": self._loaded,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "load_time_ms": round(self._load_time, 1),
        }


class GraphRAGRetriever:
    """
    Hybrid Graph+Vector retriever for the tax agent RAG pipeline.

    Flow:
        1. Embed query → find anchor entities by embedding similarity
        2. Expand anchors via graph traversal (1-2 hops)
        3. Collect chunk_ids from all traversed entities
        4. Score by: vector_similarity * 0.4 + graph_proximity * 0.3 + authority * 0.3
        5. Return enriched chunks with subgraph context
    """

    def __init__(self, graph_manager: KnowledgeGraphManager | None = None):
        self._manager = graph_manager or KnowledgeGraphManager()

    def is_available(self, db) -> bool:
        """Check if the knowledge graph has been built."""
        if not self._manager.is_loaded:
            self._manager.load_from_db(db)
        return self._manager.is_loaded and self._manager.node_count >= 3

    def retrieve(
        self,
        query: str,
        db,
        intent: str = "general_tax_query",
        top_k: int = 5,
        max_hops: int = 2,
        anchor_count: int = 5,
    ) -> GraphRAGResult:
        """
        Perform hybrid graph+vector retrieval.

        Args:
            query: User query
            db: Database session
            intent: Classified intent
            top_k: Number of final chunks to return
            max_hops: Max traversal depth
            anchor_count: Number of anchor entities from vector search

        Returns:
            GraphRAGResult with enriched chunks and subgraph
        """
        t0 = time.perf_counter()

        if not self.is_available(db):
            return GraphRAGResult(
                chunks=[], subgraph={"nodes": [], "edges": []},
                traversal_path=[], anchor_entities=[],
                expansion_depth=0, method="vector_fallback",
            )

        # Step 1: Find anchor entities via embedding similarity
        anchors = self._find_anchor_entities(query, db, anchor_count)
        if not anchors:
            return GraphRAGResult(
                chunks=[], subgraph={"nodes": [], "edges": []},
                traversal_path=[], anchor_entities=[],
                expansion_depth=0, method="vector_fallback",
            )

        anchor_keys = [a["entity_key"] for a in anchors]

        # Step 2: Expand via graph traversal
        expanded = set(anchor_keys)
        traversal_path = list(anchor_keys)

        for anchor_key in anchor_keys:
            neighbors = self._manager.get_neighbors(
                anchor_key, max_hops=max_hops,
            )
            for n in neighbors:
                nk = n["entity_key"]
                if nk not in expanded:
                    expanded.add(nk)
                    traversal_path.append(nk)

        # Step 3: Collect chunk_ids from all traversed entities
        all_chunk_ids = set()
        entity_chunk_map: dict[int, list[str]] = {}  # chunk_id → entity_keys

        for ek in expanded:
            attrs = self._manager._entity_map.get(ek, {})
            for cid in attrs.get("chunk_ids", []):
                all_chunk_ids.add(cid)
                entity_chunk_map.setdefault(cid, []).append(ek)

        # Step 4: Fetch chunk texts from DB
        chunks = []
        if all_chunk_ids:
            chunks = self._fetch_chunks(db, list(all_chunk_ids))

        # Step 5: Score chunks
        scored_chunks = []
        authority_path = self._build_authority_path(expanded)
        effective_rollup = self._rollup_effective_status(expanded)
        official_scope_rollup = self._rollup_official_letter_scope(expanded)
        relation_path = self._build_relation_path(anchor_keys, expanded)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            entity_keys_for_chunk = entity_chunk_map.get(cid, [])

            # Graph proximity score (closer to anchor = higher)
            min_distance = float("inf")
            max_authority = 0
            usable_status_boost = 0.0
            official_letter_penalty = 0.0
            strongest_relation = 0.0
            for ek in entity_keys_for_chunk:
                attrs = self._manager._entity_map.get(ek, {})
                auth = attrs.get("authority_rank", 50)
                max_authority = max(max_authority, auth)
                estatus = attrs.get("effective_status", {})
                if estatus.get("is_usable"):
                    usable_status_boost = max(usable_status_boost, 0.08)
                if attrs.get("official_letter_scope", {}).get("is_official_letter"):
                    official_letter_penalty = max(official_letter_penalty, 0.07)
                if ek in anchor_keys:
                    min_distance = 0
                else:
                    min_distance = min(min_distance, self._distance_to_anchor(ek, anchor_keys))
                strongest_relation = max(strongest_relation, self._strongest_relation_weight(ek, anchor_keys))

            graph_score = max(0, 1.0 - min_distance * 0.3)
            authority_score = max_authority / 100.0

            # Combined score
            final_score = (
                0.38 * graph_score
                + 0.28 * authority_score
                + 0.18 * strongest_relation
                + 0.16 * 0.5
                + usable_status_boost
                - official_letter_penalty
            )
            final_score = max(0.0, min(1.0, final_score))
            chunk["graph_score"] = round(final_score, 6)
            chunk["graph_entities"] = entity_keys_for_chunk
            chunk["graph_distance"] = min_distance if min_distance != float("inf") else -1
            chunk["authority_rank"] = max_authority
            chunk["authority_path"] = self._authority_path_for_entities(entity_keys_for_chunk)
            chunk["effective_status"] = self._rollup_effective_status(entity_keys_for_chunk)
            chunk["official_letter_scope"] = self._rollup_official_letter_scope(entity_keys_for_chunk)
            chunk["relation_path"] = self._build_relation_path(anchor_keys, entity_keys_for_chunk)
            scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x["graph_score"], reverse=True)
        top_chunks = scored_chunks[:top_k * 3]  # Return more for reranking

        # Step 6: Build subgraph for visualization
        subgraph = self._manager.get_subgraph(list(expanded)[:30])

        latency = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "[GraphRAG] Retrieved %d chunks via %d entities (%d anchors, %d expanded) in %.0fms",
            len(top_chunks), len(expanded), len(anchor_keys),
            len(expanded) - len(anchor_keys), latency,
        )

        return GraphRAGResult(
            chunks=top_chunks,
            subgraph=subgraph,
            traversal_path=traversal_path,
            anchor_entities=anchor_keys,
            expansion_depth=max_hops,
            method="graphrag",
            total_entities=self._manager.node_count,
            total_relations=self._manager.edge_count,
            latency_ms=latency,
            authority_path=authority_path,
            effective_status=effective_rollup,
            official_letter_scope=official_scope_rollup,
            relation_path=relation_path,
        )

    def _distance_to_anchor(self, entity_key: str, anchor_keys: list[str]) -> int:
        if not self._manager.is_loaded:
            return 3
        distances = []
        for anchor in anchor_keys:
            try:
                distances.append(nx.shortest_path_length(self._manager._graph.to_undirected(), anchor, entity_key))
            except Exception:
                pass
        return min(distances) if distances else 3

    def _strongest_relation_weight(self, entity_key: str, anchor_keys: list[str]) -> float:
        if not self._manager.is_loaded:
            return 0.0
        best = 0.0
        for anchor in anchor_keys:
            for u, v in ((anchor, entity_key), (entity_key, anchor)):
                if self._manager._graph.has_edge(u, v):
                    data = self._manager._graph.get_edge_data(u, v) or {}
                    rel_type = str(data.get("relation_type") or "related_to")
                    best = max(best, float(data.get("weight") or RELATION_WEIGHTS.get(rel_type, 0.6)))
        return best

    def _build_authority_path(self, entity_keys: set[str] | list[str]) -> list[dict[str, Any]]:
        nodes = []
        for key in entity_keys:
            attrs = self._manager._entity_map.get(key, {})
            if not attrs:
                continue
            nodes.append({
                "entity_key": key,
                "display_name": attrs.get("display_name", key),
                "entity_type": attrs.get("entity_type", "unknown"),
                "authority_rank": attrs.get("authority_rank", 50),
                "effective_status": attrs.get("effective_status", {}),
            })
        nodes.sort(key=lambda item: item.get("authority_rank", 0), reverse=True)
        return nodes[:12]

    def _authority_path_for_entities(self, entity_keys: list[str]) -> list[dict[str, Any]]:
        return self._build_authority_path(entity_keys)

    def _rollup_effective_status(self, entity_keys: set[str] | list[str]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        non_usable = []
        for key in entity_keys:
            attrs = self._manager._entity_map.get(key, {})
            status_payload = attrs.get("effective_status") or {}
            state = str(status_payload.get("state") or "unknown")
            counts[state] = counts.get(state, 0) + 1
            if state in {"expired", "repealed", "pending"}:
                non_usable.append({
                    "entity_key": key,
                    "display_name": attrs.get("display_name", key),
                    "state": state,
                })
        dominant = max(counts.items(), key=lambda item: item[1])[0] if counts else "unknown"
        return {
            "dominant_state": dominant,
            "counts": counts,
            "non_usable": non_usable[:8],
            "has_non_usable": bool(non_usable),
        }

    def _rollup_official_letter_scope(self, entity_keys: set[str] | list[str]) -> dict[str, Any]:
        letters = []
        warnings = []
        for key in entity_keys:
            attrs = self._manager._entity_map.get(key, {})
            scope = attrs.get("official_letter_scope") or {}
            if scope.get("is_official_letter"):
                letters.append({
                    "entity_key": key,
                    "display_name": attrs.get("display_name", key),
                    "scope": scope.get("scope"),
                    "binding_level": scope.get("binding_level"),
                })
                warnings.extend(scope.get("warnings") or [])
        return {
            "has_official_letter": bool(letters),
            "letters": letters[:8],
            "warnings": sorted(set(warnings))[:5],
        }

    def _build_relation_path(
        self,
        anchor_keys: list[str],
        entity_keys: set[str] | list[str],
    ) -> list[dict[str, Any]]:
        if not self._manager.is_loaded:
            return []
        graph = self._manager._graph
        paths = []
        for entity_key in list(entity_keys)[:20]:
            if entity_key in anchor_keys:
                continue
            for anchor in anchor_keys[:5]:
                try:
                    node_path = nx.shortest_path(graph.to_undirected(), anchor, entity_key)
                except Exception:
                    continue
                edges = []
                for a, b in zip(node_path, node_path[1:]):
                    data = graph.get_edge_data(a, b) or graph.get_edge_data(b, a) or {}
                    edges.append({
                        "source": a,
                        "target": b,
                        "relation": data.get("relation_type", "related_to"),
                        "weight": data.get("weight", RELATION_WEIGHTS.get(str(data.get("relation_type") or "related_to"), 0.6)),
                    })
                paths.append({
                    "anchor": anchor,
                    "target": entity_key,
                    "nodes": node_path,
                    "edges": edges,
                })
                break
        return paths[:12]

    def _find_anchor_entities(
        self,
        query: str,
        db,
        top_k: int = 5,
    ) -> list[dict]:
        """Find starting entities by embedding similarity."""
        from ml_engine.tax_agent_embeddings import get_embedding_engine

        engine = get_embedding_engine()
        query_emb = engine.embed_query(query)
        q_vec = query_emb.vector

        # Score all entities with embeddings
        scored = []
        for key, attrs in self._manager._entity_map.items():
            emb = attrs.get("embedding")
            if emb is not None and len(emb) == len(q_vec):
                sim = float(np.dot(q_vec, emb) / (
                    max(np.linalg.norm(q_vec), 1e-9) * max(np.linalg.norm(emb), 1e-9)
                ))
                scored.append({
                    "entity_key": key,
                    "similarity": sim,
                    "display_name": attrs.get("display_name", key),
                    "entity_type": attrs.get("entity_type", ""),
                    "authority_rank": attrs.get("authority_rank", 50),
                })
            else:
                # Fallback: keyword matching on display_name + description
                text = f"{attrs.get('display_name', '')} {attrs.get('description', '')}".lower()
                query_lower = query.lower()
                overlap = sum(1 for w in query_lower.split() if w in text)
                if overlap > 0:
                    scored.append({
                        "entity_key": key,
                        "similarity": min(0.5, overlap * 0.15),
                        "display_name": attrs.get("display_name", key),
                        "entity_type": attrs.get("entity_type", ""),
                        "authority_rank": attrs.get("authority_rank", 50),
                    })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def _fetch_chunks(self, db, chunk_ids: list[int]) -> list[dict]:
        """Fetch chunk texts from knowledge_chunks."""
        from sqlalchemy import text as sql_text

        if not chunk_ids:
            return []

        rows = db.execute(
            sql_text("""
                SELECT kc.id AS chunk_id, kc.chunk_key, kc.chunk_text, kc.heading,
                       kc.metadata_json, kd.title, kd.doc_type, kd.document_key,
                       kdv.version_tag
                FROM knowledge_chunks kc
                JOIN knowledge_document_versions kdv ON kdv.id = kc.version_id
                JOIN knowledge_documents kd ON kd.id = kdv.document_id
                WHERE kc.id = ANY(:ids)
            """),
            {"ids": chunk_ids},
        ).mappings().all()

        return [
            {
                "chunk_id": int(r["chunk_id"]),
                "chunk_key": str(r["chunk_key"]),
                "text": str(r.get("chunk_text") or ""),
                "heading": str(r.get("heading") or ""),
                "title": str(r.get("title") or ""),
                "doc_type": str(r.get("doc_type") or ""),
                "document_key": str(r.get("document_key") or ""),
                "version_tag": str(r.get("version_tag") or ""),
                "source": "graphrag",
            }
            for r in rows
        ]


# ─── Singletons ───────────────────────────────────────────────────────────────

_graph_manager: KnowledgeGraphManager | None = None
_retriever: GraphRAGRetriever | None = None


def get_graph_manager() -> KnowledgeGraphManager:
    """Get or create the singleton graph manager."""
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = KnowledgeGraphManager()
    return _graph_manager


def get_graphrag_retriever() -> GraphRAGRetriever:
    """Get or create the singleton GraphRAG retriever."""
    global _retriever
    if _retriever is None:
        _retriever = GraphRAGRetriever(get_graph_manager())
    return _retriever
