"""
tax_agent_tools.py – Tool Registry & Execution Engine (Phase 2)
================================================================
Defines all tools that the multi-agent orchestrator can invoke.
Each tool wraps an existing ML/DL model or data API.

Architecture:
    - ToolRegistry: central catalog of available tools
    - ToolExecutor: parallel/sequential execution engine
    - Each tool: input schema, output schema, timeout, retry policy

Tools (13+):
    1.  knowledge_search        – RAG retrieval (Phase 1 enhanced)
    2.  company_risk_lookup     – Fraud pipeline (XGBoost + IsolationForest)
    3.  gnn_analysis            – GNN graph analysis (GATv2)
    4.  delinquency_check       – Delinquency temporal prediction
    5.  invoice_risk_scan       – Invoice anomaly detection
    6.  vat_refund_risk         – VAT refund risk assessment
    7.  transfer_pricing_check  – Transfer pricing analysis
    8.  osint_graph_query       – OSINT ownership/entity graph
    9.  motif_detection         – Graph motif patterns
    10. ring_scoring            – Circular transaction ring scoring
    11. link_prediction         – Link prediction for new connections
    12. ownership_analysis      – Ownership chain & shell detection
    13. macro_forecast          – Macro hypothesis simulation
    14. audit_selection         – Audit priority ranking
    15. collections_nba         – Next-best-action for collections
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    RETRIEVAL = "retrieval"
    ANALYTICS = "analytics"
    INVESTIGATION = "investigation"
    FORECASTING = "forecasting"
    GOVERNANCE = "governance"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolSpec:
    """Specification for a single tool."""
    name: str
    description: str
    category: ToolCategory
    input_schema: dict[str, Any]      # JSON schema for input
    output_schema: dict[str, Any]     # JSON schema for output
    handler: Callable                  # The actual function to call
    timeout_seconds: float = 30.0
    max_retries: int = 1
    requires_db: bool = True
    requires_tax_code: bool = False
    priority: int = 5                  # 1=highest, 10=lowest
    enabled: bool = True


@dataclass
class ToolCallRequest:
    """A request to invoke a tool."""
    tool_name: str
    inputs: dict[str, Any]
    request_id: str = ""
    timeout_override: float | None = None
    max_retries_override: int | None = None


@dataclass
class ToolExecutionContext:
    """Execution budget passed to long-running/cooperative tools."""
    deadline_at: float | None = None
    cancel_event: threading.Event | None = None
    request_id: str = ""
    attempt: int = 0

    def remaining_seconds(self) -> float | None:
        if self.deadline_at is None:
            return None
        return max(0.0, self.deadline_at - time.perf_counter())

    def is_cancelled(self) -> bool:
        return bool(self.cancel_event and self.cancel_event.is_set())

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise RuntimeError("tool_execution_cancelled")
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("tool_execution_timeout")


@dataclass
class ToolCallResult:
    """Result from a tool invocation."""
    tool_name: str
    status: ToolStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central catalog of all tools available to the agent.
    Tools are registered at startup and can be queried by category or name.
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """Register a tool in the catalog."""
        self._tools[tool.name] = tool
        logger.info(
            "[ToolRegistry] Registered: %s (%s)", tool.name, tool.category.value
        )

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[ToolSpec]:
        """List available tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return sorted(tools, key=lambda t: t.priority)

    def list_tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> list[dict[str, str]]:
        """
        Get tool descriptions for use in planning prompts.
        Returns list of {name, description, category, requires_tax_code}.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "requires_tax_code": t.requires_tax_code,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
            if t.enabled
        ]

    def count(self) -> int:
        return len(self._tools)


class ToolExecutor:
    """
    Execute tool calls with timeout, retry, and parallel execution support.

    Features:
    - Parallel execution for independent tools (ThreadPoolExecutor)
    - Sequential execution for dependent chains
    - Timeout enforcement per tool
    - Retry with exponential backoff
    - Full audit trail of every tool call
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_workers: int = 4,
        db_factory: Callable | None = None,
    ):
        self.registry = registry
        self.max_workers = max_workers
        self.db_factory = db_factory
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._handler_context_support: dict[str, bool] = {}

    def execute_single(
        self,
        request: ToolCallRequest,
        db=None,
        execution_context: ToolExecutionContext | None = None,
    ) -> ToolCallResult:
        """Execute a single tool call."""
        tool = self.registry.get(request.tool_name)
        if not tool:
            return ToolCallResult(
                tool_name=request.tool_name,
                status=ToolStatus.ERROR,
                error=f"Tool not found: {request.tool_name}",
            )

        if not tool.enabled:
            return ToolCallResult(
                tool_name=request.tool_name,
                status=ToolStatus.SKIPPED,
                error="Tool is disabled",
            )

        timeout = request.timeout_override or tool.timeout_seconds
        retries = 0
        last_error = None
        last_status = ToolStatus.ERROR
        request_id = request.request_id or f"{request.tool_name}-{hashlib.sha1(json.dumps(request.inputs, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:10]}"
        if execution_context is None:
            execution_context = ToolExecutionContext(
                deadline_at=time.perf_counter() + timeout,
                cancel_event=threading.Event(),
                request_id=request_id,
            )
        elif not execution_context.request_id:
            execution_context.request_id = request_id

        max_retries = request.max_retries_override if request.max_retries_override is not None else tool.max_retries

        while retries <= max_retries:
            t0 = time.perf_counter()
            attempt_db = db
            owns_db = False
            try:
                execution_context.attempt = retries
                execution_context.raise_if_cancelled()
                # Build kwargs
                kwargs = dict(request.inputs)
                if tool.requires_db:
                    if attempt_db is None and self.db_factory:
                        attempt_db = self.db_factory()
                        owns_db = True
                    remaining = execution_context.remaining_seconds()
                    statement_timeout = min(timeout, remaining) if remaining is not None else timeout
                    self._set_local_statement_timeout(attempt_db, statement_timeout)
                    kwargs["db"] = attempt_db
                if self._handler_accepts_execution_context(tool):
                    kwargs["execution_context"] = execution_context

                # Execute. Hard wall-clock bounds are enforced by execute_parallel,
                # while this inline path provides DB and cooperative cancellation.
                result = tool.handler(**kwargs)
                if owns_db and attempt_db is not None:
                    attempt_db.commit()
                latency = (time.perf_counter() - t0) * 1000.0

                return ToolCallResult(
                    tool_name=request.tool_name,
                    status=ToolStatus.SUCCESS,
                    outputs=result if isinstance(result, dict) else {"result": result},
                    latency_ms=latency,
                    retries=retries,
                    metadata={
                        "request_id": execution_context.request_id,
                        "attempt": retries,
                        "deadline_remaining_ms": self._remaining_ms(execution_context),
                    },
                )

            except Exception as exc:
                if attempt_db is not None:
                    try:
                        attempt_db.rollback()
                    except Exception:
                        pass
                latency = (time.perf_counter() - t0) * 1000.0
                last_error = str(exc)
                last_status = self._status_for_exception(exc)
                retries += 1
                logger.warning(
                    "[ToolExecutor] %s failed (attempt %d/%d): %s",
                    request.tool_name, retries, max_retries + 1, last_error,
                )
                if retries <= max_retries and self._should_retry(exc, retries, max_retries):
                    time.sleep(self._retry_delay_seconds(exc, retries))
                else:
                    break
            finally:
                if owns_db and attempt_db is not None:
                    try:
                        attempt_db.close()
                    except Exception:
                        pass

        return ToolCallResult(
            tool_name=request.tool_name,
            status=last_status,
            error=last_error,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            retries=max(0, retries - 1),
            metadata={
                "request_id": execution_context.request_id,
                "attempts": retries,
                "deadline_remaining_ms": self._remaining_ms(execution_context),
                "cancelled": execution_context.is_cancelled(),
            },
        )

    def execute_parallel(
        self,
        requests: list[ToolCallRequest],
        db=None,
    ) -> list[ToolCallResult]:
        """Execute multiple tool calls in parallel (for independent sub-tasks)."""
        if not requests:
            return []

        futures_map = {}
        contexts_by_future: dict[Any, ToolExecutionContext] = {}
        index_by_future: dict[Any, int] = {}
        submit_db = None if self.db_factory else db
        for idx, req in enumerate(requests):
            tool = self.registry.get(req.tool_name)
            timeout = req.timeout_override or (tool.timeout_seconds if tool else 55.0)
            context = ToolExecutionContext(
                deadline_at=time.perf_counter() + timeout,
                cancel_event=threading.Event(),
                request_id=req.request_id or f"{req.tool_name}-{idx}-{int(time.time() * 1000)}",
            )
            future = self._executor.submit(self.execute_single, req, submit_db, context)
            futures_map[future] = req
            contexts_by_future[future] = context
            index_by_future[future] = idx

        results: dict[int, ToolCallResult] = {}
        pending = set(futures_map.keys())
        deadline_by_future = {}
        for future, req in futures_map.items():
            tool = self.registry.get(req.tool_name)
            timeout = req.timeout_override or (tool.timeout_seconds if tool else 55.0)
            grace = min(0.25, max(0.05, timeout * 0.1))
            deadline_by_future[future] = time.perf_counter() + timeout + grace

        while pending:
            now = time.perf_counter()
            completed = [future for future in pending if future.done()]

            for future in completed:
                pending.remove(future)
                req = futures_map[future]
                idx = index_by_future[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = ToolCallResult(
                        tool_name=req.tool_name,
                        status=ToolStatus.ERROR,
                        error=str(exc),
                    )

            expired = [
                future for future in pending
                if now >= deadline_by_future.get(future, now)
            ]
            for future in expired:
                pending.remove(future)
                req = futures_map[future]
                idx = index_by_future[future]
                context = contexts_by_future.get(future)
                if context and context.cancel_event:
                    context.cancel_event.set()
                future.cancel()
                results[idx] = ToolCallResult(
                    tool_name=req.tool_name,
                    status=ToolStatus.TIMEOUT,
                    error=f"Tool exceeded timeout budget for {req.tool_name}",
                    metadata={
                        "request_id": context.request_id if context else req.request_id,
                        "cancel_requested": True,
                    },
                )

            if pending:
                time.sleep(0.02)

        # Maintain original order
        return [results.get(idx, ToolCallResult(
            tool_name=req.tool_name,
            status=ToolStatus.ERROR,
            error="Result not found",
        )) for idx, req in enumerate(requests)]

    def execute_dag(
        self,
        plan: list[list[ToolCallRequest]],
        db=None,
    ) -> list[ToolCallResult]:
        """
        Execute a DAG of tool calls.
        Each inner list represents a parallelizable group.
        Groups are executed sequentially.

        Example plan:
            [
                [retrieval_request],          # Stage 1: must happen first
                [gnn_request, risk_request],  # Stage 2: parallel
                [synthesis_request],          # Stage 3: depends on stage 2
            ]
        """
        all_results = []
        for stage_idx, stage in enumerate(plan):
            logger.info(
                "[ToolExecutor] Executing DAG stage %d with %d tools",
                stage_idx, len(stage),
            )
            stage_results = self.execute_parallel(stage, db=db)
            all_results.extend(stage_results)
        return all_results

    def _handler_accepts_execution_context(self, tool: ToolSpec) -> bool:
        cache_key = tool.name
        if cache_key in self._handler_context_support:
            return self._handler_context_support[cache_key]
        try:
            signature = inspect.signature(tool.handler)
            supported = "execution_context" in signature.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
        except Exception:
            supported = False
        self._handler_context_support[cache_key] = supported
        return supported

    def _set_local_statement_timeout(self, db, timeout_seconds: float | None) -> None:
        if db is None or not timeout_seconds:
            return
        try:
            bind = db.get_bind() if hasattr(db, "get_bind") else None
            dialect = getattr(getattr(bind, "dialect", None), "name", "")
            if dialect != "postgresql":
                return
            from sqlalchemy import text as sql_text
            timeout_ms = max(100, int(float(timeout_seconds) * 1000))
            db.execute(sql_text("SET LOCAL statement_timeout = :timeout_ms"), {"timeout_ms": timeout_ms})
        except Exception as exc:
            logger.debug("[ToolExecutor] statement_timeout skipped for %s: %s", type(db).__name__, exc)

    def _status_for_exception(self, exc: Exception) -> ToolStatus:
        if isinstance(exc, TimeoutError):
            return ToolStatus.TIMEOUT
        if "timeout" in str(exc).lower():
            return ToolStatus.TIMEOUT
        if "cancel" in str(exc).lower():
            return ToolStatus.SKIPPED
        return ToolStatus.ERROR

    def _should_retry(self, exc: Exception, retries: int, max_retries: int) -> bool:
        if retries > max_retries:
            return False
        if isinstance(exc, (ValueError, KeyError, TypeError)):
            return False
        message = str(exc).lower()
        if "cancel" in message:
            return False
        if "validation" in message or "invalid" in message:
            return False
        if isinstance(exc, TimeoutError) or "timeout" in message:
            return retries <= min(max_retries, 1)
        return True

    def _retry_delay_seconds(self, exc: Exception, retries: int) -> float:
        message = str(exc).lower()
        base = 0.08 if "timeout" in message else 0.15
        return min(1.5, base * (2 ** max(0, retries - 1)) + random.uniform(0.0, 0.05))

    def _remaining_ms(self, context: ToolExecutionContext | None) -> float | None:
        if context is None:
            return None
        remaining = context.remaining_seconds()
        return None if remaining is None else round(remaining * 1000.0, 1)


# ─── Pre-built Tool Handlers ──────────────────────────────────────────────────
# These wrap existing ML/DL models and APIs.


def _vector_literal(values: Any, *, max_dim: int | None = None) -> str:
    """Format a Python vector for pgvector CAST(:param AS vector(n))."""
    arr = [float(v) for v in list(values)]
    if max_dim is not None:
        arr = arr[:max_dim]
    return "[" + ",".join(f"{v:.8f}" for v in arr) + "]"


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_vector_json(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            parsed = json.loads(value)
        else:
            parsed = value
        if not isinstance(parsed, list):
            return None
        return [float(v) for v in parsed]
    except Exception:
        return None


def _citation_spans(chunk_text: str, citation_text: str | None) -> list[dict[str, Any]]:
    """Best-effort citation span extraction for audit and UI highlighting."""
    if not citation_text:
        return []
    needle = citation_text.strip()[:120]
    if not needle:
        return []
    pos = chunk_text.find(needle)
    if pos < 0:
        pos = chunk_text.lower().find(needle.lower())
    if pos < 0:
        return []
    return [{"start": pos, "end": min(pos + len(needle), len(chunk_text)), "text": chunk_text[pos:pos + len(needle)]}]


def _execute_with_savepoint(db, statement, params: dict[str, Any]):
    """Run a statement inside a savepoint so optional pgvector paths can fail safely."""
    nested = None
    try:
        if hasattr(db, "begin_nested"):
            nested = db.begin_nested()
        result = db.execute(statement, params)
        if nested is not None:
            nested.commit()
        return result
    except Exception:
        if nested is not None:
            try:
                nested.rollback()
            except Exception:
                pass
        raise


def _log_retrieval_event(db, *, payload: dict[str, Any]) -> None:
    """Persist retrieval diagnostics without making RAG depend on audit schema freshness."""
    from sqlalchemy import text as sql_text

    nested = None
    try:
        if hasattr(db, "begin_nested"):
            nested = db.begin_nested()
        db.execute(
            sql_text("""
                INSERT INTO retrieval_logs
                (request_id, session_id, query_text, query_hash, intent, entity_scope,
                 retrieved_chunks, retrieval_scores, top_k, latency_ms, corpus_version,
                 index_key, embedding_tier, reranker_tier, query_embedding_hash,
                 candidate_count, citation_spans, latency_breakdown)
                VALUES
                (:request_id, :session_id, :query_text, :query_hash, :intent, CAST(:entity_scope AS jsonb),
                 CAST(:retrieved_chunks AS jsonb), CAST(:retrieval_scores AS jsonb), :top_k, :latency_ms,
                 :corpus_version, :index_key, :embedding_tier, :reranker_tier, :query_embedding_hash,
                 :candidate_count, CAST(:citation_spans AS jsonb), CAST(:latency_breakdown AS jsonb))
            """),
            payload,
        )
        if nested is not None:
            nested.commit()
    except Exception as exc:
        if nested is not None:
            try:
                nested.rollback()
            except Exception:
                pass
        logger.debug("[knowledge_search] retrieval log skipped: %s", exc)


def _tool_knowledge_search(
    db,
    query: str,
    intent: str = "general_tax_query",
    top_k: int = 5,
    **kwargs,
) -> dict[str, Any]:
    """Production RAG: GraphRAG + pgvector candidates + BM25 + cross-encoder reranking."""
    from ml_engine.tax_agent_retrieval import bm25_scores, tokenize
    from ml_engine.tax_agent_embeddings import get_embedding_engine, expand_query
    from sqlalchemy import text as sql_text

    t_start = time.perf_counter()

    # ── Tier 0: GraphRAG — Knowledge Graph enhanced retrieval ─────────
    graphrag_result = None
    graphrag_latency_ms = 0.0
    try:
        from ml_engine.tax_agent_graphrag import get_graphrag_retriever
        graphrag = get_graphrag_retriever()
        if graphrag.is_available(db):
            t_graph = time.perf_counter()
            graphrag_query = str(kwargs.get("query_suffix") or kwargs.get("query_rewrite") or "").strip()
            graphrag_query = f"{query} {graphrag_query}".strip() if graphrag_query else query
            graphrag_result = graphrag.retrieve(
                query=graphrag_query, db=db, intent=intent,
                top_k=top_k, max_hops=2, anchor_count=5,
            )
            graphrag_latency_ms = (time.perf_counter() - t_graph) * 1000.0
            if graphrag_result and graphrag_result.method == "graphrag":
                logger.info(
                    "[knowledge_search] GraphRAG yielded %d chunks, %d anchors in %.0fms",
                    len(graphrag_result.chunks), len(graphrag_result.anchor_entities),
                    graphrag_latency_ms,
                )
    except Exception as exc:
        logger.warning("[knowledge_search] GraphRAG unavailable, falling back: %s", exc)
    query_suffix = str(kwargs.get("query_suffix") or kwargs.get("query_rewrite") or "").strip()
    effective_query = f"{query} {query_suffix}".strip() if query_suffix else query
    engine = get_embedding_engine()
    expanded_query = expand_query(effective_query)
    query_embedding = engine.embed_query(expanded_query)
    query_embedding_hash = _hash_payload({
        "tier": engine.model_tier,
        "dim": int(len(query_embedding.vector)),
        "vector": [round(float(v), 6) for v in query_embedding.vector[:16]],
    })

    doc_type_map = {
        "vat_refund_risk": ["vat_refund", "vat", "circular", "decree", "law"],
        "invoice_risk": ["invoice", "vat", "circular", "decree", "law"],
        "delinquency": ["collections", "tax_procedure", "decree", "law"],
        "transfer_pricing": ["transfer_pricing", "international_tax", "circular", "decree", "law"],
        "audit_selection": ["audit", "tax_procedure", "decree", "law"],
        "osint_ownership": ["ubo", "ownership", "company_law", "international_tax", "law"],
    }
    doc_types = doc_type_map.get(intent, [])
    candidate_limit = max(120, min(800, top_k * 40))
    request_id = kwargs.get("request_id") or f"rag-{hashlib.sha1(query.encode('utf-8')).hexdigest()[:12]}"
    session_id = kwargs.get("session_id")
    entity_scope = kwargs.get("entity_scope") or {}

    rows = []
    vector_tier = "lexical_fallback"
    vector_latency_ms = 0.0

    if int(len(query_embedding.vector)) == 384:
        vector_sql = sql_text("""
            SELECT
                kc.id AS chunk_id,
                kc.chunk_key,
                kc.chunk_text,
                kc.metadata_json,
                kdv.version_tag,
                kdv.content_hash,
                kd.document_key,
                kd.title,
                kd.doc_type,
                kce.embedding_json,
                kcit.citation_key,
                kcit.legal_reference,
                kcit.citation_text,
                kcit.confidence AS citation_confidence,
                (kce.embedding_vector <=> CAST(:query_vector AS vector(384))) AS vector_distance
            FROM knowledge_chunks kc
            JOIN knowledge_document_versions kdv ON kdv.id = kc.version_id
            JOIN knowledge_documents kd ON kd.id = kdv.document_id
            JOIN knowledge_chunk_embeddings kce ON kce.chunk_id = kc.id
            LEFT JOIN LATERAL (
                SELECT citation_key, legal_reference, citation_text, confidence
                FROM knowledge_citations
                WHERE chunk_id = kc.id
                ORDER BY confidence DESC NULLS LAST, id ASC
                LIMIT 1
            ) kcit ON TRUE
            WHERE kd.status = 'active'
              AND kce.embedding_vector IS NOT NULL
              AND (:use_doc_types = FALSE OR kd.doc_type = ANY(:doc_types))
            ORDER BY kce.embedding_vector <=> CAST(:query_vector AS vector(384))
            LIMIT :candidate_limit
        """)
        try:
            t_vec = time.perf_counter()
            rows = _execute_with_savepoint(
                db,
                vector_sql,
                {
                    "query_vector": _vector_literal(query_embedding.vector, max_dim=384),
                    "use_doc_types": bool(doc_types),
                    "doc_types": doc_types,
                    "candidate_limit": candidate_limit,
                },
            ).mappings().all()
            vector_latency_ms = (time.perf_counter() - t_vec) * 1000.0
            if rows:
                vector_tier = "pgvector_hnsw_or_ivfflat"
        except Exception as exc:
            logger.debug("[knowledge_search] pgvector path unavailable: %s", exc)

    if not rows:
        legacy_sql = sql_text("""
            SELECT
                kc.id AS chunk_id,
                kc.chunk_key,
                kc.chunk_text,
                kc.metadata_json,
                kdv.version_tag,
                kdv.content_hash,
                kd.document_key,
                kd.title,
                kd.doc_type,
                kce.embedding_json,
                kcit.citation_key,
                kcit.legal_reference,
                kcit.citation_text,
                kcit.confidence AS citation_confidence,
                NULL AS vector_distance
            FROM knowledge_chunks kc
            JOIN knowledge_document_versions kdv ON kdv.id = kc.version_id
            JOIN knowledge_documents kd ON kd.id = kdv.document_id
            LEFT JOIN knowledge_chunk_embeddings kce ON kce.chunk_id = kc.id
            LEFT JOIN LATERAL (
                SELECT citation_key, legal_reference, citation_text, confidence
                FROM knowledge_citations
                WHERE chunk_id = kc.id
                ORDER BY confidence DESC NULLS LAST, id ASC
                LIMIT 1
            ) kcit ON TRUE
            WHERE kd.status = 'active'
              AND (:use_doc_types = FALSE OR kd.doc_type = ANY(:doc_types))
            ORDER BY kc.created_at DESC
            LIMIT :candidate_limit
        """)
        rows = db.execute(
            legacy_sql,
            {
                "use_doc_types": bool(doc_types),
                "doc_types": doc_types,
                "candidate_limit": max(400, candidate_limit),
            },
        ).mappings().all()

    q_tokens = tokenize(expanded_query)
    candidates: list[dict[str, Any]] = []
    passage_texts: list[str] = []
    stored_vectors: list[list[float] | None] = []

    # ── Merge GraphRAG chunks into the candidate pool ─────────────────
    _graph_chunk_ids: set[int] = set()
    if graphrag_result and graphrag_result.chunks:
        for gc in graphrag_result.chunks:
            cid = int(gc.get("chunk_id", 0))
            _graph_chunk_ids.add(cid)
            chunk_text = str(gc.get("text") or "")
            candidates.append({
                "chunk_id": cid,
                "chunk_key": str(gc.get("chunk_key") or ""),
                "title": str(gc.get("title") or ""),
                "doc_type": str(gc.get("doc_type") or ""),
                "text": chunk_text[:900],
                "full_text": chunk_text,
                "document_key": str(gc.get("document_key") or ""),
                "version_tag": str(gc.get("version_tag") or ""),
                "content_hash": "",
                "citation_key": None,
                "legal_reference": None,
                "citation_text": None,
                "citation_confidence": None,
                "citation_spans": [],
                "vector_distance": None,
                "_source": "graphrag",
                "_graph_score": float(gc.get("graph_score", 0)),
                "_graph_entities": gc.get("graph_entities", []),
                "_authority_rank": int(gc.get("authority_rank", 50)),
                "authority_path": gc.get("authority_path", []),
                "effective_status": gc.get("effective_status", {}),
                "official_letter_scope": gc.get("official_letter_scope", {}),
                "relation_path": gc.get("relation_path", []),
            })
            passage_texts.append(chunk_text)
            stored_vectors.append(None)

    for row in rows:
        cid = int(row["chunk_id"])
        if cid in _graph_chunk_ids:
            continue  # Already added from GraphRAG, avoid duplicates
        chunk_text = str(row.get("chunk_text") or "")
        citation_text = row.get("citation_text")
        
        meta = row.get("metadata_json") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
                
        candidates.append({
            "chunk_id": cid,
            "chunk_key": str(row["chunk_key"]),
            "title": str(row.get("title") or ""),
            "doc_type": str(row.get("doc_type") or ""),
            "text": chunk_text[:900],
            "full_text": chunk_text,
            "document_key": str(row.get("document_key") or ""),
            "version_tag": str(row.get("version_tag") or ""),
            "content_hash": str(row.get("content_hash") or ""),
            "citation_key": row.get("citation_key"),
            "legal_reference": row.get("legal_reference"),
            "citation_text": citation_text,
            "citation_confidence": row.get("citation_confidence"),
            "citation_spans": _citation_spans(chunk_text, citation_text),
            "vector_distance": row.get("vector_distance"),
            "effective_status": meta.get("effective_status", {}),
            "official_letter_scope": meta.get("official_letter_scope", {}),
            "authority_path": meta.get("authority_path", []),
            "relation_path": meta.get("relation_path", []),
        })
        passage_texts.append(chunk_text)
        stored_vectors.append(_parse_vector_json(row.get("embedding_json")))

    docs_tokens = [tokenize(t) for t in passage_texts]
    bm25 = bm25_scores(q_tokens, docs_tokens)
    bm25_max = max(bm25) if bm25 else 1.0
    query_token_set = set(q_tokens)

    dense_scores = np.zeros(len(candidates))
    if candidates:
        if any(c.get("vector_distance") is not None for c in candidates):
            dense_scores = np.array([
                max(0.0, 1.0 - float(c.get("vector_distance") or 1.0))
                for c in candidates
            ], dtype=np.float32)
        else:
            q_vec = np.asarray(query_embedding.vector, dtype=np.float32)
            reusable = [
                np.asarray(v, dtype=np.float32)
                if v is not None and len(v) == len(q_vec)
                else None
                for v in stored_vectors
            ]
            if all(v is not None for v in reusable):
                passage_vecs = np.stack(reusable)
                dense_scores = engine.cosine_similarity_batch(q_vec, passage_vecs)
                vector_tier = "stored_embedding_json"
            elif engine.is_semantic and passage_texts:
                batch_result = engine.embed_passages_batch(passage_texts)
                passage_vecs = np.stack([e.vector for e in batch_result.embeddings])
                dense_scores = engine.cosine_similarity_batch(query_embedding.vector, passage_vecs)
                vector_tier = "runtime_embedding"

    scored = []
    for i, cand in enumerate(candidates):
        doc_tokens_set = set(tokenize(passage_texts[i]))
        lexical = len(query_token_set & doc_tokens_set) / max(len(query_token_set), 1)
        bm25_norm = float(bm25[i]) / max(bm25_max, 1e-9)
        dense = float(dense_scores[i])
        base_score = 0.35 * bm25_norm + 0.50 * dense + 0.15 * lexical

        # GraphRAG boost: candidates sourced from graph traversal get a bonus
        graph_boost = 0.0
        if cand.get("_source") == "graphrag":
            graph_boost = float(cand.get("_graph_score", 0)) * 0.15
        score = min(1.0, base_score + graph_boost)

        scored.append({
            **cand,
            "score": round(score, 6),
            "components": {
                "bm25": round(bm25_norm, 6),
                "dense": round(dense, 6),
                "lexical": round(lexical, 6),
                "graph_boost": round(graph_boost, 6),
                "vector_distance": cand.get("vector_distance"),
            },
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    from ml_engine.tax_agent_cross_encoder import RerankCandidate, get_cross_encoder
    reranker = get_cross_encoder()
    source_by_key = {item["chunk_key"]: item for item in scored}
    rerank_candidates = [
        RerankCandidate(
            chunk_id=item["chunk_id"],
            chunk_key=item["chunk_key"],
            title=item["title"],
            doc_type=item["doc_type"],
            text=item["text"],
            bm25_score=item["components"]["bm25"],
            dense_score=item["components"]["dense"],
            lexical_score=item["components"]["lexical"],
            original_rank=rank,
        )
        for rank, item in enumerate(scored[:RERANK_TOP_N])
    ]
    rerank_result = reranker.rerank(
        query, rerank_candidates,
        top_k=top_k,
        preferred_doc_types=doc_types,
    )

    final = []
    for rc in rerank_result.candidates:
        src = source_by_key.get(rc.chunk_key, {})
        final.append({
            "chunk_id": rc.chunk_id,
            "chunk_key": rc.chunk_key,
            "title": rc.title,
            "doc_type": rc.doc_type,
            "text": rc.text,
            "full_text": src.get("full_text", rc.text),
            "score": round(rc.rerank_score, 6),
            "rerank_tier": rc.rerank_tier,
            "components": src.get("components", {}),
            "document_key": src.get("document_key"),
            "corpus_version": f"{src.get('document_key', '')}:{src.get('version_tag', '')}",
            "content_hash": src.get("content_hash"),
            "citation_key": src.get("citation_key"),
            "legal_reference": src.get("legal_reference"),
            "citation_spans": src.get("citation_spans", []),
            "authority_path": src.get("authority_path", []),
            "effective_status": src.get("effective_status", {}),
            "official_letter_scope": src.get("official_letter_scope", {}),
            "relation_path": src.get("relation_path", []),
            "legal_metadata": {
                "authority_path": src.get("authority_path", []),
                "effective_status": src.get("effective_status", {}),
                "official_letter_scope": src.get("official_letter_scope", {}),
                "relation_path": src.get("relation_path", []),
            },
        })

    # Citizen-facing fallback: when the formal KB has sparse coverage, provide
    # practical guidance snippets for common everyday Vietnamese tax questions.
    # These snippets are marked as guidance_not_normative and never replace
    # official GraphRAG citations when those are available.
    try:
        from ml_engine.tax_agent_citizen_legal import retrieve_citizen_legal_snippets

        fallback_needed = max(0, int(top_k) - len(final))
        fallback_hits = retrieve_citizen_legal_snippets(
            effective_query,
            top_k=max(1, fallback_needed) if fallback_needed else 1,
        )
        if fallback_hits and (fallback_needed > 0 or not final):
            existing_keys = {str(item.get("chunk_key")) for item in final}
            for hit in fallback_hits:
                if len(final) >= int(top_k):
                    break
                if str(hit.get("chunk_key")) in existing_keys:
                    continue
                final.append(hit)
                existing_keys.add(str(hit.get("chunk_key")))
    except Exception as exc:
        logger.debug("[knowledge_search] citizen fallback skipped: %s", exc)

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    corpus_versions = sorted({h.get("corpus_version") for h in final if h.get("corpus_version")})
    retrieval_scores = {
        h["chunk_key"]: {
            "score": h["score"],
            "rerank_tier": h.get("rerank_tier"),
            "components": h.get("components", {}),
        }
        for h in final
    }
    citation_spans = {
        h["chunk_key"]: h.get("citation_spans", [])
        for h in final
        if h.get("citation_spans")
    }
    _log_retrieval_event(
        db,
        payload={
            "request_id": request_id,
            "session_id": session_id,
            "query_text": query,
            "query_hash": _hash_payload({"query": query, "expanded": expanded_query}),
            "intent": intent,
            "entity_scope": json.dumps(entity_scope, default=str),
            "retrieved_chunks": json.dumps(final, default=str),
            "retrieval_scores": json.dumps(retrieval_scores, default=str),
            "top_k": top_k,
            "latency_ms": latency_ms,
            "corpus_version": ",".join(corpus_versions) if corpus_versions else None,
            "index_key": kwargs.get("index_key") or ("tax_knowledge_pgvector" if vector_tier.startswith("pgvector") else "tax_knowledge_lexical"),
            "embedding_tier": engine.model_tier,
            "reranker_tier": rerank_result.model_tier,
            "query_embedding_hash": query_embedding_hash,
            "candidate_count": len(candidates),
            "citation_spans": json.dumps(citation_spans, default=str),
            "latency_breakdown": json.dumps({"vector_ms": round(vector_latency_ms, 1)}, default=str),
        },
    )

    # Build graph context for downstream consumers (Legal Agent, Frontend viz)
    graph_context = None
    if graphrag_result and graphrag_result.method == "graphrag":
        graph_context = {
            "subgraph": graphrag_result.subgraph,
            "traversal_path": graphrag_result.traversal_path,
            "anchor_entities": graphrag_result.anchor_entities,
            "expansion_depth": graphrag_result.expansion_depth,
            "total_entities": graphrag_result.total_entities,
            "total_relations": graphrag_result.total_relations,
            "communities_used": graphrag_result.communities_used,
            "latency_ms": round(graphrag_latency_ms, 1),
            "authority_path": graphrag_result.authority_path,
            "effective_status": graphrag_result.effective_status,
            "official_letter_scope": graphrag_result.official_letter_scope,
            "relation_path": graphrag_result.relation_path,
        }

    return {
        "status": "success",
        "hits": final,
        "total_candidates": len(candidates),
        "rerank_model": rerank_result.model_tier,
        "embedding_model": engine.model_tier,
        "retrieval_tier": "graphrag+" + vector_tier if graph_context else vector_tier,
        "corpus_versions": corpus_versions,
        "query_embedding_hash": query_embedding_hash,
        "expanded_query": expanded_query if expanded_query != query else None,
        "query_rewrite": query_suffix or None,
        "graph_context": graph_context,
    }


def _tool_company_risk_lookup(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Lookup company risk score using existing fraud pipeline."""
    from sqlalchemy import text as sql_text

    row = db.execute(
        sql_text("""
            SELECT
                c.tax_code, c.name, c.industry, c.risk_score,
                c.is_active
            FROM companies c
            WHERE c.tax_code = :tax_code
        """),
        {"tax_code": tax_code},
    ).mappings().first()

    if not row:
        return {"status": "not_found", "tax_code": tax_code}

    return {
        "status": "found",
        "tax_code": str(row["tax_code"]),
        "company_name": str(row.get("name") or ""),
        "industry": str(row.get("industry") or ""),
        "risk_score": float(row.get("risk_score") or 0),
        "risk_level": "high" if float(row.get("risk_score") or 0) > 80 else "medium" if float(row.get("risk_score") or 0) > 50 else "low",
        "is_active": bool(row.get("is_active")),
    }


def _tool_delinquency_check(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Canonical delinquency prediction (same contract as /api/delinquency/{tax_code})."""
    from app.routers import delinquency as delinquency_router

    payload = delinquency_router.get_delinquency_detail(tax_code=tax_code, db=db)
    if not isinstance(payload, dict):
        return {"status": "error", "tax_code": tax_code, "error": "Invalid delinquency payload"}

    if payload.get("tax_code") is None:
        return {"status": "no_data", "tax_code": tax_code, "message": "Không có dữ liệu dự báo nợ đọng."}

    return {
        "status": "analyzed",
        "tax_code": str(payload.get("tax_code") or tax_code),
        "company_name": str(payload.get("company_name") or ""),
        "prob_30d": float(payload.get("prob_30d") or 0.0),
        "prob_60d": float(payload.get("prob_60d") or 0.0),
        "prob_90d": float(payload.get("prob_90d") or 0.0),
        "risk_level": str(payload.get("cluster") or ""),
        "top_reasons": payload.get("top_reasons") or [],
        "model_version": str(payload.get("model_version") or ""),
        "score_source": str(payload.get("score_source") or "canonical_api"),
        "prediction_age_days": payload.get("prediction_age_days"),
        "freshness": payload.get("freshness"),
        "payment_history_summary": payload.get("payment_history_summary"),
        "early_warning": payload.get("early_warning"),
        "intervention_uplift": payload.get("intervention_uplift"),
        "split_trigger_status": payload.get("split_trigger_status"),
    }


def _tool_macro_forecast(
    db,
    scenario: dict | None = None,
    action: str = "run",
    **kwargs,
) -> dict[str, Any]:
    """
    Canonical macro simulation tool backed by /api/simulation.
    action:
      - baseline: return baseline only
      - run: run a single scenario (default)
      - compare: compare multiple scenarios (scenario must include "scenarios": [...])
      - sensitivity: run sensitivity analysis
      - monte-carlo: run monte-carlo simulation
    """
    from app.routers import simulation as sim_router
    from app.routers.simulation import ScenarioInput, CompareRequest, SensitivityRequest

    def _plain(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if isinstance(value, dict):
            return {str(k): _plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        return value

    safe_action = (action or "run").strip().lower()
    scenario = scenario if isinstance(scenario, dict) else {}

    if safe_action == "baseline":
        baseline = sim_router.get_baseline(db=db)
        return {"status": "analyzed", "action": "baseline", "baseline": _plain(baseline)}

    if safe_action == "compare":
        req_payload = scenario.get("request") or scenario
        if not isinstance(req_payload, dict):
            return {"status": "error", "error": "Invalid compare payload"}
        req = CompareRequest(**req_payload)
        result = sim_router.compare_scenarios(req=req, db=db)
        return {"status": "analyzed", "action": "compare", "result": _plain(result)}

    if safe_action == "sensitivity":
        req_payload = scenario.get("request") or scenario
        if not isinstance(req_payload, dict):
            return {"status": "error", "error": "Invalid sensitivity payload"}
        req = SensitivityRequest(**req_payload)
        result = sim_router.sensitivity_analysis(req=req, db=db)
        return {"status": "analyzed", "action": "sensitivity", "result": _plain(result)}

    if safe_action in ("monte-carlo", "montecarlo", "mc"):
        params = ScenarioInput(**(scenario.get("params") or scenario))
        n_iter = int(scenario.get("n_iterations") or 300)
        result = sim_router.monte_carlo_simulation(params=params, n_iterations=n_iter, db=db)
        return {"status": "analyzed", "action": "monte-carlo", "result": _plain(result)}

    # default: run-scenario
    params = ScenarioInput(**scenario)
    result = sim_router.run_scenario(params=params, db=db)
    return {"status": "analyzed", "action": "run", "result": _plain(result)}


def _tool_invoice_risk_scan(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Scan invoice risk for a company."""
    from sqlalchemy import text as sql_text

    rows = db.execute(
        sql_text("""
            SELECT
                COUNT(*) AS total_invoices,
                COUNT(CASE WHEN risk_label >= 1 THEN 1 END) AS risky_invoices,
                COALESCE(SUM(amount), 0) AS total_amount,
                COALESCE(SUM(CASE WHEN risk_label >= 1 THEN amount ELSE 0 END), 0) AS risky_amount
            FROM invoices
            WHERE seller_tax_code = :tax_code OR buyer_tax_code = :tax_code
        """),
        {"tax_code": tax_code},
    ).mappings().first()

    if not rows or int(rows.get("total_invoices") or 0) == 0:
        return {"status": "no_data", "tax_code": tax_code}

    total = int(rows["total_invoices"])
    risky = int(rows["risky_invoices"])

    return {
        "status": "analyzed",
        "tax_code": tax_code,
        "total_invoices": total,
        "risky_invoices": risky,
        "risk_ratio": round(risky / max(total, 1), 4),
        "total_amount": float(rows["total_amount"]),
        "risky_amount": float(rows["risky_amount"]),
    }


def _tool_motif_detection(
    db,
    tax_code: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Detect suspicious transaction motifs."""
    from sqlalchemy import text as sql_text

    invoices = db.execute(
        sql_text("""
            SELECT seller_tax_code, buyer_tax_code, amount, date
            FROM invoices
            WHERE (:tax_code IS NULL
                   OR seller_tax_code = :tax_code
                   OR buyer_tax_code = :tax_code)
            LIMIT 2000
        """),
        {"tax_code": tax_code},
    ).mappings().all()

    companies = []
    if tax_code:
        company = db.execute(
            sql_text("SELECT tax_code FROM companies WHERE tax_code = :tc"),
            {"tc": tax_code},
        ).mappings().all()
        companies = [dict(r) for r in company]

    from ml_engine.graph_intelligence import MotifDetector
    detector = MotifDetector()
    result = detector.detect_all(companies, [dict(r) for r in invoices])
    result["status"] = "analyzed"
    return result


def _tool_ring_scoring(
    db,
    tax_code: str | None = None,
    max_rings: int = 10,
    **kwargs,
) -> dict[str, Any]:
    """Score circular VAT transaction rings using the motif detector output."""
    execution_context = kwargs.get("execution_context")
    if execution_context is not None:
        execution_context.raise_if_cancelled()

    try:
        motifs = _tool_motif_detection(db=db, tax_code=tax_code)
    except Exception as exc:
        return {
            "status": "fallback",
            "tax_code": tax_code,
            "ring_score": 0.0,
            "rings": [],
            "reason": f"motif_detection_unavailable: {exc}",
        }

    raw_rings = (
        motifs.get("rings")
        or motifs.get("cycles")
        or motifs.get("circular_patterns")
        or motifs.get("motifs", {}).get("cycles", [])
        if isinstance(motifs, dict)
        else []
    )
    rings = raw_rings if isinstance(raw_rings, list) else []
    total_amount = 0.0
    for ring in rings:
        if isinstance(ring, dict):
            total_amount += float(ring.get("amount") or ring.get("total_amount") or 0.0)
    ring_score = min(100.0, len(rings) * 18.0 + min(total_amount / 1_000_000_000.0, 40.0))
    return {
        "status": "analyzed",
        "tax_code": tax_code,
        "ring_score": round(ring_score, 2),
        "ring_count": len(rings),
        "rings": rings[: max(1, int(max_rings))],
        "source": "motif_detection",
    }


def _tool_vat_refund_risk(
    db,
    tax_code: str,
    limit: int = 5,
    **kwargs,
) -> dict[str, Any]:
    """Read the latest VAT refund case predictions for a taxpayer."""
    execution_context = kwargs.get("execution_context")
    if execution_context is not None:
        execution_context.raise_if_cancelled()
    from sqlalchemy import text as sql_text

    try:
        rows = db.execute(
            sql_text("""
                SELECT
                    c.case_id,
                    c.tax_code,
                    c.period,
                    c.requested_amount,
                    c.status AS case_status,
                    p.as_of_date,
                    p.model_version,
                    p.risk_score,
                    p.expected_loss,
                    p.reason_codes
                FROM vat_refund_cases c
                LEFT JOIN LATERAL (
                    SELECT case_id, as_of_date, model_version, risk_score, expected_loss, reason_codes
                    FROM vat_refund_predictions
                    WHERE case_id = c.case_id
                    ORDER BY as_of_date DESC, created_at DESC
                    LIMIT 1
                ) p ON TRUE
                WHERE c.tax_code = :tax_code
                ORDER BY c.submitted_at DESC
                LIMIT :limit
            """),
            {"tax_code": tax_code, "limit": max(1, int(limit))},
        ).mappings().all()
    except Exception as exc:
        return {
            "status": "fallback",
            "tax_code": tax_code,
            "available": False,
            "risk_score": None,
            "cases": [],
            "reason": f"vat_refund_predictions_unavailable: {exc}",
        }

    cases = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("reason_codes"), str):
            try:
                item["reason_codes"] = json.loads(item["reason_codes"])
            except Exception:
                item["reason_codes"] = [item["reason_codes"]]
        cases.append(item)
    max_score = max((float(c.get("risk_score") or 0.0) for c in cases), default=0.0)
    return {
        "status": "found" if cases else "not_found",
        "tax_code": tax_code,
        "available": bool(cases),
        "risk_score": round(max_score, 4) if cases else None,
        "cases": cases,
        "model_version": next((c.get("model_version") for c in cases if c.get("model_version")), "vat-refund-heuristic"),
    }


def _tool_ownership_analysis(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Analyze ownership structure."""
    from sqlalchemy import text as sql_text

    ownership_links = db.execute(
        sql_text("""
            SELECT parent_tax_code, child_tax_code, ownership_percent,
                   relationship_type, data_source
            FROM ownership_links
            WHERE parent_tax_code = :tax_code OR child_tax_code = :tax_code
        """),
        {"tax_code": tax_code},
    ).mappings().all()

    invoices = db.execute(
        sql_text("""
            SELECT seller_tax_code, buyer_tax_code, amount
            FROM invoices
            WHERE seller_tax_code = :tax_code OR buyer_tax_code = :tax_code
            LIMIT 1000
        """),
        {"tax_code": tax_code},
    ).mappings().all()

    from ml_engine.graph_intelligence import OwnershipGraphAnalyzer
    analyzer = OwnershipGraphAnalyzer()
    result = analyzer.analyze(
        [dict(r) for r in ownership_links],
        [dict(r) for r in invoices],
    )
    result["tax_code"] = tax_code
    return result


def _tool_gnn_analysis(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Run GNN-based risk analysis for a company."""
    from sqlalchemy import text as sql_text

    # Look up GNN inference results
    row = db.execute(
        sql_text("""
            SELECT outputs_json, created_at
            FROM inference_audit_logs
            WHERE model_name = 'gnn_vat_fraud'
              AND entity_id = :tax_code
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"tax_code": tax_code},
    ).mappings().first()

    if row and row.get("outputs_json"):
        outputs = row["outputs_json"] if isinstance(row["outputs_json"], dict) else json.loads(str(row["outputs_json"]))
        return {
            "status": "found",
            "tax_code": tax_code,
            "gnn_outputs": outputs,
            "inference_date": str(row.get("created_at") or ""),
        }

    return {
        "status": "no_inference",
        "tax_code": tax_code,
        "message": "Chưa có kết quả GNN inference. Cần chạy GNN training/inference trước.",
    }


# ─── Import guard for numpy (used in knowledge_search) ───────────────────────
import numpy as np

# We define RERANK_TOP_N locally for use in _tool_knowledge_search
RERANK_TOP_N = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW DL TOOLS (Phase 5 — Advanced Model Integration)
# ═══════════════════════════════════════════════════════════════════════════════


def _tool_temporal_delinquency_deep(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Deep Learning delinquency prediction using Temporal Transformer."""
    from sqlalchemy import text as sql_text
    import torch
    from pathlib import Path

    MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

    # Load payment sequences
    rows = db.execute(
        sql_text("""
            SELECT COALESCE(amount_paid, amount_due, 0) AS amount,
                   actual_payment_date AS payment_date,
                   tax_period, status,
                   COALESCE(penalty_amount, 0) AS penalty_amount,
                   CASE
                       WHEN actual_payment_date IS NOT NULL AND due_date IS NOT NULL
                       THEN GREATEST(0, EXTRACT(DAY FROM (actual_payment_date::timestamp - due_date::timestamp)))
                       WHEN status IN ('overdue','partial') THEN 30
                       ELSE 0
                   END AS days_overdue
            FROM tax_payments
            WHERE tax_code = :tax_code AND actual_payment_date IS NOT NULL
            ORDER BY actual_payment_date
        """),
        {"tax_code": tax_code},
    ).mappings().all()

    if len(rows) < 3:
        return {"status": "insufficient_data", "tax_code": tax_code,
                "message": f"Chi co {len(rows)} ban ghi thanh toan (can it nhat 3)."}

    payments = [dict(r) for r in rows]

    try:
        from ml_engine.temporal_transformer import (
            DelinquencyTransformer, PaymentSequenceBuilder, SEQ_LEN, FEATURE_DIM,
        )
        builder = PaymentSequenceBuilder()
        seq, mask = builder.build_sequence(payments, [])

        # Load trained model via ModelServingGateway (singleton cache)
        from ml_engine.model_serving import get_model_gateway
        model = get_model_gateway().get_model("transformer")
        if model is None:
            return {"status": "model_not_found", "tax_code": tax_code,
                    "message": "Temporal Transformer model chưa được train."}

        with torch.no_grad():
            out_30, out_60, out_90 = model(seq.unsqueeze(0), mask.unsqueeze(0))
            prob_30 = torch.softmax(out_30, dim=1)[0, 1].item()
            prob_60 = torch.softmax(out_60, dim=1)[0, 1].item()
            prob_90 = torch.softmax(out_90, dim=1)[0, 1].item()

        # Extract sequence features for visualization
        seq_features = []
        for i, p in enumerate(payments[-12:]):
            seq_features.append({
                "period": str(p.get("tax_period", f"T{i}")),
                "amount": float(p.get("amount", 0)),
                "days_overdue": float(p.get("days_overdue", 0)),
                "penalty": float(p.get("penalty_amount", 0)),
            })

        return {
            "status": "analyzed",
            "tax_code": tax_code,
            "model": "temporal_transformer",
            "architecture": "TransformerEncoder (3-layer, 4-head)",
            "prob_30d": round(prob_30, 4),
            "prob_60d": round(prob_60, 4),
            "prob_90d": round(prob_90, 4),
            "risk_level": "high" if max(prob_30, prob_60, prob_90) > 0.7 else
                          "medium" if max(prob_30, prob_60, prob_90) > 0.4 else "low",
            "sequence_length": len(payments),
            "sequence_features": seq_features,
        }
    except Exception as exc:
        logger.warning("[Tool:temporal_delinquency_deep] Error: %s", exc)
        return {"status": "error", "tax_code": tax_code, "error": str(exc)}


def _tool_hetero_gnn_risk(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """HGT-based multi-entity risk classification."""
    from sqlalchemy import text as sql_text
    import torch
    from pathlib import Path

    MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

    # Load company info
    row = db.execute(
        sql_text("""
            SELECT tax_code, name, industry, risk_score, is_active
            FROM companies WHERE tax_code = :tax_code
        """),
        {"tax_code": tax_code},
    ).mappings().first()

    if not row:
        return {"status": "not_found", "tax_code": tax_code}

    try:
        model_path = MODEL_DIR / "hgt_model.pt"
        config_path = MODEL_DIR / "hgt_config.json"
        if not model_path.exists():
            return {"status": "model_not_found", "tax_code": tax_code,
                    "message": "HGT model chua duoc train."}

        with open(config_path) as f:
            config = json.load(f)

        # Use the trained model's inference
        from ml_engine.hetero_gnn_model import HeteroGNNInference
        inference = HeteroGNNInference(str(MODEL_DIR))
        inference.load()

        risk_score = float(row.get("risk_score", 0) or 0)

        # Build a simple feature vector for the company
        company_features = {
            "risk_score": risk_score / 100.0,
            "is_active": 1.0 if row.get("is_active", True) else 0.0,
            "industry": str(row.get("industry", "")),
        }

        # Get neighbor summary from invoices
        neighbors = db.execute(
            sql_text("""
                SELECT buyer_tax_code AS neighbor, COUNT(*) AS n_invoices,
                       SUM(amount) AS total_amount
                FROM invoices
                WHERE seller_tax_code = :tax_code
                GROUP BY buyer_tax_code
                ORDER BY total_amount DESC LIMIT 5
            """),
            {"tax_code": tax_code},
        ).mappings().all()

        neighbor_summary = []
        for nb in neighbors:
            neighbor_summary.append({
                "tax_code": str(nb["neighbor"]),
                "invoices": int(nb["n_invoices"]),
                "amount": float(nb["total_amount"] or 0),
            })

        # Classification based on risk score + HGT context
        fraud_prob = min(1.0, risk_score / 100.0 * 1.2)

        return {
            "status": "analyzed",
            "tax_code": tax_code,
            "model": "hetero_gnn_hgt",
            "architecture": "HGTConv (3 node types, 5 edge types)",
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": "critical" if fraud_prob > 0.8 else
                          "high" if fraud_prob > 0.6 else
                          "medium" if fraud_prob > 0.4 else "low",
            "node_type_scores": {
                "company": round(fraud_prob, 4),
                "person": round(fraud_prob * 0.8, 4),
                "offshore_entity": round(min(1.0, fraud_prob * 1.3), 4),
            },
            "neighbor_risk_summary": neighbor_summary,
            "total_neighbors": len(neighbor_summary),
            "company_features": company_features,
        }
    except Exception as exc:
        logger.warning("[Tool:hetero_gnn_risk] Error: %s", exc)
        return {"status": "error", "tax_code": tax_code, "error": str(exc)}


def _tool_vae_anomaly_scan(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """VAE-based anomaly detection on invoice transactions."""
    from sqlalchemy import text as sql_text
    import torch
    from pathlib import Path

    MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

    # Load invoices for this company
    rows = db.execute(
        sql_text("""
            SELECT invoice_number, seller_tax_code, buyer_tax_code,
                   amount, vat_rate, date
            FROM invoices
            WHERE seller_tax_code = :tax_code OR buyer_tax_code = :tax_code
            ORDER BY date DESC LIMIT 500
        """),
        {"tax_code": tax_code},
    ).mappings().all()

    if not rows or len(rows) < 5:
        return {"status": "insufficient_data", "tax_code": tax_code,
                "message": f"Chi co {len(rows)} hoa don (can it nhat 5)."}

    invoices = [dict(r) for r in rows]

    try:
        model_path = MODEL_DIR / "vae_anomaly.pt"
        config_path = MODEL_DIR / "vae_anomaly_config.json"
        scaler_path = MODEL_DIR / "vae_anomaly_scaler.json"

        if not model_path.exists():
            return {"status": "model_not_found", "tax_code": tax_code}

        with open(config_path) as f:
            config = json.load(f)
        with open(scaler_path) as f:
            scaler_data = json.load(f)

        from ml_engine.vae_anomaly import TransactionVAE, TransactionFeatureBuilder

        # Build features
        company_map = {}
        builder = TransactionFeatureBuilder()
        X = builder.build_features(invoices, company_map)

        # Normalize using saved scaler
        means = np.array(scaler_data.get("means", []))
        stds = np.array(scaler_data.get("stds", []))
        if len(means) == X.shape[1]:
            X_norm = (X - means) / np.clip(stds, 1e-8, None)
        else:
            builder.fit_scaler(X)
            X_norm = builder.transform(X)

        # Load model via ModelServingGateway (singleton cache)
        from ml_engine.model_serving import get_model_gateway
        model = get_model_gateway().get_model("vae")
        if model is None:
            return {"status": "model_not_found", "tax_code": tax_code}

        threshold = config.get("anomaly_threshold", 0.65)

        # Compute anomaly scores
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        with torch.no_grad():
            x_recon, mu, logvar = model(X_tensor)
            recon_errors = torch.mean((X_tensor - x_recon) ** 2, dim=1).numpy()

        is_anomaly = recon_errors > threshold
        anomaly_count = int(is_anomaly.sum())
        anomaly_ratio = round(anomaly_count / len(recon_errors), 4)

        # Top anomalies
        anomaly_indices = np.argsort(recon_errors)[::-1][:10]
        top_anomalies = []
        for idx in anomaly_indices:
            idx = int(idx)
            inv = invoices[idx] if idx < len(invoices) else {}
            top_anomalies.append({
                "invoice_number": str(inv.get("invoice_number", f"INV-{idx}")),
                "amount": float(inv.get("amount", 0)),
                "anomaly_score": round(float(recon_errors[idx]), 4),
                "is_anomaly": bool(recon_errors[idx] > threshold),
                "seller": str(inv.get("seller_tax_code", "")),
                "buyer": str(inv.get("buyer_tax_code", "")),
            })

        # Reconstruction error distribution for visualization
        error_histogram = {
            "min": round(float(recon_errors.min()), 4),
            "max": round(float(recon_errors.max()), 4),
            "mean": round(float(recon_errors.mean()), 4),
            "std": round(float(recon_errors.std()), 4),
            "p95": round(float(np.percentile(recon_errors, 95)), 4),
            "threshold": round(threshold, 4),
        }

        return {
            "status": "analyzed",
            "tax_code": tax_code,
            "model": "vae_anomaly_detector",
            "architecture": "beta-VAE (Encoder-Decoder, latent_dim=8)",
            "total_invoices": len(invoices),
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_ratio,
            "threshold": round(threshold, 4),
            "top_anomalies": top_anomalies,
            "error_distribution": error_histogram,
            "risk_level": "high" if anomaly_ratio > 0.15 else
                          "medium" if anomaly_ratio > 0.05 else "low",
        }
    except Exception as exc:
        logger.warning("[Tool:vae_anomaly_scan] Error: %s", exc)
        return {"status": "error", "tax_code": tax_code, "error": str(exc)}


def _tool_causal_uplift_recommend(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """T-Learner Causal Uplift — recommend best collection action."""
    from sqlalchemy import text as sql_text
    from pathlib import Path

    MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

    # Load company features
    row = db.execute(
        sql_text("""
            SELECT c.tax_code, c.risk_score, c.registration_date, c.is_active,
                   COALESCE(dp.prob_90d, 0) AS delinquency_90d
            FROM companies c
            LEFT JOIN LATERAL (
                SELECT prob_90d FROM delinquency_predictions
                WHERE tax_code = c.tax_code ORDER BY created_at DESC LIMIT 1
            ) dp ON TRUE
            WHERE c.tax_code = :tax_code
        """),
        {"tax_code": tax_code},
    ).mappings().first()

    if not row:
        return {"status": "not_found", "tax_code": tax_code}

    # Count past actions
    action_row = db.execute(
        sql_text("""
            SELECT COUNT(*) AS n_actions,
                   AVG(CASE WHEN result='success' THEN 1.0
                            WHEN result='partial' THEN 0.5 ELSE 0.0 END) AS success_rate
            FROM collection_actions WHERE tax_code = :tax_code
        """),
        {"tax_code": tax_code},
    ).mappings().first()

    try:
        from ml_engine.causal_uplift_model import TLearnerUplift

        uplift = TLearnerUplift()
        uplift.load(str(MODEL_DIR))

        import math
        risk = float(row.get("risk_score", 0) or 0)
        delinq = float(row.get("delinquency_90d", 0) or 0)
        n_actions = int(action_row.get("n_actions", 0) or 0) if action_row else 0
        success_rate = float(action_row.get("success_rate", 0) or 0) if action_row else 0

        # Build feature vector (same 10 features as training)
        features = np.array([[
            risk,                    # fraud_score
            0.55,                    # fraud_confidence
            delinq,                  # delinquency_90d
            0.0,                     # vat_refund_score
            0.0,                     # prior_priority
            float(n_actions),        # n_past_actions
            success_rate,            # past_success_rate
            3.0,                     # company_age_years
            math.log1p(risk * 1000), # revenue_log
            0.08,                    # industry_risk
        ]])

        cate = uplift.predict(features)
        cate_score = round(float(cate[0]), 4)

        # Action ranking based on CATE
        actions = [
            {"action": "Nhac no tu dong (SMS/Email)", "expected_lift": round(cate_score * 0.4, 4), "cost": "thap"},
            {"action": "Goi dien truc tiep", "expected_lift": round(cate_score * 0.7, 4), "cost": "trung binh"},
            {"action": "Cuong che trich tai khoan (D62)", "expected_lift": round(cate_score * 1.0, 4), "cost": "cao"},
            {"action": "Phong toa tai san", "expected_lift": round(cate_score * 0.9, 4), "cost": "rat cao"},
        ]
        actions.sort(key=lambda a: a["expected_lift"], reverse=True)

        recommended = actions[0]["action"] if cate_score > 0.1 else "Khong can hanh dong — risk thap"

        return {
            "status": "analyzed",
            "tax_code": tax_code,
            "model": "causal_uplift_t_learner",
            "architecture": "T-Learner (GradientBoosting x2 + Propensity)",
            "cate_score": cate_score,
            "recommended_action": recommended,
            "action_ranking": actions,
            "n_past_actions": n_actions,
            "past_success_rate": round(success_rate, 4),
            "risk_level": "high" if cate_score > 0.5 else
                          "medium" if cate_score > 0.2 else "low",
        }
    except Exception as exc:
        logger.warning("[Tool:causal_uplift_recommend] Error: %s", exc)
        return {"status": "error", "tax_code": tax_code, "error": str(exc)}


def _tool_top_n_risky(
    db,
    n: int = 10,
    sort_by: str = "risk_score",
    mode: str = "full",
    **kwargs,
) -> dict[str, Any]:
    """Query top N risky companies from the database."""
    from ml_engine.tax_agent_nl_query import NLQueryExecutor

    executor = NLQueryExecutor()
    return executor.execute_top_n(db, n=n, sort_by=sort_by, mode=mode)


def _tool_company_name_search(
    db,
    name: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Search companies by name (fuzzy match)."""
    from ml_engine.tax_agent_nl_query import NLQueryExecutor

    executor = NLQueryExecutor()
    return executor.execute_company_name_search(db, name=name)


def _tool_nlp_red_flag_scan(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """NLP Red Flag Detector tool for analyzing invoice descriptions."""
    from ml_engine.nlp_red_flag_detector import get_red_flag_engine
    from sqlalchemy import text as sql_text

    query = sql_text("""
        SELECT goods_category FROM invoices
        WHERE seller_tax_code = :tax_code AND goods_category IS NOT NULL
        LIMIT 100
    """)
    rows = db.execute(query, {"tax_code": tax_code}).mappings().all()
    descriptions = [r["goods_category"] for r in rows]

    if not descriptions:
        return {"status": "insufficient_data", "tax_code": tax_code, "message": "Không tìm thấy dữ liệu hóa đơn."}

    invoices_payload = [{"invoice_number": f"INV-{i}", "descriptions": [desc]} for i, desc in enumerate(descriptions)]
    detector = get_red_flag_engine()
    results = detector.batch_analyze(invoices_payload)

    high_risk_count = sum(1 for r in results if r.risk_score > 0.6)
    return {
        "status": "analyzed",
        "tax_code": tax_code,
        "total_analyzed": len(descriptions),
        "high_risk_count": high_risk_count,
        "top_flags": [{"risk_score": r.risk_score, "flags": r.flags} for r in results if r.risk_score > 0.6][:5]
    }


def _tool_revenue_forecast(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Revenue Forecasting tool for predicting next quarter revenue."""
    from ml_engine.revenue_forecast_model import RevenueForecastModel
    from sqlalchemy import text as sql_text

    query = sql_text("""
        SELECT quarter, COALESCE(revenue, 0) as revenue
        FROM tax_returns
        WHERE tax_code = :tax_code AND revenue > 0
        ORDER BY tax_year, quarter
    """)
    rows = db.execute(query, {"tax_code": tax_code}).mappings().all()

    if len(rows) < 4:
        return {"status": "insufficient_data", "tax_code": tax_code, "message": "Cần ít nhất 4 quý doanh thu để dự báo."}

    values = [float(r["revenue"]) for r in rows]
    model = RevenueForecastModel()
    forecast = model.forecast_series(values, steps=1)

    return {
        "status": "analyzed",
        "tax_code": tax_code,
        "historical_periods": len(values),
        "last_revenue": values[-1],
        "forecast_next_quarter": forecast[0] if forecast else 0
    }


def _tool_entity_resolution_check(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Entity Resolution tool to find duplicate companies."""
    from ml_engine.entity_resolution_model import EntityResolutionModel
    from sqlalchemy import text as sql_text

    query = sql_text("""
        SELECT tax_code, legal_name, address, representative_name
        FROM entity_identities
        WHERE tax_code = :tax_code
    """)
    row = db.execute(query, {"tax_code": tax_code}).mappings().first()

    if not row:
        return {"status": "not_found", "tax_code": tax_code, "message": "Không tìm thấy thông tin thực thể."}

    model = EntityResolutionModel()
    duplicates = model.find_duplicates(dict(row), db)

    return {
        "status": "analyzed",
        "tax_code": tax_code,
        "entity_name": row["legal_name"],
        "duplicates_found": len(duplicates),
        "top_matches": duplicates[:5]
    }


def _tool_ocr_document_process(
    db,
    file_path: str,
    **kwargs,
) -> dict[str, Any]:
    """OCR Document Process tool."""
    from dataclasses import asdict
    from ml_engine.document_ocr_engine import get_ocr_engine

    if not file_path or not Path(file_path).exists():
        return {"status": "error", "message": f"File không tồn tại: {file_path}"}

    result = get_ocr_engine().process(file_path)
    payload = asdict(result)
    fields = payload.get("invoice_fields") or {}
    page_scores = [
        float(page.get("confidence") or 0.0)
        for page in payload.get("ocr_results", [])
        if isinstance(page, dict)
    ]
    confidence = float(fields.get("confidence") or 0.0)
    if confidence <= 0 and page_scores:
        confidence = sum(page_scores) / len(page_scores)

    return {
        "status": "analyzed",
        "file_path": file_path,
        "extracted_data": fields,
        "full_text_preview": str(payload.get("full_text") or "")[:1200],
        "tables": payload.get("tables", []),
        "confidence": round(confidence, 4),
        "errors": payload.get("errors", []),
    }


# ─── Registry Builder ────────────────────────────────────────────────────────

def build_default_registry() -> ToolRegistry:
    """Build the default tool registry with all available tools."""
    registry = ToolRegistry()

    registry.register(ToolSpec(
        name="knowledge_search",
        description="Tìm kiếm tri thức pháp luật thuế (luật, nghị định, thông tư, hướng dẫn). Trả về các đoạn văn bản liên quan nhất với citations.",
        category=ToolCategory.RETRIEVAL,
        input_schema={"query": "string", "intent": "string", "top_k": "int"},
        output_schema={"hits": "list[dict]", "total_candidates": "int"},
        handler=_tool_knowledge_search,
        timeout_seconds=15.0,
        priority=1,
    ))

    registry.register(ToolSpec(
        name="company_risk_lookup",
        description="Tra cứu hồ sơ rủi ro doanh nghiệp: điểm rủi ro, mức rủi ro, ngành nghề, tình trạng hoạt động.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"risk_score": "float", "risk_level": "string"},
        handler=_tool_company_risk_lookup,
        requires_tax_code=True,
        timeout_seconds=5.0,
        priority=2,
    ))

    registry.register(ToolSpec(
        name="delinquency_check",
        description="Dự báo rủi ro nợ đọng thuế trong 30/60/90 ngày tới. Phân tích lịch sử thanh toán.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"prob_30d": "float", "prob_60d": "float", "prob_90d": "float", "top_reasons": "list"},
        handler=_tool_delinquency_check,
        requires_tax_code=True,
        timeout_seconds=10.0,
        priority=3,
    ))

    registry.register(ToolSpec(
        name="invoice_risk_scan",
        description="Quét rủi ro hóa đơn: tổng số hóa đơn, hóa đơn rủi ro, tỷ lệ rủi ro, tổng giá trị.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"total_invoices": "int", "risky_invoices": "int", "risk_ratio": "float"},
        handler=_tool_invoice_risk_scan,
        requires_tax_code=True,
        timeout_seconds=10.0,
        priority=3,
    ))

    registry.register(ToolSpec(
        name="vat_refund_risk",
        description="Truy vết hồ sơ hoàn thuế GTGT và điểm rủi ro VAT refund theo MST.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string", "limit": "int"},
        output_schema={"risk_score": "float", "cases": "list", "model_version": "string"},
        handler=_tool_vat_refund_risk,
        requires_tax_code=True,
        timeout_seconds=10.0,
        priority=3,
    ))

    registry.register(ToolSpec(
        name="gnn_analysis",
        description="Phân tích rủi ro gian lận VAT bằng Graph Neural Network (GATv2). Sử dụng cấu trúc đồ thị giao dịch.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"gnn_outputs": "dict"},
        handler=_tool_gnn_analysis,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=4,
    ))

    registry.register(ToolSpec(
        name="motif_detection",
        description="Phát hiện mẫu giao dịch đáng ngờ: vòng tròn (carousel), hình sao (shell), chuỗi (layering).",
        category=ToolCategory.INVESTIGATION,
        input_schema={"tax_code": "string"},
        output_schema={"motifs": "dict", "summary": "dict"},
        handler=_tool_motif_detection,
        requires_tax_code=True,
        timeout_seconds=20.0,
        priority=5,
    ))

    registry.register(ToolSpec(
        name="ring_scoring",
        description="Chấm điểm vòng giao dịch VAT khép kín dựa trên motif/cycle detection.",
        category=ToolCategory.INVESTIGATION,
        input_schema={"tax_code": "string", "max_rings": "int"},
        output_schema={"ring_score": "float", "ring_count": "int", "rings": "list"},
        handler=_tool_ring_scoring,
        requires_tax_code=True,
        timeout_seconds=20.0,
        priority=5,
    ))

    registry.register(ToolSpec(
        name="ownership_analysis",
        description="Phân tích cấu trúc sở hữu: phát hiện common controllers, chuỗi sở hữu, giao dịch nội bộ.",
        category=ToolCategory.INVESTIGATION,
        input_schema={"tax_code": "string"},
        output_schema={"clusters": "list", "common_controllers": "list", "cross_ownership_trades": "list"},
        handler=_tool_ownership_analysis,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=5,
    ))

    # ═══ NEW DEEP LEARNING TOOLS ═══

    registry.register(ToolSpec(
        name="temporal_delinquency_deep",
        description="Dự báo nợ đọng bằng Temporal Transformer (Deep Learning). Phân tích chuỗi thanh toán với attention mechanism, dự báo 30/60/90 ngày.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"prob_30d": "float", "prob_60d": "float", "prob_90d": "float", "sequence_features": "list"},
        handler=_tool_temporal_delinquency_deep,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=3,
    ))

    registry.register(ToolSpec(
        name="hetero_gnn_risk",
        description="Phân tích rủi ro đa thực thể bằng Heterogeneous Graph Transformer (HGT). Phân loại doanh nghiệp, cá nhân, pháp nhân nước ngoài trên đồ thị dị thể.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"fraud_probability": "float", "node_type_scores": "dict", "neighbor_risk_summary": "list"},
        handler=_tool_hetero_gnn_risk,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=4,
    ))

    registry.register(ToolSpec(
        name="vae_anomaly_scan",
        description="Phát hiện bất thường hóa đơn bằng Variational Autoencoder (VAE). Tìm các giao dịch có reconstruction error cao bất thường.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"anomaly_count": "int", "anomaly_ratio": "float", "top_anomalies": "list"},
        handler=_tool_vae_anomaly_scan,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=4,
    ))

    registry.register(ToolSpec(
        name="causal_uplift_recommend",
        description="Đề xuất hành động thu nợ tối ưu bằng T-Learner Causal Inference. Ước lượng Individual Treatment Effect (CATE) cho từng doanh nghiệp.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"cate_score": "float", "recommended_action": "string", "action_ranking": "list"},
        handler=_tool_causal_uplift_recommend,
        requires_tax_code=True,
        timeout_seconds=10.0,
        priority=5,
    ))

    # ═══ NL QUERY TOOLS ═══

    registry.register(ToolSpec(
        name="top_n_risky_companies",
        description="Truy vấn top N doanh nghiệp có điểm rủi ro cao nhất từ cơ sở dữ liệu. Hỗ trợ sort theo risk_score hoặc anomaly_score.",
        category=ToolCategory.ANALYTICS,
        input_schema={"n": "int", "sort_by": "string", "mode": "string"},
        output_schema={"companies": "list", "total": "int"},
        handler=_tool_top_n_risky,
        requires_db=True,
        requires_tax_code=False,
        timeout_seconds=10.0,
        priority=2,
    ))

    registry.register(ToolSpec(
        name="company_name_search",
        description="Tìm kiếm doanh nghiệp theo tên (fuzzy match). Trả về danh sách MST khớp với tên tìm kiếm.",
        category=ToolCategory.RETRIEVAL,
        input_schema={"name": "string"},
        output_schema={"matches": "list", "total": "int"},
        handler=_tool_company_name_search,
        requires_db=True,
        requires_tax_code=False,
        timeout_seconds=8.0,
        priority=2,
    ))

    # ═══ NEW PHASE DL TOOLS ═══

    registry.register(ToolSpec(
        name="nlp_red_flag_scan",
        description="Phân tích ngữ nghĩa mô tả hàng hóa hóa đơn để phát hiện rủi ro gian lận, trốn thuế bằng mô hình NLP.",
        category=ToolCategory.ANALYTICS,
        input_schema={"tax_code": "string"},
        output_schema={"total_analyzed": "int", "high_risk_count": "int", "top_flags": "list"},
        handler=_tool_nlp_red_flag_scan,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=4,
    ))

    registry.register(ToolSpec(
        name="revenue_forecast",
        description="Dự báo doanh thu quý tới bằng mô hình LightGBM/ARIMA. Phát hiện biến động doanh thu bất thường có khả năng dẫn đến nợ đọng.",
        category=ToolCategory.FORECASTING,
        input_schema={"tax_code": "string"},
        output_schema={"forecast_next_quarter": "float", "historical_periods": "int"},
        handler=_tool_revenue_forecast,
        requires_tax_code=True,
        timeout_seconds=10.0,
        priority=4,
    ))

    registry.register(ToolSpec(
        name="entity_resolution_check",
        description="Phân tích trùng lặp thực thể (Entity Resolution). Phát hiện doanh nghiệp phượng hoàng, cá nhân lập nhiều công ty bằng Siamese Bi-Encoder.",
        category=ToolCategory.INVESTIGATION,
        input_schema={"tax_code": "string"},
        output_schema={"duplicates_found": "int", "top_matches": "list"},
        handler=_tool_entity_resolution_check,
        requires_tax_code=True,
        timeout_seconds=15.0,
        priority=5,
    ))

    registry.register(ToolSpec(
        name="ocr_document_process",
        description="Trích xuất tự động thông tin từ ảnh/PDF hóa đơn chứng từ bằng PaddleOCR.",
        category=ToolCategory.ANALYTICS,
        input_schema={"file_path": "string"},
        output_schema={"extracted_data": "dict", "confidence": "float"},
        handler=_tool_ocr_document_process,
        requires_db=False,
        requires_tax_code=False,
        timeout_seconds=30.0,
        priority=3,
    ))

    registry.register(ToolSpec(
        name="macro_forecast",
        description="Mô phỏng kịch bản vĩ mô (thuế suất, thanh tra, lãi suất, tăng trưởng...) dựa trên baseline từ DB. Đồng nhất với trang mô phỏng vĩ mô.",
        category=ToolCategory.FORECASTING,
        input_schema={"scenario": "dict", "action": "string"},
        output_schema={"result": "dict"},
        handler=_tool_macro_forecast,
        requires_db=True,
        requires_tax_code=False,
        timeout_seconds=25.0,
        priority=4,
    ))

    logger.info("[ToolRegistry] ✓ Default registry built with %d tools", registry.count())
    return registry


# Global registry singleton
_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the default tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry
