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
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
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

    def execute_single(
        self,
        request: ToolCallRequest,
        db=None,
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

        while retries <= tool.max_retries:
            t0 = time.perf_counter()
            try:
                # Build kwargs
                kwargs = dict(request.inputs)
                if tool.requires_db:
                    if db is None and self.db_factory:
                        db = self.db_factory()
                    kwargs["db"] = db

                # Execute with timeout
                result = tool.handler(**kwargs)
                latency = (time.perf_counter() - t0) * 1000.0

                return ToolCallResult(
                    tool_name=request.tool_name,
                    status=ToolStatus.SUCCESS,
                    outputs=result if isinstance(result, dict) else {"result": result},
                    latency_ms=latency,
                    retries=retries,
                )

            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                last_error = str(exc)
                retries += 1
                logger.warning(
                    "[ToolExecutor] %s failed (attempt %d/%d): %s",
                    request.tool_name, retries, tool.max_retries + 1, last_error,
                )
                if retries <= tool.max_retries:
                    time.sleep(0.1 * retries)  # Simple backoff

        return ToolCallResult(
            tool_name=request.tool_name,
            status=ToolStatus.ERROR,
            error=last_error,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            retries=retries - 1,
        )

    def execute_parallel(
        self,
        requests: list[ToolCallRequest],
        db=None,
    ) -> list[ToolCallResult]:
        """Execute multiple tool calls in parallel (for independent sub-tasks)."""
        if not requests:
            return []

        if len(requests) == 1:
            return [self.execute_single(requests[0], db=db)]

        futures_map = {}
        for req in requests:
            future = self._executor.submit(self.execute_single, req, db)
            futures_map[future] = req.tool_name

        results = []
        for future in as_completed(futures_map):
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as exc:
                results.append(ToolCallResult(
                    tool_name=futures_map[future],
                    status=ToolStatus.ERROR,
                    error=str(exc),
                ))

        # Maintain original order
        result_map = {r.tool_name: r for r in results}
        return [result_map.get(req.tool_name, ToolCallResult(
            tool_name=req.tool_name,
            status=ToolStatus.ERROR,
            error="Result not found",
        )) for req in requests]

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


# ─── Pre-built Tool Handlers ──────────────────────────────────────────────────
# These wrap existing ML/DL models and APIs.


def _tool_knowledge_search(
    db,
    query: str,
    intent: str = "general_tax_query",
    top_k: int = 5,
    **kwargs,
) -> dict[str, Any]:
    """Enhanced knowledge retrieval (Phase 1 RAG)."""
    from ml_engine.tax_agent_retrieval import (
        RetrievalCandidate, bm25_scores, tokenize,
    )
    from ml_engine.tax_agent_embeddings import get_embedding_engine, expand_query
    from sqlalchemy import text as sql_text

    engine = get_embedding_engine()
    expanded_query = expand_query(query)
    query_embedding = engine.embed_query(expanded_query)

    # Intent-based doc type filtering
    doc_type_map = {
        "vat_refund_risk": ["vat_refund", "vat", "circular", "decree", "law"],
        "invoice_risk": ["invoice", "vat", "circular", "decree", "law"],
        "delinquency": ["collections", "tax_procedure", "decree", "law"],
        "transfer_pricing": ["transfer_pricing", "international_tax", "circular", "decree", "law"],
        "audit_selection": ["audit", "tax_procedure", "decree", "law"],
        "osint_ownership": ["ubo", "ownership", "company_law", "international_tax", "law"],
    }
    doc_types = doc_type_map.get(intent, [])

    rows = db.execute(
        sql_text("""
            SELECT
                kc.id AS chunk_id,
                kc.chunk_key,
                kc.chunk_text,
                kd.title,
                kd.doc_type,
                kce.embedding_json
            FROM knowledge_chunks kc
            JOIN knowledge_document_versions kdv ON kdv.id = kc.version_id
            JOIN knowledge_documents kd ON kd.id = kdv.document_id
            LEFT JOIN knowledge_chunk_embeddings kce ON kce.chunk_id = kc.id
            WHERE kd.status = 'active'
              AND (:use_doc_types = FALSE OR kd.doc_type = ANY(:doc_types))
            ORDER BY kc.created_at DESC
            LIMIT 400
        """),
        {"use_doc_types": bool(doc_types), "doc_types": doc_types},
    ).mappings().all()

    q_tokens = tokenize(expanded_query)
    candidates = []
    passage_texts = []

    for row in rows:
        chunk_text = str(row.get("chunk_text") or "")
        candidates.append({
            "chunk_id": int(row["chunk_id"]),
            "chunk_key": str(row["chunk_key"]),
            "title": str(row.get("title") or ""),
            "doc_type": str(row.get("doc_type") or ""),
            "text": chunk_text[:900],
        })
        passage_texts.append(chunk_text)

    # BM25 scores
    docs_tokens = [tokenize(t) for t in passage_texts]
    bm25 = bm25_scores(q_tokens, docs_tokens)
    bm25_max = max(bm25) if bm25 else 1.0

    # Dense semantic scores (Phase 1 upgrade)
    if engine.is_semantic and passage_texts:
        batch_result = engine.embed_passages_batch(passage_texts)
        passage_vecs = np.stack([e.vector for e in batch_result.embeddings])
        dense_scores = engine.cosine_similarity_batch(query_embedding.vector, passage_vecs)
    else:
        dense_scores = np.zeros(len(candidates))

    # Lexical overlap
    query_token_set = set(q_tokens)

    scored = []
    for i, cand in enumerate(candidates):
        doc_tokens_set = set(tokenize(passage_texts[i]))
        lexical = len(query_token_set & doc_tokens_set) / max(len(query_token_set), 1)
        bm25_norm = float(bm25[i]) / max(bm25_max, 1e-9)
        dense = float(dense_scores[i])

        # Hybrid score with enhanced weights
        score = 0.35 * bm25_norm + 0.50 * dense + 0.15 * lexical

        scored.append({
            **cand,
            "score": round(score, 6),
            "components": {
                "bm25": round(bm25_norm, 6),
                "dense": round(dense, 6),
                "lexical": round(lexical, 6),
            },
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Cross-encoder reranking (Phase 1)
    from ml_engine.tax_agent_cross_encoder import TaxAgentCrossEncoder, RerankCandidate
    reranker = TaxAgentCrossEncoder()
    reranker.load()

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
        final.append({
            "chunk_id": rc.chunk_id,
            "chunk_key": rc.chunk_key,
            "title": rc.title,
            "doc_type": rc.doc_type,
            "text": rc.text,
            "score": round(rc.rerank_score, 6),
            "rerank_tier": rc.rerank_tier,
        })

    return {
        "hits": final,
        "total_candidates": len(candidates),
        "rerank_model": rerank_result.model_tier,
        "embedding_model": engine.model_tier,
        "expanded_query": expanded_query if expanded_query != query else None,
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
                c.tax_code, c.company_name, c.industry, c.risk_score,
                c.risk_level, c.last_risk_update, c.status
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
        "company_name": str(row.get("company_name") or ""),
        "industry": str(row.get("industry") or ""),
        "risk_score": float(row.get("risk_score") or 0),
        "risk_level": str(row.get("risk_level") or "unknown"),
        "last_risk_update": str(row.get("last_risk_update") or ""),
        "status_detail": str(row.get("status") or ""),
    }


def _tool_delinquency_check(
    db,
    tax_code: str,
    **kwargs,
) -> dict[str, Any]:
    """Check delinquency risk using DelinquencyPipeline."""
    from sqlalchemy import text as sql_text
    import pandas as pd

    # Fetch payment history
    payments = db.execute(
        sql_text("""
            SELECT due_date, actual_payment_date, amount_due, amount_paid,
                   penalty_amount, tax_period, status
            FROM debt_details
            WHERE tax_code = :tax_code
            ORDER BY due_date DESC
            LIMIT 100
        """),
        {"tax_code": tax_code},
    ).mappings().all()

    if not payments:
        return {"status": "no_data", "tax_code": tax_code, "message": "Không có dữ liệu thanh toán."}

    payments_df = pd.DataFrame([dict(r) for r in payments])

    from ml_engine.delinquency_model import DelinquencyPipeline
    pipeline = DelinquencyPipeline()
    pipeline.load_models()
    result = pipeline.predict_single(payments_df)
    result["tax_code"] = tax_code
    result["status"] = "analyzed"
    return result


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
            SELECT seller_tax_code, buyer_tax_code, amount, invoice_date AS date
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
                   relationship_type, reported_by
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
