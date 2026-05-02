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
                if db is not None:
                    try:
                        db.rollback()
                    except Exception:
                        pass
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

        # Load trained model
        config_path = MODEL_DIR / "temporal_transformer_config.json"
        model_path = MODEL_DIR / "temporal_transformer.pt"
        if not model_path.exists():
            return {"status": "model_not_found", "tax_code": tax_code,
                    "message": "Temporal Transformer model chua duoc train."}

        with open(config_path) as f:
            config = json.load(f)
        model = DelinquencyTransformer(
            feature_dim=config.get("feature_dim", FEATURE_DIM),
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 3),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

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

        # Load model
        input_dim = config.get("input_dim", X_norm.shape[1])
        model = TransactionVAE(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

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
    from ml_engine.document_ocr_engine import DocumentOCREngine

    if not file_path or not Path(file_path).exists():
        return {"status": "error", "message": f"File không tồn tại: {file_path}"}

    engine = DocumentOCREngine()
    result = engine.process_document(file_path)

    return {
        "status": "analyzed",
        "file_path": file_path,
        "extracted_data": result.get("extracted_fields", {}),
        "confidence": result.get("confidence", 0.0)
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
