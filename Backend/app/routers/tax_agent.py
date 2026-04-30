"""
tax_agent.py – Tax Agent API Router (Multi-Agent Orchestrator v2)
==================================================================
Upgraded from single-agent RAG to Multi-Agent Orchestration system.

Endpoints:
    POST /api/tax-agent/chat          – Standard chat (backward compatible)
    POST /api/tax-agent/chat/v2       – Enhanced chat with full orchestration
    GET  /api/tax-agent/chat/stream   – SSE streaming response
    GET  /api/tax-agent/status        – Agent system status
    GET  /api/tax-agent/tools         – List available tools

Architecture:
    v1 /chat: Preserved for backward compatibility (delegates to orchestrator)
    v2 /chat/v2: Full orchestrator response with tool results, reasoning, etc.
    SSE /chat/stream: Server-Sent Events for real-time response streaming
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from enum import Enum
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db
from ml_engine.model_registry import AuditContext, ModelRegistryService
from ml_engine.tax_agent_intent_model import TaxAgentIntentModel
from ml_engine.tax_agent_retrieval import RetrievalCandidate, bm25_scores, embed_hash_tfidf, hybrid_score, tokenize
from ml_engine.tax_agent_reranker import TaxAgentReranker


router = APIRouter(prefix="/api/tax-agent", tags=["Tax Agent"])

_MODEL_DIR = None
try:
    from pathlib import Path

    _MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
except Exception:
    _MODEL_DIR = None


INTENT_RULES = {
    "vat_refund_risk": ["hoan thue", "vat", "ho so hoan", "refund", "đề nghị hoàn"],
    "invoice_risk": ["hoa don", "invoice", "xuat hoa don", "mua vao", "ban ra"],
    "delinquency": ["no dong", "cham nop", "delinquency", "qua han", "thu no"],
    "osint_ownership": ["offshore", "so huu", "ubo", "phoenix", "cong ty me"],
    "transfer_pricing": ["chuyen gia", "transfer pricing", "gia giao dich lien ket", "mispricing"],
    "audit_selection": ["thanh tra", "audit", "kiem tra", "xep hang ho so"],
}


# ═══════════════════════════════════════════════════════════════════════
#  Model Modes (Claude-style analysis mode selector)
# ═══════════════════════════════════════════════════════════════════════

class ModelMode(str, Enum):
    """Analysis mode selector — each mode activates a different tool/signal profile."""
    FRAUD = "fraud"              # Gian lận thuế (XGBoost + GNN + VAE)
    VAT = "vat"                  # Hoàn thuế VAT + Invoice Risk
    DELINQUENCY = "delinquency"  # Dự báo nợ động (Transformer + Delinquency)
    MACRO = "macro"              # Mô phỏng vĩ mô (Macro Hypothesis)
    FULL = "full"                # Toàn diện (all tools)


# ═══════════════════════════════════════════════════════════════════════
#  Request / Response Models
# ═══════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: int | None = None
    message: str = Field(..., min_length=1, max_length=12000)
    top_k: int = Field(5, ge=1, le=20)
    model_mode: ModelMode = ModelMode.FULL


class ChatResponse(BaseModel):
    """Backward-compatible v1 response."""
    session_id: str
    intent: str
    confidence: float
    abstained: bool
    escalation_required: bool
    answer: str
    citations: list[dict[str, Any]]
    retrieval_context: list[dict[str, Any]]
    policy_traces: list[dict[str, Any]]


class ChatResponseV2(BaseModel):
    """Enhanced v2 response with full orchestration data."""
    session_id: str
    # Intent
    intent: str
    intent_confidence: float
    # Plan
    complexity: str
    reasoning_trace: str
    tools_used: list[str]
    plan_steps: list[dict[str, Any]]
    # Response
    answer: str
    summary: str
    citations: list[dict[str, Any]]
    recommendations: list[str]
    confidence: float
    # Governance
    abstained: bool
    escalation_required: bool
    escalation_domain: str
    compliance_warnings: list[str]
    # Context
    active_tax_code: str | None
    active_tax_period: str | None
    # Performance
    latency_ms: float
    latency_breakdown: dict[str, float]
    synthesis_tier: str
    # Traces
    policy_traces: list[dict[str, Any]]
    tool_results: dict[str, dict[str, Any]]
    # Visualization data for frontend charts
    visualization_data: dict[str, Any] = {}
    # Model mode used for this response
    model_mode: str = "full"


class AgentStatus(BaseModel):
    """Agent system status."""
    status: str
    version: str
    embedding_engine: dict[str, Any]
    tools_count: int
    tools: list[dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════
#  Legacy Helpers (kept for v1 backward compat)
# ═══════════════════════════════════════════════════════════════════════

def _tokenize(text_value: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", (text_value or "").lower()) if tok]


def _embed(text_value: str, dim: int = 96) -> np.ndarray:
    vec = np.zeros((dim,), dtype=float)
    toks = _tokenize(text_value)
    if not toks:
        return vec
    for tok in toks:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _infer_intent(message: str) -> tuple[str, float]:
    # Learned-first intent model (offline). Falls back to keyword rules.
    if _MODEL_DIR is not None:
        model = TaxAgentIntentModel(_MODEL_DIR)
        if model.load():
            intent, conf, _ = model.predict(message)
            if intent and conf >= 0.45:
                return intent, min(0.95, max(0.15, float(conf)))

    normalized = message.lower()
    best = ("general_tax_query", 0.15)
    for intent, keywords in INTENT_RULES.items():
        score = sum(1 for kw in keywords if kw in normalized)
        if score > best[1]:
            best = (intent, float(score))
    conf = min(0.95, 0.25 + 0.15 * best[1])
    if best[0] == "general_tax_query":
        conf = 0.22
    return best[0], conf


def _ensure_session(db: Session, *, session_id: str, user_id: int | None) -> None:
    row = db.execute(text("SELECT 1 FROM agent_sessions WHERE session_id = :session_id"), {"session_id": session_id}).fetchone()
    if row:
        return
    db.execute(
        text(
            """
            INSERT INTO agent_sessions (session_id, user_id, channel, status, metadata_json)
            VALUES (:session_id, :user_id, 'chat', 'active', CAST(:metadata_json AS jsonb))
            """
        ),
        {
            "session_id": session_id,
            "user_id": user_id,
            "metadata_json": json.dumps({"source": "tax_agent_router"}),
        },
    )


def _run_retrieval(db: Session, *, session_id: str, query_text: str, intent: str, top_k: int) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    query_vec = embed_hash_tfidf(query_text)
    doc_types: list[str] = []
    if intent == "vat_refund_risk":
        doc_types = ["vat_refund", "vat", "circular", "decree", "law"]
    elif intent == "invoice_risk":
        doc_types = ["invoice", "vat", "circular", "decree", "law"]
    elif intent == "delinquency":
        doc_types = ["collections", "tax_procedure", "decree", "law"]
    elif intent == "transfer_pricing":
        doc_types = ["transfer_pricing", "international_tax", "circular", "decree", "law"]
    elif intent == "audit_selection":
        doc_types = ["audit", "tax_procedure", "decree", "law"]
    elif intent == "osint_ownership":
        doc_types = ["ubo", "ownership", "company_law", "international_tax", "law"]
    rows = db.execute(
        text(
            """
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
            """
        ),
        {
            "use_doc_types": bool(doc_types),
            "doc_types": doc_types,
        },
    ).mappings().all()
    # Hybrid retrieval: BM25 + dense + lexical.
    candidates: list[RetrievalCandidate] = []
    for row in rows:
        candidates.append(
            RetrievalCandidate(
                chunk_id=int(row["chunk_id"]),
                chunk_key=str(row["chunk_key"]),
                title=str(row.get("title") or ""),
                doc_type=str(row.get("doc_type") or "") or None,
                text=str(row.get("chunk_text") or ""),
                embedding=row.get("embedding_json") if isinstance(row.get("embedding_json"), list) else None,
            )
        )
    q_tokens = tokenize(query_text)
    docs_tokens = [tokenize(c.text) for c in candidates]
    bm25 = bm25_scores(q_tokens, docs_tokens)
    bm25_max = max(bm25) if bm25 else 1.0
    query_token_set = set(q_tokens)

    scored: list[tuple[float, dict[str, Any]]] = []
    for c, s_bm25 in zip(candidates, bm25):
        emb = c.embedding
        dense = 0.0
        if isinstance(emb, list) and emb:
            vec = np.asarray(emb, dtype=float)
            dense = float(np.dot(query_vec[: len(vec)], vec[: len(query_vec)]))
        txt_tokens = set(tokenize(c.text))
        lexical = len(query_token_set.intersection(txt_tokens)) / max(1, len(query_token_set))
        s, comps = hybrid_score(
            query_text=query_text,
            candidate=c,
            bm25=(float(s_bm25) / max(bm25_max, 1e-9)),
            dense=dense,
            lexical=lexical,
        )
        scored.append(
            (
                s,
                {
                    "chunk_id": c.chunk_id,
                    "chunk_key": c.chunk_key,
                    "title": c.title,
                    "doc_type": c.doc_type,
                    "text": c.text[:900],
                    "components": comps,
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    # Rerank on top-N
    top_n = scored[: max(20, top_k)]
    items = [{"score": float(s), **payload} for s, payload in top_n]
    if _MODEL_DIR is not None:
        rr = TaxAgentReranker(_MODEL_DIR)
        rr.load()
        items = rr.rerank(items, preferred_doc_types=doc_types)
    selected = items[:top_k]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    db.execute(
        text(
            """
            INSERT INTO retrieval_logs
            (request_id, session_id, query_text, query_hash, intent, entity_scope, retrieved_chunks, retrieval_scores, top_k, latency_ms)
            VALUES
            (:request_id, :session_id, :query_text, :query_hash, :intent, CAST(:entity_scope AS jsonb), CAST(:retrieved_chunks AS jsonb), CAST(:retrieval_scores AS jsonb), :top_k, :latency_ms)
            """
        ),
        {
            "request_id": f"ret-{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "query_text": query_text,
            "query_hash": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            "intent": intent,
            "entity_scope": json.dumps({}),
            "retrieved_chunks": json.dumps([item["chunk_key"] for item in selected]),
            "retrieval_scores": json.dumps([round(float(item["score"]), 6) for item in selected]),
            "top_k": top_k,
            "latency_ms": latency_ms,
        },
    )
    return (selected, latency_ms)


def _apply_policy(
    db: Session,
    *,
    session_id: str,
    turn_id: int,
    intent: str,
    intent_conf: float,
    retrieval_context: list[dict[str, Any]],
) -> tuple[bool, bool, list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    abstain = False
    escalate = False
    rules = [
        ("min_intent_confidence", intent_conf >= 0.35, float(intent_conf), "intent confidence too low"),
        ("min_retrieval_hits", len(retrieval_context) >= 2, float(len(retrieval_context)), "insufficient legal context"),
        (
            "intent_requires_scope",
            intent not in {"general_tax_query", "transfer_pricing"} or len(retrieval_context) >= 3,
            float(len(retrieval_context)),
            "high-risk intent needs stronger citations",
        ),
    ]
    for rule_key, passed, score, reason in rules:
        decision = "allow" if passed else "block"
        if not passed:
            abstain = True
            if intent in {"audit_selection", "transfer_pricing", "vat_refund_risk"}:
                escalate = True
        payload = {"intent": intent, "retrieval_hits": len(retrieval_context)}
        db.execute(
            text(
                """
                INSERT INTO policy_execution_logs (session_id, turn_id, rule_key, decision, reason, score, payload_json)
                VALUES (:session_id, :turn_id, :rule_key, :decision, :reason, :score, CAST(:payload_json AS jsonb))
                """
            ),
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "rule_key": rule_key,
                "decision": decision,
                "reason": None if passed else reason,
                "score": score,
                "payload_json": json.dumps(payload),
            },
        )
        traces.append({"rule_key": rule_key, "decision": decision, "score": score, "reason": None if passed else reason})
    return abstain, escalate, traces


def _compose_answer(intent: str, retrieval_context: list[dict[str, Any]], abstained: bool) -> str:
    if abstained:
        return (
            "Tôi chưa đủ độ tin cậy để kết luận ngay từ dữ liệu hiện có. "
            "Bạn vui lòng cung cấp thêm bối cảnh (MST, kỳ thuế, loại hồ sơ) hoặc chuyển hồ sơ cho chuyên viên để xác minh."
        )
    snippets = [f"- {item['title']}: {item['text'][:180]}..." for item in retrieval_context[:3]]
    body = "\n".join(snippets) if snippets else "- Chưa có trích dẫn luật phù hợp."
    return (
        f"Nhận diện intent: `{intent}`.\n"
        "Dựa trên tri thức nội bộ, đây là các căn cứ ưu tiên:\n"
        f"{body}\n"
        "Khuyến nghị: đối chiếu thêm số liệu nghiệp vụ trước khi ra quyết định cuối cùng."
    )


# ═══════════════════════════════════════════════════════════════════════
#  V1 Endpoint (Backward Compatible — Legacy Single-Agent)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
def chat_tax_agent(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    V1 chat endpoint — backward compatible.
    Delegates to multi-agent orchestrator internally but returns v1 response shape.
    """
    session_id = payload.session_id or f"sess-{uuid.uuid4().hex[:10]}"

    try:
        # Try the new orchestrator first
        from ml_engine.tax_agent_orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        orch_response = orchestrator.process(
            db,
            session_id=session_id,
            message=payload.message,
            user_id=payload.user_id,
            top_k=payload.top_k,
        )

        # Map to v1 response shape
        retrieval_context = orch_response.tool_results.get("knowledge_search", {}).get("hits", [])

        return ChatResponse(
            session_id=orch_response.session_id,
            intent=orch_response.intent,
            confidence=orch_response.intent_confidence,
            abstained=orch_response.abstained,
            escalation_required=orch_response.escalation_required,
            answer=orch_response.answer,
            citations=orch_response.citations,
            retrieval_context=retrieval_context,
            policy_traces=orch_response.policy_traces,
        )

    except Exception as exc:
        # Fallback to legacy single-agent pipeline if orchestrator fails
        import logging
        logging.getLogger(__name__).warning(
            "[TaxAgent] Orchestrator failed, falling back to legacy: %s", exc,
        )
        return _legacy_chat(payload, db, session_id)


def _legacy_chat(payload: ChatRequest, db: Session, session_id: str) -> ChatResponse:
    """Legacy single-agent pipeline (fallback)."""
    _ensure_session(db, session_id=session_id, user_id=payload.user_id)

    turn_index_row = db.execute(
        text("SELECT COALESCE(MAX(turn_index), 0) FROM agent_turns WHERE session_id = :session_id"),
        {"session_id": session_id},
    ).fetchone()
    turn_index = int(turn_index_row[0] or 0) + 1
    db.execute(
        text(
            """
            INSERT INTO agent_turns (session_id, turn_index, role, message_text)
            VALUES (:session_id, :turn_index, 'user', :message_text)
            """
        ),
        {"session_id": session_id, "turn_index": turn_index, "message_text": payload.message},
    )
    turn_row = db.execute(
        text(
            """
            INSERT INTO agent_turns (session_id, turn_index, role, message_text)
            VALUES (:session_id, :turn_index, 'assistant', '')
            RETURNING id
            """
        ),
        {"session_id": session_id, "turn_index": turn_index + 1},
    ).fetchone()
    if not turn_row:
        raise HTTPException(status_code=500, detail="Không thể tạo assistant turn.")
    assistant_turn_id = int(turn_row[0])

    intent, intent_conf = _infer_intent(payload.message)
    retrieval_context, _ = _run_retrieval(
        db,
        session_id=session_id,
        query_text=payload.message,
        intent=intent,
        top_k=payload.top_k,
    )
    abstain, escalate, policy_traces = _apply_policy(
        db,
        session_id=session_id,
        turn_id=assistant_turn_id,
        intent=intent,
        intent_conf=intent_conf,
        retrieval_context=retrieval_context,
    )
    answer = _compose_answer(intent, retrieval_context, abstain)
    citations = []
    for item in retrieval_context[:3]:
        citations.append(
            {
                "chunk_key": item["chunk_key"],
                "title": item["title"],
                "score": round(float(item["score"]), 4),
            }
        )

    db.execute(
        text(
            """
            UPDATE agent_turns
            SET message_text = :message_text, normalized_intent = :normalized_intent, confidence = :confidence, citations_json = CAST(:citations_json AS jsonb)
            WHERE id = :turn_id
            """
        ),
        {
            "turn_id": assistant_turn_id,
            "message_text": answer,
            "normalized_intent": intent,
            "confidence": intent_conf,
            "citations_json": json.dumps(citations),
        },
    )
    db.execute(
        text(
            """
            INSERT INTO agent_decision_traces
            (session_id, turn_id, intent, selected_track, confidence, abstained, escalation_required, evidence_json, answer_text)
            VALUES
            (:session_id, :turn_id, :intent, :selected_track, :confidence, :abstained, :escalation_required, CAST(:evidence_json AS jsonb), :answer_text)
            """
        ),
        {
            "session_id": session_id,
            "turn_id": assistant_turn_id,
            "intent": intent,
            "selected_track": intent,
            "confidence": intent_conf,
            "abstained": abstain,
            "escalation_required": escalate,
            "evidence_json": json.dumps({"retrieval_context": retrieval_context[:5]}),
            "answer_text": answer,
        },
    )
    db.execute(
        text(
            """
            INSERT INTO agent_tool_calls (session_id, turn_id, tool_name, tool_input, tool_output, status, latency_ms)
            VALUES (:session_id, :turn_id, 'knowledge_retrieval', CAST(:tool_input AS jsonb), CAST(:tool_output AS jsonb), 'ok', :latency_ms)
            """
        ),
        {
            "session_id": session_id,
            "turn_id": assistant_turn_id,
            "tool_input": json.dumps({"query": payload.message, "intent": intent, "top_k": payload.top_k}),
            "tool_output": json.dumps({"hits": len(retrieval_context), "top_chunk": retrieval_context[0]["chunk_key"] if retrieval_context else None}),
            "latency_ms": None,
        },
    )
    db.commit()

    registry = ModelRegistryService(db)
    registry.log_inference(
        model_name="tax_agent_router",
        model_version="tax-agent-v1",
        entity_type="session",
        entity_id=session_id,
        input_features={"intent": intent, "intent_confidence": intent_conf, "query_len": len(payload.message)},
        outputs={"abstained": abstain, "escalation_required": escalate, "citations": len(citations)},
        ctx=AuditContext(request_id=f"agent-{uuid.uuid4().hex[:12]}"),
    )

    return ChatResponse(
        session_id=session_id,
        intent=intent,
        confidence=round(intent_conf, 4),
        abstained=abstain,
        escalation_required=escalate,
        answer=answer,
        citations=citations,
        retrieval_context=retrieval_context,
        policy_traces=policy_traces,
    )


# ═══════════════════════════════════════════════════════════════════════
#  V2 Endpoint (Full Multi-Agent Orchestration)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/chat/v2", response_model=ChatResponseV2)
def chat_tax_agent_v2(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    V2 chat endpoint — full multi-agent orchestration.
    Returns complete orchestration data: tools used, reasoning, recommendations, etc.
    """
    session_id = payload.session_id or f"sess-{uuid.uuid4().hex[:10]}"

    from ml_engine.tax_agent_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    orch_response = orchestrator.process(
        db,
        session_id=session_id,
        message=payload.message,
        user_id=payload.user_id,
        top_k=payload.top_k,
        model_mode=payload.model_mode.value,
    )

    return _build_v2_response(orch_response, model_mode=payload.model_mode.value)


@router.post("/chat/v2/with-file")
async def chat_with_file(
    message: str = Form(..., min_length=1, max_length=12000),
    session_id: str = Form(None),
    model_mode: str = Form("full"),
    user_id: int = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    """
    V2 chat endpoint with optional CSV file attachment.
    If a CSV file is attached, triggers inline batch analysis and merges results.
    """
    session_id = session_id or f"sess-{uuid.uuid4().hex[:10]}"
    resolved_mode = model_mode if model_mode in ("fraud", "vat", "delinquency", "macro", "full") else "full"

    from ml_engine.tax_agent_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    # If file is attached, handle batch analysis inline
    csv_results = None
    if file and file.filename and file.filename.lower().endswith(".csv"):
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > 100:
            raise HTTPException(status_code=400, detail="File quá lớn (tối đa 100MB cho chat inline)")
        csv_results = {
            "filename": file.filename,
            "content": content,
            "size_mb": round(file_size_mb, 2),
        }

    orch_response = orchestrator.process(
        db,
        session_id=session_id,
        message=message,
        user_id=user_id,
        top_k=5,
        model_mode=resolved_mode,
        csv_attachment=csv_results,
    )

    return _build_v2_response(orch_response, model_mode=resolved_mode)


# ═══════════════════════════════════════════════════════════════════════
#  V2 Streaming Endpoint (SSE — Server-Sent Events)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/chat/v2/stream")
def chat_tax_agent_v2_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    V2 chat endpoint with SSE streaming.
    Yields real-time events as the orchestrator processes each pipeline step.
    
    Event types:
    - thinking:    Step progress (intent classification, planning, etc.)
    - tool_start:  Tool execution started
    - tool_done:   Tool execution completed
    - sub_agent:   Sub-agent status update
    - text_chunk:  Partial answer text (for typewriter effect)
    - viz_data:    Visualization data for charts
    - done:        Final complete response
    """
    import json as _json

    session_id = payload.session_id or f"sess-{uuid.uuid4().hex[:10]}"

    from ml_engine.tax_agent_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    def event_generator():
        try:
            for event in orchestrator.process_streaming(
                db,
                session_id=session_id,
                message=payload.message,
                user_id=payload.user_id,
                top_k=payload.top_k,
                model_mode=payload.model_mode.value,
            ):
                event_type = event.get("event", "message")
                event_data = _json.dumps(event.get("data", {}), ensure_ascii=False, default=str)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
        except Exception as exc:
            error_data = _json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _build_v2_response(orch_response, *, model_mode: str = "full") -> dict:
    """Build ChatResponseV2-compatible dict from OrchestratorResponse."""
    return ChatResponseV2(
        session_id=orch_response.session_id,
        intent=orch_response.intent,
        intent_confidence=orch_response.intent_confidence,
        complexity=orch_response.complexity,
        reasoning_trace=orch_response.reasoning_trace,
        tools_used=orch_response.tools_used,
        plan_steps=orch_response.plan_steps,
        answer=orch_response.answer,
        summary=orch_response.summary,
        citations=orch_response.citations,
        recommendations=orch_response.recommendations,
        confidence=orch_response.confidence,
        abstained=orch_response.abstained,
        escalation_required=orch_response.escalation_required,
        escalation_domain=orch_response.escalation_domain,
        compliance_warnings=orch_response.compliance_warnings,
        active_tax_code=orch_response.active_tax_code,
        active_tax_period=orch_response.active_tax_period,
        latency_ms=orch_response.latency_ms,
        latency_breakdown=orch_response.latency_breakdown,
        synthesis_tier=orch_response.synthesis_tier,
        policy_traces=orch_response.policy_traces,
        tool_results=orch_response.tool_results,
        visualization_data=orch_response.visualization_data,
        model_mode=model_mode,
    )


# ═══════════════════════════════════════════════════════════════════════
#  SSE Streaming Endpoint
# ═══════════════════════════════════════════════════════════════════════

@router.post("/chat/stream")
def chat_tax_agent_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    SSE streaming endpoint for real-time response generation.
    Streams orchestration stages as they complete.
    """
    session_id = payload.session_id or f"sess-{uuid.uuid4().hex[:10]}"

    def event_generator():
        try:
            from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator
            from ml_engine.tax_agent_planner import TaxAgentPlanner
            from ml_engine.tax_agent_memory import ConversationMemory
            from ml_engine.tax_agent_tools import get_tool_registry, ToolExecutor, ToolCallRequest
            from ml_engine.tax_agent_synthesis import TaxAgentSynthesizer
            from ml_engine.tax_agent_compliance_gate import TaxAgentComplianceGate
            from ml_engine.tax_agent_intent_model import TaxAgentIntentModel
            from ml_engine.tax_agent_embeddings import get_embedding_engine

            model_dir = Path(__file__).resolve().parents[2] / "data" / "models"

            # Stage 1: Intent
            yield _sse_event("stage", {"stage": "intent", "status": "running"})
            intent_model = TaxAgentIntentModel(model_dir)
            intent_model.load()
            intent, conf, _ = intent_model.predict(payload.message)
            conf = min(0.95, max(0.15, float(conf)))
            yield _sse_event("intent", {
                "intent": intent,
                "confidence": round(conf, 4),
            })

            # Stage 2: Memory & Context
            yield _sse_event("stage", {"stage": "context", "status": "running"})
            memory = ConversationMemory(db)
            context = memory.build_context(session_id, 0, payload.message)
            yield _sse_event("context", {
                "active_tax_code": context.active_tax_code,
                "active_tax_period": context.active_tax_period,
                "entities_count": len(context.active_entities),
            })

            # Stage 3: Planning
            yield _sse_event("stage", {"stage": "planning", "status": "running"})
            planner = TaxAgentPlanner()
            plan = planner.plan(
                query=payload.message,
                intent=intent,
                intent_confidence=conf,
                tax_code=context.active_tax_code,
                tax_period=context.active_tax_period,
            )
            yield _sse_event("plan", {
                "complexity": plan.complexity.value,
                "steps": [s.tool_name for s in plan.steps],
                "reasoning": plan.reasoning,
            })

            # Stage 4: Tool Execution (stream each tool result)
            yield _sse_event("stage", {"stage": "tools", "status": "running"})
            registry = get_tool_registry()
            executor = ToolExecutor(registry)
            all_tool_results = {}

            for stage_idx, stage in enumerate(plan.get_stages()):
                for step in stage:
                    yield _sse_event("tool_start", {"tool": step.tool_name})
                    request = ToolCallRequest(
                        tool_name=step.tool_name,
                        inputs=step.tool_inputs,
                    )
                    result = executor.execute_single(request, db=db)
                    all_tool_results[step.tool_name] = {
                        "status": result.status.value,
                        **(result.outputs or {}),
                    }
                    yield _sse_event("tool_done", {
                        "tool": step.tool_name,
                        "status": result.status.value,
                        "latency_ms": round(result.latency_ms, 1),
                    })

            # Stage 5: Synthesis
            yield _sse_event("stage", {"stage": "synthesis", "status": "running"})
            synthesizer = TaxAgentSynthesizer()
            synthesis_result = synthesizer.synthesize(
                query=payload.message,
                intent=intent,
                tool_results=all_tool_results,
                reasoning_trace=plan.reasoning,
                tax_code=context.active_tax_code,
            )
            answer = synthesizer.format_response_text(synthesis_result)

            # Stage 6: Stream the final answer
            yield _sse_event("answer", {
                "answer": answer,
                "summary": synthesis_result.summary,
                "confidence": synthesis_result.confidence,
                "citations": [
                    {"title": e.title, "citation_key": e.citation_key}
                    for e in synthesis_result.evidence[:5]
                    if e.source_type == "legal"
                ],
                "recommendations": synthesis_result.recommendations,
            })

            yield _sse_event("done", {
                "session_id": session_id,
                "tools_used": list(all_tool_results.keys()),
                "complexity": plan.complexity.value,
            })

        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ═══════════════════════════════════════════════════════════════════════
#  Status & Tools Endpoints
# ═══════════════════════════════════════════════════════════════════════

@router.get("/status")
def agent_status():
    """Get agent system status — embedding engine, tools, versions."""
    try:
        from ml_engine.tax_agent_embeddings import get_embedding_engine
        from ml_engine.tax_agent_tools import get_tool_registry

        engine = get_embedding_engine()
        registry = get_tool_registry()

        return {
            "status": "operational",
            "version": "multi-agent-v2",
            "embedding_engine": engine.status(),
            "tools_count": registry.count(),
            "tools": [
                {"name": t.name, "category": t.category.value, "enabled": t.enabled}
                for t in registry.list_tools(enabled_only=False)
            ],
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "version": "multi-agent-v2",
            "error": str(exc),
        }


@router.get("/tools")
def list_agent_tools():
    """List all available tools with descriptions."""
    try:
        from ml_engine.tax_agent_tools import get_tool_registry

        registry = get_tool_registry()
        return {
            "tools": registry.get_tool_descriptions(),
            "total": registry.count(),
        }
    except Exception as exc:
        return {"tools": [], "total": 0, "error": str(exc)}
