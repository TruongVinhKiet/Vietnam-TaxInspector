"""
tax_agent_orchestrator.py – Central Multi-Agent Orchestrator (Phase 2+3+4)
===========================================================================
The brain of the Tax Intelligence Multi-Agent System.

Upgraded in Phase 3+4:
    - Enhanced intent classification (semantic + multi-intent)
    - Sub-agent dispatch (Legal Research, Analytics, Investigation)
    - Enriched synthesis with domain-specific analysis

Architecture:
    User Message
         ↓
    [Orchestrator]
         ↓
    1. Memory → Build conversation context
    2. Intent Classifier → Classify intent + extract entities
    3. Planner → Generate execution plan (DAG)
    4. Tool Executor → Execute tools (parallel/sequential)
    5. Synthesizer → Generate grounded response
    6. Compliance Gate → Safety + policy check
    7. Audit Trail → Log everything
         ↓
    Response to User

Designed for:
    - Deterministic execution (auditable, reproducible)
    - Graceful degradation (fallback at every tier)
    - Full governance compliance (tax authority requirements)
    - Future: Custom LLM integration for synthesis
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text as sql_text

from ml_engine.tax_agent_mode_contracts import (
    AGENT_RESPONSE_SCHEMA_VERSION,
    MODE_CONTRACT_VERSION,
    AgentModeContractRegistry,
    canonical_run_state,
)
from ml_engine.tax_agent_text_normalization import normalize_vietnamese_text
from ml_engine.tax_agent_turns import AgentTurnRepository

logger = logging.getLogger(__name__)


def _json_sanitize(value: Any) -> Any:
    """Recursively sanitize payload so it is valid JSON/JSONB."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return value


def _json_dumps_safe(value: Any, *, ensure_ascii: bool = False) -> str:
    """JSON dump that never emits NaN/Infinity tokens."""
    return json.dumps(_json_sanitize(value), ensure_ascii=ensure_ascii, default=str, allow_nan=False)


def _detect_csv_columns(csv_content: bytes) -> set[str]:
    """Parse CSV header and return normalized column names."""
    try:
        text = csv_content.decode("utf-8-sig", errors="replace")
        first_line = text.splitlines()[0] if text else ""
        return {col.strip().lower().strip('"').strip("'") for col in first_line.split(",") if col.strip()}
    except Exception:
        return set()


def _is_vat_invoice_schema(columns: set[str]) -> bool:
    """Heuristic: detect VAT invoice-like CSV by marker columns."""
    vat_markers = {"seller_tax_code", "buyer_tax_code", "invoice_date", "vat_amount", "invoice_number"}
    return len(columns & vat_markers) >= 2


def _normalize_mst(value: Any) -> str:
    raw = re.sub(r"\D+", "", str(value or ""))
    return raw[:10] if len(raw) >= 10 else raw


_MST_IN_CHAT_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]?\d{10})(?:-\d{3})?(?![0-9])"
)


def _first_mst_from_message(message: str) -> str | None:
    """
    MST/code mentioned in plain text. Allows one leading letter immediately before the 10-digit
    core (e.g. seller_tax_code-style like S2900010280).
    """
    m = _MST_IN_CHAT_RE.search(message or "")
    if not m:
        return None
    return _normalize_mst(m.group(1))


# ═══════════════════════════════════════════════════════════════════════
#  Model Mode Profiles — tool/sub-agent selection per mode
# ═══════════════════════════════════════════════════════════════════════

MODE_TOOL_PROFILES: dict[str, dict[str, Any]] = {
    "fraud": {
        "required_tools": [
            "company_risk_lookup", "invoice_risk_scan",
            "gnn_analysis", "hetero_gnn_risk", "vae_anomaly_scan",
            "motif_detection", "ownership_analysis",
            "nlp_red_flag_scan",
        ],
        "optional_tools": ["temporal_delinquency_deep", "ring_scoring",
                           "entity_resolution_check"],
        "sub_agents": ["analytics", "investigation"],
        "label": "🔍 Phân tích Gian lận",
    },
    "vat": {
        "required_tools": [
            "company_risk_lookup", "invoice_risk_scan",
            "vat_refund_risk", "vae_anomaly_scan",
            "nlp_red_flag_scan",
        ],
        "optional_tools": ["gnn_analysis", "motif_detection", "ring_scoring", "ownership_analysis"],
        "sub_agents": ["analytics", "investigation"],
        "label": "📄 Rủi ro VAT",
    },
    "delinquency": {
        "required_tools": [
            "company_risk_lookup", "delinquency_check",
            "temporal_delinquency_deep", "causal_uplift_recommend",
            "revenue_forecast",
        ],
        "optional_tools": [],
        "sub_agents": ["analytics"],
        "label": "📊 Dự báo Nợ động",
    },
    "macro": {
        "required_tools": ["macro_forecast"],
        "optional_tools": ["revenue_forecast"],
        "sub_agents": ["analytics"],
        "label": "🌐 Mô phỏng Vĩ mô",
    },
    "legal": {
        "required_tools": ["knowledge_search", "company_risk_lookup"],
        "optional_tools": ["nlp_red_flag_scan"],
        "sub_agents": ["legal"],
        "label": "⚖️ Tư vấn Pháp lý",
    },
    "full": {
        "required_tools": None,  # Auto-router supplies the active domain scope
        "optional_tools": [],
        "sub_agents": ["legal", "analytics", "investigation"],
        "label": "⚡ Toàn diện",
    },
}


def _mode_tool_profile_from_registry(mode: str) -> dict[str, Any]:
    """Use AgentModeContractRegistry as the source of truth for tool scope."""
    contract = AgentModeContractRegistry.get(mode)
    sub_agents_by_panel = {
        "legal": ["legal"],
        "fraud": ["analytics", "investigation"],
        "vat": ["analytics", "investigation"],
        "delinquency": ["analytics"],
        "simulation": ["analytics"],
        "auto": ["legal", "analytics", "investigation"],
    }
    return {
        "required_tools": None if contract.allowed_tools is None else sorted(contract.allowed_tools),
        "optional_tools": [],
        "sub_agents": sub_agents_by_panel.get(contract.workspace_panel, ["analytics"]),
        "label": contract.label,
        "selected_model_bundle": list(contract.selected_model_bundle),
    }


# Compatibility shim for older planner code. The registry above is the source
# of truth; this dict is rebuilt from it so "full" means Auto Orchestrator, not
# "run every tool".
MODE_TOOL_PROFILES = {
    mode: _mode_tool_profile_from_registry(mode)
    for mode in ("fraud", "vat", "delinquency", "macro", "legal", "full")
}


@dataclass
class OrchestratorResponse:
    """Complete response from the orchestrator."""
    session_id: str
    # Intent
    intent: str
    intent_confidence: float
    # Plan
    complexity: str
    reasoning_trace: str
    tools_used: list[str]
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
    # Metadata
    latency_ms: float
    latency_breakdown: dict[str, float]
    tool_results: dict[str, dict[str, Any]]
    synthesis_tier: str
    verification: dict[str, Any] = field(default_factory=dict)
    clarification_needed: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    # Full traces for audit
    policy_traces: list[dict[str, Any]] = field(default_factory=list)
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    # Chart-ready visualization data for frontend
    visualization_data: dict[str, Any] = field(default_factory=dict)
    # Model mode used
    model_mode: str = "full"
    # Routing/focus metadata
    dialogue_act: str = "task"
    answer_contract: str = "risk_profile"
    routing_decision: dict[str, Any] = field(default_factory=dict)
    focus_score: float = 1.0
    route_violation: bool = False
    selected_model_bundle: list[str] = field(default_factory=list)
    mode_validation: dict[str, Any] = field(default_factory=dict)
    mode_mismatch: bool = False
    suggested_mode: str | None = None
    suppressed_domains: list[str] = field(default_factory=list)
    analysis_blocks: list[dict[str, Any]] = field(default_factory=list)
    legal_workspace: dict[str, Any] = field(default_factory=dict)
    simulation_workspace: dict[str, Any] = field(default_factory=dict)
    schema_version: str = AGENT_RESPONSE_SCHEMA_VERSION
    mode_contract_version: str = MODE_CONTRACT_VERSION
    run_id: str = ""
    run_state: str = "finalized"
    mode_workspace: dict[str, Any] = field(default_factory=dict)
    error_detail: dict[str, Any] = field(default_factory=dict)
    intent_model_version: str = ""
    planner_policy_version: str = ""
    debate_session_id: str | None = None
    graph_reasoning_path: list[dict[str, Any]] = field(default_factory=list)


class TaxAgentOrchestrator:
    """
    Central orchestrator for the Tax Intelligence Multi-Agent System.

    Coordinates:
    - EnhancedIntentClassifier (semantic intent, multi-intent, entities)
    - ConversationMemory (context management)
    - TaxAgentPlanner (task decomposition)
    - ToolExecutor (tool execution)
    - Sub-Agents: LegalResearchAgent, AnalyticsAgent, InvestigationAgent
    - TaxAgentSynthesizer (response generation)
    - TaxAgentComplianceGate (policy enforcement)

    Usage:
        orchestrator = TaxAgentOrchestrator()
        response = orchestrator.process(
            db=db,
            session_id="...",
            message="...",
            user_id=1,
            top_k=5,
        )
    """

    def __init__(self):
        self._intent_model = None
        self._enhanced_intent = None
        self._planner = None
        self._tool_registry = None
        self._tool_executor = None
        self._synthesizer = None
        self._compliance_gate = None
        self._memory = None
        self._embedding_engine = None
        self._task_router = None
        # Phase 3: Sub-agents
        self._legal_agent = None
        self._analytics_agent = None
        self._investigation_agent = None
        # Enterprise v2: Debate + GraphRAG
        self._debate_engine = None
        self._graph_reasoner = None
        # Enterprise v3: Agentic LLM (LoRA V4)
        self._agentic_llm = None
        self._initialized = False

    def _ensure_initialized(self, db=None):
        """Lazy initialization of all components."""
        if self._initialized:
            return

        from pathlib import Path
        model_dir = Path(__file__).resolve().parent.parent / "data" / "models"

        # Phase 4: Enhanced Intent Classifier (semantic + multi-intent)
        try:
            from ml_engine.tax_agent_enhanced_intent import get_intent_classifier
            self._enhanced_intent = get_intent_classifier()
            logger.info("[Orchestrator] Enhanced intent: tier=%s", self._enhanced_intent.tier)
        except Exception as exc:
            logger.warning("[Orchestrator] Enhanced intent failed, using legacy: %s", exc)

        # Legacy Intent Model (fallback)
        from ml_engine.tax_agent_intent_model import TaxAgentIntentModel
        self._intent_model = TaxAgentIntentModel(model_dir)
        self._intent_model.load()

        # Planner
        from ml_engine.tax_agent_planner import TaxAgentPlanner
        self._planner = TaxAgentPlanner()

        # Focus router
        from ml_engine.tax_agent_task_router import TaskRouter
        self._task_router = TaskRouter()

        # Tool Registry & Executor
        from ml_engine.tax_agent_tools import get_tool_registry, ToolExecutor
        self._tool_registry = get_tool_registry()
        try:
            from app.database import SessionLocal
        except Exception:
            SessionLocal = None
        self._tool_executor = ToolExecutor(self._tool_registry, db_factory=SessionLocal)

        # Synthesizer
        from ml_engine.tax_agent_synthesis import TaxAgentSynthesizer
        self._synthesizer = TaxAgentSynthesizer()

        # Compliance Gate
        from ml_engine.tax_agent_compliance_gate import TaxAgentComplianceGate
        self._compliance_gate = TaxAgentComplianceGate(db)

        # Memory
        from ml_engine.tax_agent_memory import ConversationMemory
        self._memory = ConversationMemory(db)

        # Embedding Engine (pre-load for retrieval)
        from ml_engine.tax_agent_embeddings import get_embedding_engine
        self._embedding_engine = get_embedding_engine()

        # Phase 3: Specialized Sub-Agents
        from ml_engine.tax_agent_legal_research import LegalResearchAgent
        self._legal_agent = LegalResearchAgent()

        from ml_engine.tax_agent_analytics_agent import AnalyticsAgent
        self._analytics_agent = AnalyticsAgent()

        from ml_engine.tax_agent_investigation_agent import InvestigationAgent
        self._investigation_agent = InvestigationAgent()

        # Enterprise v2: Multi-Agent Debate Engine
        try:
            from ml_engine.tax_agent_debate import MultiAgentDebateEngine
            self._debate_engine = MultiAgentDebateEngine()
            logger.info("[Orchestrator] ✓ Debate engine loaded")
        except Exception as exc:
            logger.warning("[Orchestrator] Debate engine not available: %s", exc)

        # Enterprise v2: Legal GraphRAG Reasoner
        try:
            from ml_engine.tax_agent_legal_graph_reasoner import LegalGraphReasoner
            self._graph_reasoner = LegalGraphReasoner()
            logger.info("[Orchestrator] ✓ Legal GraphRAG reasoner loaded")
        except Exception as exc:
            logger.warning("[Orchestrator] GraphRAG reasoner not available: %s", exc)

        # Enterprise v3: Agentic LLM (LoRA V4)
        try:
            from ml_engine.tax_agent_agentic_llm import get_agentic_llm
            self._agentic_llm = get_agentic_llm()
            loaded = self._agentic_llm.load()
            if loaded:
                logger.info("[Orchestrator] ✓ Agentic LLM V4 loaded")
            else:
                logger.info("[Orchestrator] Agentic LLM V4 not available")
        except Exception as exc:
            logger.warning("[Orchestrator] Agentic LLM init failed: %s", exc)

        self._initialized = True
        logger.info(
            "[Orchestrator] ✓ Initialized all components "
            "(tools=%d, embedding=%s, sub_agents=3, agentic_llm=%s)",
            self._tool_registry.count(),
            self._embedding_engine.model_tier,
            self._agentic_llm.is_available if self._agentic_llm else False,
        )

    def process(
        self,
        db,
        *,
        session_id: str,
        message: str,
        user_id: int | None = None,
        top_k: int = 5,
        model_mode: str = "full",
        csv_attachment: dict | None = None,
        attachment_analysis: dict | None = None,
        simulation_params: dict[str, Any] | None = None,
    ) -> OrchestratorResponse:
        """
        Process a user message through the full multi-agent pipeline.

        Flow:
        1. Build context (memory)
        2. Classify intent
        3. Generate execution plan
        4. Execute tools
        5. Synthesize response
        6. Run compliance checks
        7. Log everything
        """
        done_payload: dict[str, Any] | None = None
        for event in self.process_streaming(
            db,
            session_id=session_id,
            message=message,
            user_id=user_id,
            top_k=top_k,
            model_mode=model_mode,
            csv_attachment=csv_attachment,
            attachment_analysis=attachment_analysis,
            simulation_params=simulation_params,
        ):
            if event.get("event") == "done":
                done_payload = event.get("data", {})

        if not done_payload:
            raise RuntimeError("TaxAgentOrchestrator finished without a done payload")

        return OrchestratorResponse(
            session_id=done_payload.get("session_id", session_id),
            intent=done_payload.get("intent", "general_tax_query"),
            intent_confidence=float(done_payload.get("intent_confidence", 0.0)),
            complexity=done_payload.get("complexity", "simple"),
            reasoning_trace=done_payload.get("reasoning_trace", ""),
            tools_used=done_payload.get("tools_used", []),
            answer=done_payload.get("answer", ""),
            summary=done_payload.get("summary", ""),
            citations=done_payload.get("citations", []),
            recommendations=done_payload.get("recommendations", []),
            confidence=float(done_payload.get("confidence", 0.0)),
            abstained=bool(done_payload.get("abstained", False)),
            escalation_required=bool(done_payload.get("escalation_required", False)),
            escalation_domain=done_payload.get("escalation_domain", "none"),
            compliance_warnings=done_payload.get("compliance_warnings", []),
            active_tax_code=done_payload.get("active_tax_code"),
            active_tax_period=done_payload.get("active_tax_period"),
            latency_ms=float(done_payload.get("latency_ms", 0.0)),
            latency_breakdown=done_payload.get("latency_breakdown", {}),
            tool_results=done_payload.get("tool_results", {}),
            synthesis_tier=done_payload.get("synthesis_tier", "template"),
            verification=done_payload.get("verification", {}),
            clarification_needed=bool(done_payload.get("clarification_needed", False)),
            clarification_questions=done_payload.get("clarification_questions", []),
            policy_traces=done_payload.get("policy_traces", []),
            plan_steps=done_payload.get("plan_steps", []),
            visualization_data=done_payload.get("visualization_data", {}),
            model_mode=model_mode,
            dialogue_act=done_payload.get("dialogue_act", "task"),
            answer_contract=done_payload.get("answer_contract", "risk_profile"),
            routing_decision=done_payload.get("routing_decision", {}),
            focus_score=float(done_payload.get("focus_score", 1.0)),
            route_violation=bool(done_payload.get("route_violation", False)),
            selected_model_bundle=done_payload.get("selected_model_bundle", []),
            mode_validation=done_payload.get("mode_validation", {}),
            mode_mismatch=bool(done_payload.get("mode_mismatch", False)),
            suggested_mode=done_payload.get("suggested_mode"),
            suppressed_domains=done_payload.get("suppressed_domains", []),
            analysis_blocks=done_payload.get("analysis_blocks", []),
            legal_workspace=done_payload.get("legal_workspace", {}),
            simulation_workspace=done_payload.get("simulation_workspace", {}),
            schema_version=done_payload.get("schema_version", AGENT_RESPONSE_SCHEMA_VERSION),
            mode_contract_version=done_payload.get("mode_contract_version", MODE_CONTRACT_VERSION),
            run_id=done_payload.get("run_id", ""),
            run_state=canonical_run_state(done_payload.get("run_state", "finalized")),
            mode_workspace=done_payload.get("mode_workspace", {}),
            error_detail=done_payload.get("error_detail", {}),
            intent_model_version=done_payload.get("intent_model_version", ""),
            planner_policy_version=done_payload.get("planner_policy_version", ""),
            debate_session_id=done_payload.get("debate_session_id"),
            graph_reasoning_path=done_payload.get("graph_reasoning_path", []),
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  STREAMING VERSION — yields SSE events at each pipeline step
    # ═══════════════════════════════════════════════════════════════════════

    def process_streaming(
        self,
        db,
        session_id: str,
        message: str,
        user_id: int | None = None,
        top_k: int = 5,
        model_mode: str = "full",
        csv_attachment: dict | None = None,
        attachment_analysis: dict | None = None,
        simulation_params: dict[str, Any] | None = None,
    ):
        """
        Streaming version of process(). Yields SSE event dicts:
        
        {"event": "thinking",    "data": {"step": "...", "detail": "..."}}
        {"event": "tool_start",  "data": {"tool": "...", "description": "..."}}
        {"event": "tool_done",   "data": {"tool": "...", "status": "...", "latency_ms": ...}}
        {"event": "sub_agent",   "data": {"agent": "...", "status": "..."}}
        {"event": "text_chunk",  "data": {"chunk": "..."}}
        {"event": "viz_data",    "data": {...}}
        {"event": "done",        "data": {...full response...}}
        """
        import json as _json

        t_total_start = time.perf_counter()
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        latency_breakdown: dict[str, float] = {}

        self._ensure_initialized(db)
        self._compliance_gate.db = db
        self._memory.db = db
        yield {"event": "state", "data": {"run_id": run_id, "run_state": "queued", "model_mode": model_mode}}

        # ─── Step 1: Context ────────────────────────────────────────────
        yield {"event": "thinking", "data": {"step": "context", "detail": "Đang xây dựng ngữ cảnh hội thoại..."}}
        t0 = time.perf_counter()
        self._ensure_session(db, session_id=session_id, user_id=user_id)
        turn_repo = AgentTurnRepository(db)
        turn_pair = turn_repo.allocate_turn_pair(
            session_id=session_id,
            user_message=message,
        )
        turn_index = turn_pair.user_turn_index
        assistant_turn_id = turn_pair.assistant_turn_id

        context = self._memory.build_context(session_id, turn_index, message, model_mode=model_mode)
        latency_breakdown["context"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 1.5: Conversation Intelligence ─────────────────────────
        conv_intel_result = None
        try:
            from ml_engine.tax_agent_conversation_intelligence import ConversationIntelligence
            conv_intel = ConversationIntelligence()
            conv_intel_result = conv_intel.process(
                message=message,
                active_tax_code=context.active_tax_code,
                recent_turns=context.recent_turns,
                active_entities=context.active_entities,
                intent_history=context.active_intent_history,
                has_attachment=bool(attachment_analysis),
            )
            if conv_intel_result.resolved_message != message:
                message = conv_intel_result.resolved_message
                yield {"event": "thinking", "data": {
                    "step": "conv_intel", "detail": f"Đã giải quyết ngữ cảnh: {message[:60]}...",
                }}
            if not getattr(conv_intel_result, "should_plan", True):
                answer = (
                    conv_intel_result.direct_response
                    or conv_intel_result.clarification_prompt
                    or "Bạn có thể cung cấp thêm thông tin để tôi hỗ trợ chính xác hơn."
                )
                dialogue_act = getattr(conv_intel_result, "dialogue_act", "smalltalk")
                direct_intent = "clarification" if dialogue_act == "clarification" else "smalltalk"
                from ml_engine.tax_agent_task_router import AnswerContract, RoutingDecision
                routing_decision = RoutingDecision(
                    intent=direct_intent,
                    answer_contract=(
                        AnswerContract.CLARIFICATION
                        if direct_intent == "clarification" else AnswerContract.SMALLTALK
                    ),
                    allowed_tools=set(),
                    allow_legal=False,
                    route_confidence=0.98,
                    reason=dialogue_act,
                )
                total_latency = (time.perf_counter() - t_total_start) * 1000.0
                try:
                    turn_repo.update_assistant_turn(
                        turn_id=assistant_turn_id,
                        message_text=answer,
                        normalized_intent=direct_intent,
                        confidence=0.98,
                        citations_json="[]",
                    )
                    db.execute(sql_text("""
                        INSERT INTO agent_decision_traces
                        (session_id, turn_id, intent, selected_track, confidence,
                         abstained, escalation_required, evidence_json, answer_text)
                        VALUES (:session_id, :turn_id, :intent, :selected_track, :confidence,
                         FALSE, FALSE, CAST(:evidence_json AS jsonb), :answer_text)
                    """), {
                        "session_id": session_id,
                        "turn_id": assistant_turn_id,
                        "intent": direct_intent,
                        "selected_track": "direct",
                        "confidence": 0.98,
                        "evidence_json": _json_dumps_safe({
                            "dialogue_act": dialogue_act,
                            "routing": routing_decision.to_dict(),
                        }),
                        "answer_text": answer,
                    })
                    self._persist_route_event(
                        db,
                        session_id=session_id,
                        turn_id=assistant_turn_id,
                        dialogue_act=dialogue_act,
                        intent=direct_intent,
                        model_mode=model_mode,
                        selected_tools=[],
                        routing_decision=routing_decision,
                    )
                    db.commit()
                except Exception as exc:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    logger.debug("[Orchestrator:stream] Direct dialogue persist skipped: %s", exc)

                payload = {
                    "schema_version": AGENT_RESPONSE_SCHEMA_VERSION,
                    "mode_contract_version": MODE_CONTRACT_VERSION,
                    "run_id": run_id,
                    "run_state": "finalized",
                    "session_id": session_id,
                    "intent": direct_intent,
                    "intent_confidence": 0.98,
                    "complexity": "direct",
                    "reasoning_trace": f"Direct dialogue act: {dialogue_act}",
                    "tools_used": [],
                    "answer": answer,
                    "summary": answer,
                    "citations": [],
                    "recommendations": [],
                    "confidence": 0.98,
                    "abstained": False,
                    "escalation_required": False,
                    "escalation_domain": "none",
                    "compliance_warnings": [],
                    "active_tax_code": context.active_tax_code,
                    "active_tax_period": context.active_tax_period,
                    "latency_ms": round(total_latency, 1),
                    "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
                    "synthesis_tier": "direct",
                    "verification": {},
                    "clarification_needed": direct_intent == "clarification",
                    "clarification_questions": [answer] if direct_intent == "clarification" else [],
                    "tool_results": {},
                    "policy_traces": [],
                    "visualization_data": {},
                    "model_mode": model_mode,
                    "plan_steps": [],
                    "dialogue_act": dialogue_act,
                    "answer_contract": routing_decision.answer_contract.value,
                    "routing_decision": routing_decision.to_dict(),
                    "focus_score": routing_decision.focus_score,
                    "route_violation": routing_decision.route_violation,
                    "selected_model_bundle": routing_decision.selected_model_bundle,
                    "mode_validation": routing_decision.mode_validation,
                    "mode_mismatch": routing_decision.mode_mismatch,
                    "suggested_mode": routing_decision.suggested_mode,
                    "suppressed_domains": routing_decision.suppressed_domains,
                    "analysis_blocks": [],
                    "mode_workspace": self._build_mode_workspace(
                        model_mode=model_mode,
                        legal_workspace={},
                        simulation_workspace={},
                        tool_results={},
                        viz_data={},
                        analysis_blocks=[],
                    ),
                    "error_detail": {},
                }
                try:
                    from ml_engine.tax_agent_telemetry import get_telemetry
                    get_telemetry().record_from_orchestrator(OrchestratorResponse(
                        session_id=session_id,
                        intent=direct_intent,
                        intent_confidence=0.98,
                        complexity="direct",
                        reasoning_trace=payload["reasoning_trace"],
                        tools_used=[],
                        answer=answer,
                        summary=answer,
                        citations=[],
                        recommendations=[],
                        confidence=0.98,
                        abstained=False,
                        escalation_required=False,
                        escalation_domain="none",
                        compliance_warnings=[],
                        active_tax_code=context.active_tax_code,
                        active_tax_period=context.active_tax_period,
                        latency_ms=total_latency,
                        latency_breakdown=latency_breakdown,
                        tool_results={},
                        synthesis_tier="direct",
                        dialogue_act=dialogue_act,
                        answer_contract=routing_decision.answer_contract.value,
                        routing_decision=routing_decision.to_dict(),
                        selected_model_bundle=routing_decision.selected_model_bundle,
                        mode_validation=routing_decision.mode_validation,
                        mode_mismatch=routing_decision.mode_mismatch,
                        suggested_mode=routing_decision.suggested_mode,
                        suppressed_domains=routing_decision.suppressed_domains,
                    ))
                except Exception as exc:
                    logger.debug("[Orchestrator:stream] Direct telemetry skipped: %s", exc)
                yield {"event": "text_chunk", "data": {"chunk": answer}}
                yield {"event": "done", "data": payload}
                return
            if conv_intel_result.is_ambiguous and conv_intel_result.clarification_prompt:
                # Return clarification instead of running full pipeline
                try:
                    turn_repo.update_assistant_turn(
                        turn_id=assistant_turn_id,
                        message_text=conv_intel_result.clarification_prompt,
                        normalized_intent="clarification",
                        confidence=0.8,
                        citations_json="[]",
                    )
                    db.execute(
                        sql_text(
                            """
                            INSERT INTO agent_decision_traces
                            (session_id, turn_id, intent, selected_track, confidence,
                             abstained, escalation_required, evidence_json, answer_text)
                            VALUES (:session_id, :turn_id, 'clarification', 'clarification', :confidence,
                             FALSE, FALSE, CAST(:evidence_json AS jsonb), :answer_text)
                            """
                        ),
                        {
                            "session_id": session_id,
                            "turn_id": assistant_turn_id,
                            "confidence": 0.8,
                            "evidence_json": _json_dumps_safe({"is_ambiguous": True}),
                            "answer_text": conv_intel_result.clarification_prompt,
                        },
                    )
                    db.commit()
                except Exception as exc:
                    logger.debug("[Orchestrator:stream] Clarification persistence skipped: %s", exc)
                yield {"event": "text_chunk", "data": {"chunk": conv_intel_result.clarification_prompt}}
                yield {"event": "done", "data": {
                    "schema_version": AGENT_RESPONSE_SCHEMA_VERSION,
                    "mode_contract_version": MODE_CONTRACT_VERSION,
                    "run_id": run_id,
                    "run_state": "finalized",
                    "session_id": session_id, "answer": conv_intel_result.clarification_prompt,
                    "intent": "clarification", "is_ambiguous": True,
                    "mode_workspace": self._build_mode_workspace(
                        model_mode=model_mode,
                        legal_workspace={},
                        simulation_workspace={},
                        tool_results={},
                        viz_data={},
                        analysis_blocks=[],
                    ),
                    "error_detail": {},
                }}
                return
        except Exception as exc:
            logger.debug("[Orchestrator:stream] ConvIntel error: %s", exc)

        # ─── Step 2: Intent Classification ──────────────────────────────
        yield {"event": "thinking", "data": {"step": "intent", "detail": "Đang phân loại ý định..."}}
        t0 = time.perf_counter()
        multi_intent_result = None
        intent_meta = {}

        if self._enhanced_intent is not None:
            try:
                multi_intent_result = self._enhanced_intent.classify(
                    message, context_intents=context.active_intent_history,
                )
                intent = multi_intent_result.primary_intent
                intent_conf = multi_intent_result.primary_confidence
                intent_meta = {
                    "multi_intent": [i.intent for i in multi_intent_result.secondary_intents[:3]],
                    "classifier_tier": multi_intent_result.classification_source,
                    "source": "enhanced",
                }
                for ent in multi_intent_result.extracted_entities:
                    if ent["type"] == "tax_code" and not context.active_tax_code:
                        context.active_tax_code = ent["value"]
                    elif ent["type"] == "tax_period" and not context.active_tax_period:
                        context.active_tax_period = ent["value"]
            except Exception as exc:
                logger.warning("[Orchestrator:stream] Enhanced intent failed: %s", exc)
                multi_intent_result = None

        if multi_intent_result is None:
            intent, intent_conf, intent_meta = self._intent_model.predict(message)
            if intent_conf < 0.45:
                rule_intent, rule_conf = self._rule_based_intent(message)
                if rule_conf > intent_conf:
                    intent, intent_conf = rule_intent, rule_conf

        intent_conf = min(0.95, max(0.15, float(intent_conf)))
        latency_breakdown["intent"] = (time.perf_counter() - t0) * 1000.0

        # ─── Intent Override: "phân tích chi tiết MST X" from row-click ───
        _detail_keywords = ("phân tích chi tiết", "phân tích rủi ro", "chi tiết doanh nghiệp", "kiểm tra mst", "phân tích mst", "phân tích chi tiết mst")
        _msg_lower = message.lower()
        _has_detail_keyword = any(kw in _msg_lower for kw in _detail_keywords)
        _has_explicit_mst = bool(re.search(r"\b\d{10}(?:-\d{3})?\b", message))
        _explicit_mst_match = re.search(r"\b(\d{10}(?:-\d{3})?)\b", message)
        if _explicit_mst_match:
            context.active_tax_code = _explicit_mst_match.group(1)
        # Override when intent leaked from previous turn (batch/top_n) but user is asking about a specific company
        if intent in ("top_n_query", "batch_analysis") and _has_explicit_mst and _has_detail_keyword:
            intent = "general_tax_query"
            intent_conf = 0.88
            logger.info("[Orchestrator] Intent override: %s -> general_tax_query (specific MST detail request)", intent)
        _stripped_message = message.strip()
        if intent == "batch_analysis" and re.fullmatch(r"\d{10}(?:-\d{3})?", _stripped_message):
            intent = "general_tax_query"
            intent_conf = 0.85
            logger.info("[Orchestrator] Intent override: batch_analysis -> general_tax_query (raw MST message)")

        yield {"event": "thinking", "data": {
            "step": "intent_done",
            "detail": f"Intent: {intent} ({intent_conf:.0%})",
            "intent": intent,
            "confidence": round(intent_conf, 4),
        }}

        # ─── Step 2.5: NL Query Fast Paths ──────────────────────────────
        t0 = time.perf_counter()
        nl_results = {}
        if getattr(context, "prior_answer_facts", None):
            nl_results["_prior_answer_facts"] = {
                "status": "available",
                "facts": context.prior_answer_facts[:12],
                "source": "session_memory",
            }
        try:
            from ml_engine.tax_agent_nl_query import NLQueryExecutor
            nl_executor = NLQueryExecutor()

            if intent == "top_n_query":
                quantity = 10
                if multi_intent_result and multi_intent_result.extracted_entities:
                    for ent in multi_intent_result.extracted_entities:
                        if ent.get("type") == "quantity":
                            quantity = min(50, max(1, int(ent["value"])))

                # Session memory: if batch data exists from a recent upload, use that
                if context.last_batch_data and context.last_batch_data.get("companies"):
                    batch_companies = context.last_batch_data["companies"]
                    sorted_companies = sorted(batch_companies, key=lambda c: float(c.get("risk_score", 0)), reverse=True)
                    top_slice = sorted_companies[:quantity]
                    src_filename = context.last_batch_data.get("filename", "file đã upload")
                    yield {"event": "thinking", "data": {"step": "nl_query", "detail": f"Truy vấn top {quantity} DN rủi ro từ {src_filename}..."}}
                    nl_results["top_n_risky_companies"] = {
                        "companies": [
                            {"stt": i + 1, **c} for i, c in enumerate(top_slice)
                        ],
                        "total": len(batch_companies),
                        "query_n": quantity,
                        "source": f"session_memory:{src_filename}",
                        "status": "success",
                    }
                    logger.info("[Orchestrator] top_n served from session memory (%d companies from %s)", len(top_slice), src_filename)
                else:
                    yield {"event": "thinking", "data": {"step": "nl_query", "detail": "Đang truy vấn top DN rủi ro..."}}
                    nl_results["top_n_risky_companies"] = nl_executor.execute_top_n(
                        db, n=quantity, sort_by="risk_score", mode=model_mode,
                    )

            elif intent == "company_name_lookup":
                company_name = ""
                if multi_intent_result and multi_intent_result.extracted_entities:
                    for ent in multi_intent_result.extracted_entities:
                        if ent.get("type") == "company_name":
                            company_name = ent["value"]
                if company_name:
                    yield {"event": "thinking", "data": {"step": "nl_query", "detail": f"Đang tìm DN: {company_name}..."}}
                    nl_results["company_name_search"] = nl_executor.execute_company_name_search(db, name=company_name)
                    matches = nl_results["company_name_search"].get("matches", [])
                    if len(matches) == 1:
                        context.active_tax_code = matches[0]["tax_code"]

            if attachment_analysis:
                analysis_type = str(attachment_analysis.get("analysis_type") or attachment_analysis.get("detected_schema", "attachment"))
                yield {"event": "thinking", "data": {"step": "attachment", "detail": f"Đã phân tích tệp đính kèm: {analysis_type}"}}
                nl_results["_attachment_analysis"] = attachment_analysis
                if attachment_analysis.get("status") == "mode_mismatch":
                    intent = "mode_mismatch"
                elif analysis_type == "risk_csv":
                    intent = "batch_analysis"
                    nl_results["_batch_results"] = attachment_analysis
                elif analysis_type == "vat_graph_csv":
                    intent = "vat_network_analysis"
                    nl_results["_vat_graph_batch_results"] = attachment_analysis
                elif analysis_type == "ocr_invoice":
                    intent = "invoice_risk"
                    nl_results["_ocr_document_results"] = attachment_analysis

            if csv_attachment:
                filename = csv_attachment.get("filename", "CSV")
                csv_columns = _detect_csv_columns(csv_attachment.get("content", b""))
                if _is_vat_invoice_schema(csv_columns):
                    intent = "vat_network_analysis"
                    yield {"event": "thinking", "data": {"step": "batch", "detail": f"Phát hiện file hóa đơn VAT: {filename}..."}}
                    vat_result = nl_executor.execute_vat_graph_inline(
                        db, csv_content=csv_attachment["content"], filename=filename,
                    )
                    nl_results["_vat_graph_batch_results"] = vat_result
                else:
                    intent = "batch_analysis"
                    yield {"event": "thinking", "data": {"step": "batch", "detail": f"Đang phân tích file {filename}..."}}
                    batch_result = nl_executor.execute_batch_inline(
                        db, csv_content=csv_attachment["content"], filename=filename,
                    )
                    nl_results["_batch_results"] = batch_result

                    # Session memory: save batch data for follow-up queries
                    if batch_result and batch_result.get("status") in ("success", "partial", "analyzed"):
                        # NLQueryExecutor returns "assessments"/"top_5", normalize to session memory keys
                        _batch_companies = batch_result.get("assessments", batch_result.get("companies", []))
                        _batch_top = batch_result.get("top_5", batch_result.get("top_risky", []))
                        context.last_batch_data = {
                            "filename": filename,
                            "total": batch_result.get("total", 0),
                            "companies": _batch_companies,
                            "company_index": {
                                _normalize_mst(c.get("tax_code") or c.get("mst")): c
                                for c in _batch_companies
                                if isinstance(c, dict) and _normalize_mst(c.get("tax_code") or c.get("mst"))
                            },
                            "by_level": batch_result.get("by_level", {}),
                            "top_risky": _batch_top,
                            "timestamp": time.time(),
                        }
                        context.last_attachment_summary = (
                            f"File {filename}: "
                            f"{batch_result.get('total', 0)} doanh nghiệp đã phân tích."
                        )
                        logger.info("[Orchestrator] Session memory updated with batch data from %s (%d companies)",
                                    filename, len(_batch_companies))
                        # Persist to memory store for cross-turn access
                        self._memory.save_batch_data(session_id, context.last_batch_data)
                        self._memory.save_attachment_summary(session_id, context.last_attachment_summary)

        except Exception as exc:
            logger.warning("[Orchestrator:stream] NL query error: %s", exc)
        latency_breakdown["nl_query"] = (time.perf_counter() - t0) * 1000.0

        try:
            self._persist_upload_session_memory(
                session_id=session_id,
                context=context,
                csv_attachment=csv_attachment,
                nl_results=nl_results,
            )
            self._inject_session_followup_rows(
                session_id=session_id,
                message=message,
                context=context,
                nl_results=nl_results,
            )
        except Exception as exc:
            logger.warning("[Orchestrator:stream] Session memory drill-down skipped: %s", exc)

        # ─── Batch analysis: clear active_tax_code to prevent GNN/motif on random MST ───
        if intent == "batch_analysis":
            context.active_tax_code = None

        routing_decision = self._task_router.route(
            query=message,
            intent=intent,
            model_mode=model_mode,
            has_attachment=bool(attachment_analysis or csv_attachment),
            attachment_analysis=attachment_analysis,
        )
        if not routing_decision.allow_legal and intent == "top_n_query":
            context.active_tax_code = None

        if routing_decision.mode_mismatch:
            answer = self._build_mode_mismatch_answer(
                model_mode=model_mode,
                routing_decision=routing_decision,
                attachment_analysis=attachment_analysis,
            )
            total_latency = (time.perf_counter() - t_total_start) * 1000.0
            payload = {
                "schema_version": AGENT_RESPONSE_SCHEMA_VERSION,
                "mode_contract_version": MODE_CONTRACT_VERSION,
                "run_id": run_id,
                "run_state": "finalized",
                "session_id": session_id,
                "intent": intent,
                "intent_confidence": round(routing_decision.route_confidence, 4),
                "complexity": "direct",
                "reasoning_trace": "Mode guard stopped incompatible request before tool execution.",
                "tools_used": [],
                "answer": answer,
                "summary": "Che do phan tich khong phu hop voi yeu cau hoac tep dinh kem.",
                "citations": [],
                "recommendations": ["Chuyen sang che do duoc goi y hoac dung Toan dien de he thong tu chon model."],
                "confidence": routing_decision.route_confidence,
                "abstained": False,
                "escalation_required": False,
                "escalation_domain": "none",
                "compliance_warnings": [],
                "active_tax_code": context.active_tax_code,
                "active_tax_period": context.active_tax_period,
                "latency_ms": round(total_latency, 1),
                "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
                "synthesis_tier": "mode_guard",
                "verification": {"status": "not_required", "reason": "mode_mismatch"},
                "clarification_needed": False,
                "clarification_questions": [],
                "tool_results": nl_results,
                "policy_traces": [],
                "visualization_data": {},
                "model_mode": model_mode,
                "plan_steps": [],
                "dialogue_act": getattr(conv_intel_result, "dialogue_act", "task") if conv_intel_result else "task",
                "answer_contract": routing_decision.answer_contract.value,
                "routing_decision": routing_decision.to_dict(),
                "focus_score": routing_decision.focus_score,
                "route_violation": routing_decision.route_violation,
                "selected_model_bundle": routing_decision.selected_model_bundle,
                "mode_validation": routing_decision.mode_validation,
                "mode_mismatch": True,
                "suggested_mode": routing_decision.suggested_mode,
                "suppressed_domains": routing_decision.suppressed_domains,
                "analysis_blocks": [
                    {"type": "mode_mismatch", "title": "Che do khong phu hop", "data": routing_decision.to_dict()}
                ],
                "mode_workspace": self._build_mode_workspace(
                    model_mode=model_mode,
                    legal_workspace={},
                    simulation_workspace={},
                    tool_results=nl_results,
                    viz_data={},
                    analysis_blocks=[
                        {"type": "mode_mismatch", "title": "Che do khong phu hop", "data": routing_decision.to_dict()}
                    ],
                ),
                "error_detail": {},
            }
            try:
                turn_repo.update_assistant_turn(
                    turn_id=assistant_turn_id,
                    message_text=answer,
                    normalized_intent=intent,
                    confidence=routing_decision.route_confidence,
                    citations_json="[]",
                )
                db.execute(sql_text("""
                    INSERT INTO agent_decision_traces
                    (session_id, turn_id, intent, selected_track, confidence,
                     abstained, escalation_required, evidence_json, answer_text)
                    VALUES (:session_id, :turn_id, :intent, 'mode_guard', :confidence,
                     FALSE, FALSE, CAST(:evidence_json AS jsonb), :answer_text)
                """), {
                    "session_id": session_id,
                    "turn_id": assistant_turn_id,
                    "intent": intent,
                    "confidence": routing_decision.route_confidence,
                    "evidence_json": _json_dumps_safe({
                        "routing": routing_decision.to_dict(),
                        "attachment_analysis": attachment_analysis or {},
                    }),
                    "answer_text": answer,
                })
                self._persist_route_event(
                    db,
                    session_id=session_id,
                    turn_id=assistant_turn_id,
                    dialogue_act=payload["dialogue_act"],
                    intent=intent,
                    model_mode=model_mode,
                    selected_tools=[],
                    routing_decision=routing_decision,
                )
                db.commit()
            except Exception as exc:
                try:
                    db.rollback()
                except Exception:
                    pass
                logger.debug("[Orchestrator:stream] Mode mismatch persist skipped: %s", exc)
            yield {"event": "text_chunk", "data": {"chunk": answer}}
            yield {"event": "done", "data": payload}
            return

        # ─── Step 2.8: Agentic LLM Advisory ──────────────────────────────
        agentic_decision = None
        if self._agentic_llm and self._agentic_llm.is_available:
            try:
                yield {"event": "thinking", "data": {"step": "agentic_llm", "detail": "🧠 Agent AI đang suy luận..."}}
                t_agent = time.perf_counter()
                agentic_decision = self._agentic_llm.infer(message)
                latency_breakdown["agentic_llm"] = (time.perf_counter() - t_agent) * 1000.0
                if agentic_decision:
                    logger.info(
                        "[Orchestrator] AgenticLLM chọn tool='%s' (intent_mapped='%s') — thought: %s",
                        agentic_decision.tool_name,
                        agentic_decision.mapped_intent,
                        agentic_decision.thought[:80],
                    )
                    yield {"event": "agent_thinking", "data": {
                        "thought": agentic_decision.thought,
                        "tool": agentic_decision.tool_name,
                        "confidence": agentic_decision.confidence,
                    }}
                    # Override intent nếu LLM V4 cho kết quả hợp lệ
                    intent = agentic_decision.mapped_intent
                    intent_conf = max(intent_conf, 0.92)
                    # Re-route với intent mới từ LLM
                    routing_decision = self._task_router.route(
                        query=message,
                        intent=intent,
                        model_mode=model_mode,
                        has_attachment=bool(attachment_analysis or csv_attachment),
                        attachment_analysis=attachment_analysis,
                    )
            except Exception as exc:
                logger.warning("[Orchestrator] AgenticLLM inference skipped: %s", exc)

        # ─── Step 3: Planning ───────────────────────────────────────────
        yield {"event": "thinking", "data": {"step": "planning", "detail": "Đang lập kế hoạch phân tích..."}}
        t0 = time.perf_counter()

        mode_profile = MODE_TOOL_PROFILES.get(model_mode, MODE_TOOL_PROFILES["full"])
        allowed_tools = mode_profile.get("required_tools")
        allowed_optional = set(mode_profile.get("optional_tools", []))
        allowed_sub_agents = set(mode_profile.get("sub_agents", ["legal", "analytics", "investigation"]))
        if routing_decision.allowed_tools is not None:
            route_tools = set(routing_decision.allowed_tools)
            allowed_tools = (
                route_tools
                if allowed_tools is None
                else set(allowed_tools).intersection(route_tools)
            )
            allowed_optional = allowed_optional.intersection(route_tools)
        if not routing_decision.allow_legal:
            allowed_sub_agents.discard("legal")
        plan_intent = "macro_forecast" if routing_decision.requested_domain == "macro" else intent
        plan = self._planner.plan(
            query=message,
            intent=plan_intent,
            intent_confidence=intent_conf,
            tax_code=context.active_tax_code,
            tax_period=context.active_tax_period,
            context_intents=context.active_intent_history,
            model_mode=model_mode,
            routing_decision=routing_decision,
            allowed_tools=routing_decision.allowed_tools,
        )
        if routing_decision.allowed_tools is not None:
            plan.steps = [s for s in plan.steps if s.tool_name in routing_decision.allowed_tools]
        elif not routing_decision.allow_legal:
            plan.steps = [s for s in plan.steps if s.tool_name != "knowledge_search"]

        # ─── Override bằng Agentic LLM ──────────────────────────────────
        if agentic_decision:
            from ml_engine.tax_agent_planner import SubTask, PlanStep
            # Chỉ override nếu tool LLM V4 chọn nằm trong danh sách cho phép của mode
            if allowed_tools is None or agentic_decision.tool_name in allowed_tools or agentic_decision.tool_name in allowed_optional:
                plan.steps = [SubTask(
                    step_id=1,
                    step_type=PlanStep.SYNTHESIZE,
                    tool_name=agentic_decision.tool_name,
                    tool_inputs=agentic_decision.tool_args,
                    description=f"Agentic LLM V4 Autonomous Call: {agentic_decision.tool_name}",
                    optional=False,
                )]
                plan.reasoning = agentic_decision.thought
                logger.info("[Orchestrator] Override plan.steps thành công với AgenticLLM V4")
            else:
                logger.warning("[Orchestrator] AgenticLLM chọn tool '%s' bị chặn bởi domain routing!", agentic_decision.tool_name)

        budget_ms = int(getattr(plan, "budget_ms", 30000) or 30000)
        latency_breakdown["planning"] = (time.perf_counter() - t0) * 1000.0

        yield {"event": "thinking", "data": {
            "step": "plan_done",
            "detail": f"Kế hoạch: {plan.complexity.value} — {len(plan.steps)} tools",
            "tools": [s.tool_name for s in plan.steps],
            "budget_ms": budget_ms,
        }}

        # ─── Step 4: Tool Execution ─────────────────────────────────────
        t0 = time.perf_counter()
        from ml_engine.tax_agent_tools import ToolCallRequest

        stages = plan.get_stages()
        all_tool_results: dict[str, dict[str, Any]] = {}

        if nl_results:
            for k, v in nl_results.items():
                all_tool_results[k] = v if isinstance(v, dict) else {"data": v, "status": "success"}

        for stage in stages:
            requests = []
            for step in stage:
                existing = all_tool_results.get(step.tool_name)
                if existing and existing.get("status") in {"success", "partial", "found", "analyzed"}:
                    continue
                if step.optional and intent_conf <= 0.6:
                    continue
                if allowed_tools is not None and step.tool_name not in allowed_tools and step.tool_name not in allowed_optional:
                    continue
                tool_inputs = dict(step.tool_inputs)
                request_id = f"req-{uuid.uuid4().hex[:8]}"
                if step.tool_name == "macro_forecast":
                    macro_scenario = self._normalize_simulation_params(simulation_params)
                    tool_inputs["scenario"] = macro_scenario
                    tool_inputs["action"] = self._infer_macro_action(message, macro_scenario)
                if step.tool_name == "knowledge_search":
                    tool_inputs.update({
                        "session_id": session_id,
                        "request_id": request_id,
                        "entity_scope": {
                            "tax_code": context.active_tax_code,
                            "tax_period": context.active_tax_period,
                        },
                        "top_k": top_k,
                    })
                requests.append(ToolCallRequest(
                    tool_name=step.tool_name,
                    inputs=tool_inputs,
                    request_id=request_id,
                    timeout_override=getattr(step, "timeout_ms", 10000) / 1000.0,
                    max_retries_override=getattr(step, "max_retries", 1),
                ))

            # Emit tool_start events
            for req in requests:
                desc = next((s.description for s in plan.steps if s.tool_name == req.tool_name), "")
                yield {"event": "tool_start", "data": {"tool": req.tool_name, "description": desc}}

            results = self._tool_executor.execute_parallel(requests, db=db)
            for result in results:
                all_tool_results[result.tool_name] = {
                    "status": result.status.value,
                    **(result.outputs or {}),
                    "_latency_ms": result.latency_ms,
                    "_error": result.error,
                }
                yield {"event": "tool_done", "data": {
                    "tool": result.tool_name,
                    "status": result.status.value,
                    "latency_ms": round(result.latency_ms or 0, 1),
                }}
            if (time.perf_counter() - t_total_start) * 1000.0 > budget_ms:
                yield {"event": "thinking", "data": {
                    "step": "budget",
                    "detail": "Execution budget reached; moving to synthesis with collected evidence.",
                    "budget_ms": budget_ms,
                }}
                break

        latency_breakdown["tools"] = (time.perf_counter() - t0) * 1000.0

        # Step 4.2: ReAct self-correction before sub-agents and synthesis.
        t0 = time.perf_counter()
        react_reflections = []
        react_escalate = False
        try:
            from ml_engine.tax_agent_react import ReActEngine
            react = ReActEngine()
            planned_tool_names = [s.tool_name for s in plan.steps]
            evidence_contracts = {
                s.tool_name: getattr(s, "evidence_contract", {}) or {}
                for s in plan.steps
            }
            max_react_iterations = int(getattr(plan, "max_react_iterations", 1) or 1)

            for iteration in range(max_react_iterations):
                if (time.perf_counter() - t_total_start) * 1000.0 > budget_ms:
                    yield {"event": "thinking", "data": {
                        "step": "budget",
                        "detail": "Skipping further ReAct retries because the plan budget is exhausted.",
                        "budget_ms": budget_ms,
                    }}
                    break
                reflection = react.reflect(
                    tool_results=all_tool_results,
                    planned_tools=planned_tool_names,
                    intent=intent,
                    iteration=iteration,
                    sub_agent_analysis=None,
                    evidence_contracts=evidence_contracts,
                )
                reflection_dict = reflection.to_dict()
                react_reflections.append(reflection_dict)
                yield {"event": "thinking", "data": {
                    "step": "react",
                    "detail": reflection.summary,
                    "iteration": iteration + 1,
                    "should_retry": reflection.should_retry,
                }}

                if any(a.get("action") == "trigger_investigation" for a in reflection_dict.get("actions", [])):
                    react_escalate = True

                if not reflection.should_retry:
                    break

                react_requests = self._build_react_tool_requests(
                    reflection.actions,
                    plan=plan,
                    context=context,
                    session_id=session_id,
                    top_k=top_k,
                )
                if routing_decision.allowed_tools is not None:
                    react_requests = [
                        req for req in react_requests
                        if req.tool_name in routing_decision.allowed_tools
                    ]
                elif not routing_decision.allow_legal:
                    react_requests = [
                        req for req in react_requests
                        if req.tool_name != "knowledge_search"
                    ]
                if not react_requests:
                    break

                for req in react_requests:
                    yield {"event": "tool_start", "data": {
                        "tool": req.tool_name,
                        "description": "ReAct retry/additional evidence",
                        "react_iteration": iteration + 1,
                    }}

                react_results = self._tool_executor.execute_parallel(react_requests, db=db)
                for result in react_results:
                    all_tool_results[result.tool_name] = {
                        "status": result.status.value,
                        **(result.outputs or {}),
                        "_latency_ms": result.latency_ms,
                        "_error": result.error,
                        "_react_iteration": iteration + 1,
                    }
                    if result.tool_name not in planned_tool_names:
                        planned_tool_names.append(result.tool_name)
                    yield {"event": "tool_done", "data": {
                        "tool": result.tool_name,
                        "status": result.status.value,
                        "latency_ms": round(result.latency_ms or 0, 1),
                        "react_iteration": iteration + 1,
                    }}
        except Exception as exc:
            logger.warning("[Orchestrator:stream] ReAct loop error: %s", exc)
        latency_breakdown["react"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 4.5: Sub-Agent Dispatch ───────────────────────────────
        t0 = time.perf_counter()
        sub_agent_analysis = {}

        try:
            ks_result = all_tool_results.get("knowledge_search", {})
            ks_hits = ks_result.get("hits", [])
            ks_graph_context = ks_result.get("graph_context")  # GraphRAG subgraph

            bypass_sub_agents = intent in ("batch_analysis", "top_n_query", "analytical_query")

            if ks_hits and "legal" in allowed_sub_agents and not bypass_sub_agents:
                graphrag_tag = " (GraphRAG)" if ks_graph_context else ""
                yield {"event": "sub_agent", "data": {"agent": "legal", "status": "running", "detail": f"Phân tích pháp lý{graphrag_tag}..."}}
                legal_opinion = self._legal_agent.research(
                    query=message, retrieval_results=ks_hits, intent=intent,
                    tax_code=context.active_tax_code,
                    graph_context=ks_graph_context,
                )
                sub_agent_analysis["legal_research"] = {
                    "analysis": legal_opinion.analysis, "conclusion": legal_opinion.conclusion,
                    "citation_chain": legal_opinion.citation_chain[:5], "authority_score": legal_opinion.authority_score,
                    "confidence": legal_opinion.confidence, "caveats": legal_opinion.caveats,
                    "applicable_laws": legal_opinion.applicable_laws[:5],
                    "graph_enhanced": bool(ks_graph_context),
                }
                yield {"event": "sub_agent", "data": {"agent": "legal", "status": "done"}}

            if context.active_tax_code and plan.complexity.value in ("moderate", "complex", "investigation") and "analytics" in allowed_sub_agents and not bypass_sub_agents:
                yield {"event": "sub_agent", "data": {"agent": "analytics", "status": "running", "detail": "Phân tích rủi ro tổng hợp..."}}
                analytics_report = self._analytics_agent.analyze(
                    tax_code=context.active_tax_code, tool_results=all_tool_results, intent=intent,
                )
                sub_agent_analysis["analytics"] = {
                    "composite_risk_score": analytics_report.composite_risk_score,
                    "risk_level": analytics_report.risk_level.value, "summary": analytics_report.summary,
                    "detailed_analysis": analytics_report.detailed_analysis,
                    "recommendations": analytics_report.recommendations, "risk_trend": analytics_report.risk_trend,
                    "confidence": analytics_report.confidence,
                }
                yield {"event": "sub_agent", "data": {"agent": "analytics", "status": "done"}}

            # G1: Expanded Investigation Agent trigger — enables 3-agent debate
            # for more queries, not just the original 3 intents.
            _investigation_intents = {
                "osint_ownership", "invoice_risk", "vat_refund_risk",
                "general_tax_query", "transfer_pricing", "vat_network_analysis",
            }
            _should_investigate = (
                not bypass_sub_agents
                and context.active_tax_code
                and plan.complexity.value in ("moderate", "complex", "investigation")
                and "investigation" in allowed_sub_agents
                and (
                    intent in _investigation_intents
                    or getattr(routing_decision, "requested_domain", "") in ("fraud", "vat")
                    or react_escalate
                )
            )
            if _should_investigate:
                yield {"event": "sub_agent", "data": {"agent": "investigation", "status": "running", "detail": "Điều tra chuyên sâu..."}}
                inv_report = self._investigation_agent.investigate(
                    tax_code=context.active_tax_code, tool_results=all_tool_results, intent=intent,
                )
                sub_agent_analysis["investigation"] = {
                    "suspicion_level": inv_report.suspicion_level.value, "overall_score": inv_report.overall_score,
                    "executive_summary": inv_report.executive_summary, "detailed_findings": inv_report.detailed_findings,
                    "patterns_count": len(inv_report.suspicious_patterns), "escalation_level": inv_report.escalation_level,
                    "recommended_actions": inv_report.recommended_actions, "confidence": inv_report.confidence,
                }
                yield {"event": "sub_agent", "data": {"agent": "investigation", "status": "done"}}

        except Exception as exc:
            logger.warning("[Orchestrator:stream] Sub-agent error: %s", exc)

        latency_breakdown["sub_agents"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 4.5: Multi-Agent Debate ────────────────────────────────
        debate_result_dict = None
        if len(sub_agent_analysis) >= 2:
            try:
                yield {"event": "thinking", "data": {"step": "debate", "detail": "Hội đồng agent đang tranh luận đa chiều..."}}
                from ml_engine.tax_agent_debate import AgentDebateProtocol
                debate = AgentDebateProtocol()
                debate_result = debate.run_debate(sub_agent_analysis, all_tool_results)
                debate_result_dict = debate_result.to_dict()
                yield {"event": "debate", "data": debate_result_dict}
            except Exception as exc:
                logger.warning("[Orchestrator:stream] Debate error: %s", exc)

        legal_review = self._legal_contradiction_review(all_tool_results)
        if legal_review.get("disagreements"):
            debate_result_dict = self._merge_legal_review_into_debate(
                debate_result_dict,
                legal_review,
            )
            yield {"event": "debate", "data": debate_result_dict}

        # ─── Step 5: Synthesis ──────────────────────────────────────────
        yield {"event": "thinking", "data": {"step": "synthesis", "detail": "Đang tổng hợp câu trả lời..."}}
        t0 = time.perf_counter()

        enriched_tool_results = dict(all_tool_results)
        for agent_name, analysis in sub_agent_analysis.items():
            enriched_tool_results[f"_sub_agent_{agent_name}"] = analysis

        synthesis_result = self._synthesizer.synthesize(
            query=message, intent=intent, tool_results=enriched_tool_results,
            reasoning_trace=plan.reasoning, tax_code=context.active_tax_code,
            answer_contract=routing_decision.answer_contract.value,
        )
        answer = self._synthesizer.format_response_text(synthesis_result)
        answer = self._enrich_with_sub_agents(answer, sub_agent_analysis)
        latency_breakdown["synthesis"] = (time.perf_counter() - t0) * 1000.0

        debate_escalate = self._debate_requires_escalation(debate_result_dict)
        if debate_result_dict:
            synthesis_result.confidence = self._confidence_after_debate(
                synthesis_result.confidence,
                debate_result_dict,
            )
            if debate_escalate:
                synthesis_result.escalation_needed = True

        # ─── Step 5.5: ReAct Self-Reflection ─────────────────────────────
        # ─── Step 6: Compliance ─────────────────────────────────────────
        t0 = time.perf_counter()
        retrieval_hits = len(all_tool_results.get("knowledge_search", {}).get("hits", []))
        compliance = self._compliance_gate.evaluate(
            query=message, intent=intent, intent_confidence=intent_conf,
            retrieval_hits=retrieval_hits, response_text=answer, tool_results=all_tool_results,
            session_id=session_id, turn_id=assistant_turn_id,
        )

        if compliance.abstain:
            synthesis_result = self._synthesizer.synthesize(
                query=message, intent=intent, tool_results=all_tool_results,
                reasoning_trace=plan.reasoning, abstained=True,
            )
            answer = self._synthesizer.format_response_text(synthesis_result)

        final_escalate = bool(
            compliance.escalate
            or debate_escalate
            or react_escalate
            or synthesis_result.escalation_needed
        )
        final_escalation_domain = compliance.escalation_domain or "none"
        if debate_escalate:
            final_escalation_domain = "adjudication"
        elif react_escalate and final_escalation_domain == "none":
            final_escalation_domain = "investigation"

        compliance_warnings = list(compliance.warnings)
        if debate_escalate:
            compliance_warnings.append("Multi-agent debate confidence is low or has major disagreement.")
        if react_escalate:
            compliance_warnings.append("ReAct detected contradictions that require deeper investigation.")

        routing_decision = self._task_router.evaluate_focus(
            decision=routing_decision,
            selected_tools=list(all_tool_results.keys()),
            answer_text=answer,
        )
        latency_breakdown["compliance"] = (time.perf_counter() - t0) * 1000.0

        # Stream only the final post-compliance answer.
        chunk_size = 80
        for i in range(0, len(answer), chunk_size):
            yield {"event": "text_chunk", "data": {"chunk": answer[i:i + chunk_size]}}

        # ─── Step 7: Audit ──────────────────────────────────────────────
        t0 = time.perf_counter()
        citations = []
        for ev in synthesis_result.evidence[:5]:
            if ev.source_type == "legal":
                citations.append({
                    "chunk_key": ev.metadata.get("chunk_key", ""),
                    "title": ev.title, "score": round(ev.score, 4), "citation_key": ev.citation_key,
                    "citation_spans": ev.metadata.get("citation_spans", []),
                    "authority_path": ev.metadata.get("authority_path", []),
                    "effective_status": ev.metadata.get("effective_status", {}),
                    "official_letter_scope": ev.metadata.get("official_letter_scope", {}),
                    "text": ev.content,
                    "full_text": ev.metadata.get("full_text", ""),
                })

        db.execute(sql_text("""
            UPDATE agent_turns SET message_text = :message_text, normalized_intent = :normalized_intent,
            confidence = :confidence, citations_json = CAST(:citations_json AS jsonb)
            WHERE id = :turn_id
        """), {
            "turn_id": assistant_turn_id, "message_text": answer,
            "normalized_intent": intent, "confidence": intent_conf,
            "citations_json": _json_dumps_safe(citations),
        })

        db.execute(sql_text("""
            INSERT INTO agent_decision_traces
            (session_id, turn_id, intent, selected_track, confidence,
             abstained, escalation_required, evidence_json, answer_text)
            VALUES (:session_id, :turn_id, :intent, :selected_track, :confidence,
             :abstained, :escalation_required, CAST(:evidence_json AS jsonb), :answer_text)
        """), {
            "session_id": session_id, "turn_id": assistant_turn_id, "intent": intent,
            "selected_track": plan.complexity.value, "confidence": synthesis_result.confidence,
            "abstained": compliance.abstain, "escalation_required": final_escalate,
            "evidence_json": _json_dumps_safe({
                "plan": {
                    "complexity": plan.complexity.value,
                    "tools": [s.tool_name for s in plan.steps],
                    "budget_ms": getattr(plan, "budget_ms", None),
                    "retry_policy": getattr(plan, "retry_policy", {}),
                    "evidence_contract": getattr(plan, "evidence_contract", {}),
                },
                "routing": routing_decision.to_dict(),
                "react": react_reflections,
                "debate": debate_result_dict,
                "legal_review": legal_review,
                "synthesis_verification": synthesis_result.verification,
                "compliance": {
                    "decision": compliance.overall_decision.value,
                    "warnings": compliance_warnings,
                    "final_escalation_domain": final_escalation_domain,
                },
            }),
            "answer_text": answer,
        })

        for tool_name, result in all_tool_results.items():
            db.execute(sql_text("""
                INSERT INTO agent_tool_calls (session_id, turn_id, tool_name, tool_input, tool_output, status, latency_ms)
                VALUES (:session_id, :turn_id, :tool_name, CAST(:tool_input AS jsonb), CAST(:tool_output AS jsonb), :status, :latency_ms)
            """), {
                "session_id": session_id, "turn_id": assistant_turn_id, "tool_name": tool_name,
                "tool_input": _json_dumps_safe(next((s.tool_inputs for s in plan.steps if s.tool_name == tool_name), {})),
                "tool_output": _json_dumps_safe({k: v for k, v in result.items() if not k.startswith("_")}),
                "status": result.get("status", "unknown"), "latency_ms": result.get("_latency_ms"),
            })

        self._persist_route_event(
            db,
            session_id=session_id,
            turn_id=assistant_turn_id,
            dialogue_act=getattr(conv_intel_result, "dialogue_act", "task") if conv_intel_result else "task",
            intent=intent,
            model_mode=model_mode,
            selected_tools=list(all_tool_results.keys()),
            routing_decision=routing_decision,
        )

        self._persist_execution_plan(
            db,
            plan=plan,
            session_id=session_id,
            turn_id=assistant_turn_id,
            query_text=message,
            intent=intent,
            tool_results=all_tool_results,
            synthesis_result=synthesis_result,
            compliance=compliance,
            latency_breakdown=latency_breakdown,
            final_escalate=final_escalate,
        )
        self._persist_debate_adjudication(
            db,
            session_id=session_id,
            turn_id=assistant_turn_id,
            tax_code=context.active_tax_code,
            debate_result=debate_result_dict,
            final_escalate=final_escalate,
        )
        self._persist_agent_workspace(
            db,
            session_id=session_id,
            turn_id=assistant_turn_id,
            query_text=message,
            intent=intent,
            tool_results=all_tool_results,
            synthesis_result=synthesis_result,
            react_reflections=react_reflections,
            debate_result=debate_result_dict,
            legal_review=legal_review,
            final_escalate=final_escalate,
            escalation_domain=final_escalation_domain,
        )
        self._persist_legal_claim_verifications(
            db,
            session_id=session_id,
            turn_id=assistant_turn_id,
            synthesis_result=synthesis_result,
        )

        self._memory.persist_entities(session_id, context.active_entities)
        db.commit()
        latency_breakdown["audit"] = (time.perf_counter() - t0) * 1000.0
        total_latency = (time.perf_counter() - t_total_start) * 1000.0

        # Visualization data
        viz_data = self._build_visualization_data(all_tool_results, sub_agent_analysis, plan, latency_breakdown)
        analysis_blocks = self._build_analysis_blocks(
            tool_results=all_tool_results,
            routing_decision=routing_decision,
            synthesis_result=synthesis_result,
        )
        prior_facts = self._extract_prior_answer_facts(
            mode=model_mode,
            intent=intent,
            answer=answer,
            summary=synthesis_result.summary,
            recommendations=synthesis_result.recommendations,
            analysis_blocks=analysis_blocks,
            tool_results=all_tool_results,
        )
        try:
            self._memory.save_prior_answer_facts(
                session_id,
                turn_id=assistant_turn_id,
                mode=model_mode,
                intent=intent,
                facts=prior_facts,
            )
        except Exception as exc:
            logger.debug("[Orchestrator:stream] Prior fact persist skipped: %s", exc)
        try:
            from ml_engine.visualization_normalizer_v3 import normalize_visualization_v3
            if isinstance(viz_data, dict):
                viz_data["v3"] = normalize_visualization_v3(viz_data)
        except Exception:
            pass
        if debate_result_dict:
            viz_data["agent_debate"] = debate_result_dict
        if legal_review.get("disagreements"):
            viz_data["legal_review"] = legal_review
        if viz_data:
            yield {"event": "viz_data", "data": viz_data}

        # ─── Final done event with full response ────────────────────────
        # ─── Telemetry Logging ──────────────────────────────────────────
        try:
            from ml_engine.tax_agent_telemetry import get_telemetry
            telemetry_resp = OrchestratorResponse(
                session_id=session_id,
                intent=intent,
                intent_confidence=intent_conf,
                complexity=plan.complexity.value,
                reasoning_trace=plan.reasoning,
                tools_used=list(all_tool_results.keys()),
                answer=answer,
                summary=synthesis_result.summary,
                citations=citations,
                recommendations=synthesis_result.recommendations,
                confidence=synthesis_result.confidence,
                abstained=compliance.abstain,
                escalation_required=final_escalate,
                escalation_domain=final_escalation_domain,
                compliance_warnings=compliance_warnings,
                active_tax_code=context.active_tax_code,
                active_tax_period=context.active_tax_period,
                latency_ms=total_latency,
                latency_breakdown=latency_breakdown,
                tool_results=all_tool_results,
                synthesis_tier=synthesis_result.synthesis_tier,
                verification=synthesis_result.verification,
                clarification_needed=synthesis_result.clarification_needed,
                clarification_questions=synthesis_result.clarification_questions,
                policy_traces=[],
                plan_steps=[{"tool": s.tool_name, "description": s.description, "optional": getattr(s, 'optional', False)} for s in plan.steps],
                visualization_data=viz_data,
                model_mode=model_mode,
                dialogue_act=getattr(conv_intel_result, "dialogue_act", "task") if conv_intel_result else "task",
                answer_contract=routing_decision.answer_contract.value,
                routing_decision=routing_decision.to_dict(),
                focus_score=routing_decision.focus_score,
                route_violation=routing_decision.route_violation,
                selected_model_bundle=routing_decision.selected_model_bundle,
                mode_validation=routing_decision.mode_validation,
                mode_mismatch=routing_decision.mode_mismatch,
                suggested_mode=routing_decision.suggested_mode,
                suppressed_domains=routing_decision.suppressed_domains,
                analysis_blocks=analysis_blocks,
            )
            get_telemetry().record_from_orchestrator(telemetry_resp)
        except Exception as e:
            logger.error(f"[Orchestrator:stream] Telemetry logging failed: {e}")

        retrieval_context = all_tool_results.get("knowledge_search", {}).get("hits", [])
        
        # Build Legal Workspace
        ks = all_tool_results.get("knowledge_search") or {}
        facts = [f"Tra cứu và phân tích {len(ks.get('hits', []) or [])} tài liệu."]
        assumptions = []
        if legal_review.get("disagreements"):
            assumptions.append("Cảnh báo: Có rủi ro xung đột hoặc văn bản hết hiệu lực.")
        
        open_questions = list(getattr(synthesis_result, "clarification_questions", []) or [])
        verification = getattr(synthesis_result, "verification", {}) or {}
        for claim in verification.get("unsupported_claims", [])[:5]:
            open_questions.append(f"Cần xác minh: {claim.get('claim', '')[:100]}")
            
        verifications = []
        for item in verification.get("verified_claims", []) or []:
            verifications.append({"claim": item.get("claim"), "is_verified": True})
        for item in verification.get("unsupported_claims", []) or []:
            verifications.append({"claim": item.get("claim"), "is_verified": False})
            
        escalations = []
        if final_escalate:
            escalations.append(f"Cần chuyên gia xem xét (Mức: {final_escalation_domain})")
            if legal_review.get("disagreements"):
                escalations.append("Xung đột pháp lý chưa giải quyết triệt để.")

        # G5+G6: Inject debate conclusions and adjudicator verdict
        debate_conclusions = []
        if debate_result_dict:
            consensus_pct = debate_result_dict.get("consensus_pct", 0)
            debate_conclusions.append(
                f"Đồng thuận agent: {consensus_pct}% — "
                f"{debate_result_dict.get('consensus_label', 'N/A')}"
            )
            for d in debate_result_dict.get("disagreements", [])[:3]:
                topic = d.get("topic", "")
                severity = d.get("severity", "")
                debate_conclusions.append(f"Bất đồng [{severity}]: {topic}")

            adj = debate_result_dict.get("adjudicator_verdict")
            if adj:
                verdict_vi = adj.get("verdict_vi", "")
                strongest = adj.get("strongest_agent_label", "")
                debate_conclusions.append(
                    f"⚖️ Trọng tài: {verdict_vi} (bằng chứng mạnh nhất: {strongest})"
                )
                escalations.append(adj.get("recommended_action", ""))

        legal_workspace = {
            "facts": facts,
            "assumptions": assumptions,
            "open_questions": open_questions,
            "verifications": verifications,
            "escalations": escalations,
            "debate_conclusions": debate_conclusions,
        }
        simulation_workspace = self._build_simulation_workspace(
            context=context,
            model_mode=model_mode,
            message=message,
            tool_results=all_tool_results,
            viz_data=viz_data,
            simulation_params=simulation_params,
        )
        mode_workspace = self._build_mode_workspace(
            model_mode=model_mode,
            legal_workspace=legal_workspace,
            simulation_workspace=simulation_workspace,
            tool_results=all_tool_results,
            viz_data=viz_data,
            analysis_blocks=analysis_blocks,
        )

        yield {"event": "done", "data": {
            "schema_version": AGENT_RESPONSE_SCHEMA_VERSION,
            "mode_contract_version": MODE_CONTRACT_VERSION,
            "run_id": run_id,
            "run_state": "finalized",
            "session_id": session_id,
            "intent": intent,
            "intent_confidence": round(intent_conf, 4),
            "complexity": plan.complexity.value,
            "reasoning_trace": plan.reasoning,
            "tools_used": list(all_tool_results.keys()),
            "answer": answer,
            "summary": synthesis_result.summary,
            "citations": citations,
            "recommendations": synthesis_result.recommendations,
            "confidence": synthesis_result.confidence,
            "abstained": compliance.abstain,
            "escalation_required": final_escalate,
            "escalation_domain": final_escalation_domain,
            "compliance_warnings": compliance_warnings,
            "active_tax_code": context.active_tax_code,
            "active_tax_period": context.active_tax_period,
            "latency_ms": round(total_latency, 1),
            "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
            "synthesis_tier": synthesis_result.synthesis_tier,
            "verification": synthesis_result.verification,
            "clarification_needed": synthesis_result.clarification_needed,
            "clarification_questions": synthesis_result.clarification_questions,
            "tool_results": all_tool_results,
            "policy_traces": [
                {
                    "rule_key": getattr(t, "rule_key", ""),
                    "decision": getattr(getattr(t, "decision", None), "value", str(getattr(t, "decision", ""))),
                    "score": getattr(t, "score", None),
                    "reason": getattr(t, "reason", None),
                    "details": getattr(t, "details", {}),
                }
                for t in getattr(compliance, "traces", [])
            ],
            "visualization_data": viz_data,
            "model_mode": model_mode,
            "dialogue_act": getattr(conv_intel_result, "dialogue_act", "task") if conv_intel_result else "task",
            "answer_contract": routing_decision.answer_contract.value,
            "routing_decision": routing_decision.to_dict(),
            "focus_score": routing_decision.focus_score,
            "route_violation": routing_decision.route_violation,
            "selected_model_bundle": routing_decision.selected_model_bundle,
            "mode_validation": routing_decision.mode_validation,
            "mode_mismatch": routing_decision.mode_mismatch,
            "suggested_mode": routing_decision.suggested_mode,
            "suppressed_domains": routing_decision.suppressed_domains,
            "analysis_blocks": analysis_blocks,
            "plan_budget_ms": budget_ms,
            "retry_policy": getattr(plan, "retry_policy", {}),
            "evidence_contract": getattr(plan, "evidence_contract", {}),
            "react_reflections": react_reflections,
            "debate": debate_result_dict,
            "legal_review": legal_review,
            "legal_workspace": legal_workspace,
            "simulation_workspace": simulation_workspace,
            "mode_workspace": mode_workspace,
            "error_detail": {},
            "intent_model_version": getattr(self._enhanced_intent, "tier", "") if self._enhanced_intent else "",
            "planner_policy_version": getattr(self._planner, "PLANNER_POLICY_VERSION", "") if self._planner else "",
            "debate_session_id": debate_result_dict.get("session_id") if debate_result_dict else None,
            "graph_reasoning_path": [],
            "plan_steps": [
                {
                    "tool": s.tool_name,
                    "description": s.description,
                    "timeout_ms": getattr(s, "timeout_ms", None),
                    "max_retries": getattr(s, "max_retries", None),
                    "evidence_contract": getattr(s, "evidence_contract", {}),
                }
                for s in plan.steps
            ],
        }}

    def _extract_prior_answer_facts(
        self,
        *,
        mode: str,
        intent: str,
        answer: str,
        summary: str,
        recommendations: list[str],
        analysis_blocks: list[dict[str, Any]],
        tool_results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract compact structured facts for follow-up grounding."""
        facts: list[dict[str, Any]] = []
        if summary:
            facts.append({
                "fact_type": "summary",
                "claim_text": summary[:700],
                "source_tool": "synthesis",
                "confidence": 0.82,
                "value_json": {"mode": mode, "intent": intent},
            })
        for rec in (recommendations or [])[:6]:
            text = str(rec or "").strip()
            if text:
                facts.append({
                    "fact_type": "recommendation",
                    "claim_text": text[:700],
                    "source_tool": "synthesis",
                    "confidence": 0.78,
                    "value_json": {"mode": mode, "intent": intent},
                })
        for block in (analysis_blocks or [])[:6]:
            if not isinstance(block, dict):
                continue
            metrics = block.get("metrics") if isinstance(block.get("metrics"), dict) else {}
            title = str(block.get("title") or block.get("type") or "analysis").strip()
            claim = str(block.get("summary") or title).strip()
            if claim:
                facts.append({
                    "fact_type": str(block.get("type") or "analysis_block"),
                    "claim_text": claim[:700],
                    "source_tool": "analysis_blocks",
                    "confidence": 0.8,
                    "value_json": {"title": title, "metrics": metrics},
                })
            for item in (metrics.get("top_records") or metrics.get("top_invoice_risks") or [])[:8]:
                if not isinstance(item, dict):
                    continue
                subject = item.get("tax_code") or item.get("mst") or item.get("seller_tax_code") or item.get("buyer_tax_code")
                score = item.get("risk_score") or item.get("edge_risk_score") or item.get("score")
                facts.append({
                    "fact_type": "ranked_subject",
                    "subject_key": str(subject) if subject else None,
                    "claim_text": f"{title}: {subject or 'subject'} co diem/chi so {score if score is not None else 'N/A'}",
                    "source_tool": "analysis_blocks",
                    "confidence": 0.76,
                    "value_json": item,
                })
        for key in ("_session_upload_row", "_vat_session_focus", "_prior_answer_facts"):
            value = tool_results.get(key) if isinstance(tool_results, dict) else None
            if isinstance(value, dict):
                facts.append({
                    "fact_type": key.strip("_"),
                    "subject_key": value.get("tax_code") or value.get("source_filename"),
                    "claim_text": f"Session memory available: {key}",
                    "source_tool": key,
                    "confidence": 0.74,
                    "value_json": value,
                })
        return facts[:80]

    def _build_mode_workspace(
        self,
        *,
        model_mode: str,
        legal_workspace: dict[str, Any],
        simulation_workspace: dict[str, Any],
        tool_results: dict[str, Any],
        viz_data: dict[str, Any],
        analysis_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a unified workspace payload across modes."""
        contract = AgentModeContractRegistry.get(model_mode)
        workspace: dict[str, Any] = {
            "schema_version": AGENT_RESPONSE_SCHEMA_VERSION,
            "mode_contract_version": MODE_CONTRACT_VERSION,
            "mode": contract.mode,
            "panel": contract.workspace_panel,
            "required_visualization_keys": list(contract.required_visualization_keys),
            "analysis_blocks": analysis_blocks[:8],
            "artifacts": [],
        }
        if contract.workspace_panel == "legal":
            workspace["legal"] = legal_workspace or {}
        elif contract.workspace_panel == "simulation":
            workspace["simulation"] = simulation_workspace or {}
        elif contract.workspace_panel == "fraud":
            fraud = (viz_data or {}).get("fraud") or {}
            batch = tool_results.get("_batch_results") or tool_results.get("_attachment_analysis") or {}
            workspace["fraud"] = {
                "summary": fraud.get("summary") or {},
                "top_companies": fraud.get("top_companies") or batch.get("top_5") or batch.get("assessments", [])[:10],
                "risk_distribution": fraud.get("risk_distribution") or batch.get("by_level") or {},
                "source": batch.get("filename") or fraud.get("summary", {}).get("source"),
            }
        elif contract.workspace_panel == "vat":
            vat = (viz_data or {}).get("vat") or {}
            workspace["vat"] = {
                "summary": vat.get("summary") or {},
                "top_invoice_risks": vat.get("top_invoice_risks") or [],
                "graph_counts": {
                    "nodes": len(((vat.get("graph") or {}).get("nodes") or [])),
                    "edges": len(((vat.get("graph") or {}).get("edges") or [])),
                },
            }
        elif contract.workspace_panel == "delinquency":
            dq = tool_results.get("delinquency_check") or {}
            deep = tool_results.get("temporal_delinquency_deep") or {}
            uplift = tool_results.get("causal_uplift_recommend") or {}
            workspace["delinquency"] = {
                "probabilities": {
                    "prob_30d": dq.get("prob_30d") or deep.get("prob_30d"),
                    "prob_60d": dq.get("prob_60d") or deep.get("prob_60d"),
                    "prob_90d": dq.get("prob_90d") or deep.get("prob_90d"),
                },
                "top_reasons": dq.get("top_reasons") or deep.get("top_reasons") or [],
                "recommended_action": uplift.get("recommended_action"),
            }
        return workspace

    def _normalize_simulation_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize macro workspace controls into app.routers.simulation.ScenarioInput shape."""
        defaults: dict[str, Any] = {
            "vat_rate": 10.0,
            "cit_rate": 20.0,
            "audit_coverage_pct": 5.0,
            "penalty_multiplier": 1.0,
            "interest_rate": 6.0,
            "economic_growth_pct": 6.5,
            "cpi_pct": 3.5,
            "unemployment_pct": 2.3,
            "exchange_rate_delta_pct": 0.0,
            "projection_years": 5,
        }
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        if not isinstance(params, dict):
            return dict(defaults)
        aliases = {
            "vat": "vat_rate",
            "cit": "cit_rate",
            "audit": "audit_coverage_pct",
            "penalty": "penalty_multiplier",
            "gdp": "economic_growth_pct",
            "cpi": "cpi_pct",
            "unemployment": "unemployment_pct",
            "usd_fx": "exchange_rate_delta_pct",
            "fx": "exchange_rate_delta_pct",
            "years": "projection_years",
            "horizon_years": "projection_years",
        }
        normalized = dict(defaults)
        for raw_key, raw_value in params.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key not in defaults:
                if key in {"industry_filter", "province_filter"} and raw_value:
                    normalized[key] = str(raw_value)
                continue
            try:
                if key == "projection_years":
                    normalized[key] = int(max(1, min(10, int(float(raw_value)))))
                else:
                    normalized[key] = float(raw_value)
            except Exception:
                continue
        return normalized

    def _infer_macro_action(self, message: str, scenario: dict[str, Any]) -> str:
        """Infer simulation action while keeping the UI control payload safe."""
        text = (message or "").lower()
        if "monte" in text and scenario.get("n_iterations"):
            return "monte-carlo"
        if ("nhay" in text or "sensitivity" in text) and scenario.get("request"):
            return "sensitivity"
        if ("so sanh" in text or "compare" in text) and scenario.get("scenarios"):
            return "compare"
        return "run"

    def _plain_payload(self, value: Any) -> Any:
        """Convert pydantic/models/nested containers to JSON-friendly dicts."""
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if isinstance(value, dict):
            return {str(k): self._plain_payload(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._plain_payload(v) for v in value]
        return value

    def _build_simulation_workspace(
        self,
        *,
        context,
        model_mode: str,
        message: str,
        tool_results: dict[str, Any],
        viz_data: dict[str, Any],
        simulation_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compact simulation workspace payload for macro-mode sidebar."""
        has_macro_result = isinstance(tool_results, dict) and "macro_forecast" in tool_results
        if model_mode != "macro" and not has_macro_result:
            return {}
        sim: dict[str, Any] = {}
        macro_tool: dict[str, Any] = {}
        for key in ("macro_forecast", "macro_scenario_simulation", "simulation", "macro_impact"):
            value = tool_results.get(key) if isinstance(tool_results, dict) else None
            if isinstance(value, dict):
                macro_tool = value
                candidate = value.get("result") or value.get("baseline") or value
                sim = self._plain_payload(candidate)
                if not isinstance(sim, dict):
                    sim = value
                break
        requested_params = self._normalize_simulation_params(simulation_params)
        params = sim.get("parameters") if isinstance(sim.get("parameters"), dict) else requested_params
        suggested = sim.get("recommended_parameters") if isinstance(sim.get("recommended_parameters"), dict) else {}
        ranges = sim.get("parameter_ranges") if isinstance(sim.get("parameter_ranges"), dict) else {}
        sensitivity = sim.get("sensitivity_top_factors") if isinstance(sim.get("sensitivity_top_factors"), list) else []
        quarterly = sim.get("quarterly_projection") if isinstance(sim.get("quarterly_projection"), list) else []
        industry_impacts = sim.get("industry_impacts") if isinstance(sim.get("industry_impacts"), list) else []
        return {
            "current_params": params,
            "recommended_params": suggested,
            "ranges": ranges,
            "sensitivity_top_factors": sensitivity[:12],
            "scenario_label": sim.get("scenario_label") or sim.get("scenario_name") or "Macro simulation",
            "projection_years": sim.get("projection_years") or sim.get("horizon_years") or 3,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_message": message[:120],
            "session_id": getattr(context, "session_id", None),
            "kpis": viz_data.get("macro_kpis") if isinstance(viz_data, dict) else {},
            "tool_action": macro_tool.get("action") or "run",
            "result": sim,
            "quarterly_projection": quarterly[:60],
            "industry_impacts": industry_impacts[:30],
        }


    def _persist_upload_session_memory(
        self,
        *,
        session_id: str,
        context,
        csv_attachment: dict | None,
        nl_results: dict[str, Any],
    ) -> None:
        """Save risk batch + VAT snapshot for follow-up turns (same browser session_id)."""
        if csv_attachment:
            # Session already updated inline after execute_batch_inline
            # Re-sync context from memory cache in case downstream reads context.last_batch_data
            cached = self._memory.get_batch_data(session_id)
            if cached:
                context.last_batch_data = cached
            return

        br = nl_results.get("_batch_results") if isinstance(nl_results, dict) else None
        if isinstance(br, dict) and br.get("analysis_type") == "risk_csv":
            if br.get("status") not in {"success", "done", "partial"}:
                return
            comps = list(br.get("assessments") or br.get("companies") or [])
            if not comps:
                return
            company_index = {
                _normalize_mst(c.get("tax_code") or c.get("mst")): c
                for c in comps
                if isinstance(c, dict) and _normalize_mst(c.get("tax_code") or c.get("mst"))
            }
            payload = {
                "filename": br.get("filename") or "upload",
                "total": int(br.get("total") or len(comps)),
                "companies": comps,
                "company_index": company_index,
                "by_level": br.get("by_level") or {},
                "top_risky": list(br.get("top_5") or br.get("top_risky") or comps[:5]),
                "batch_source": "canonical_ai_batch" if br.get("canonical_batch_id") else "attachment_analysis",
                "canonical_batch_id": br.get("canonical_batch_id"),
                "timestamp": time.time(),
            }
            context.last_batch_data = payload
            self._memory.save_batch_data(session_id, payload)
            self._memory.save_attachment_summary(
                session_id,
                f"File {payload['filename']}: {payload['total']} DN đã phân tích (phiên làm việc).",
            )

        vat = nl_results.get("_vat_graph_batch_results") if isinstance(nl_results, dict) else None
        if (
            isinstance(vat, dict)
            and vat.get("analysis_type") == "vat_graph_csv"
            and vat.get("status") != "error"
        ):
            graph = vat.get("graph") or {}
            snap = {
                "filename": vat.get("filename"),
                "batch_id": vat.get("batch_id"),
                "upload_id": vat.get("upload_id"),
                "summary": vat.get("summary") or {},
                "top_invoice_risks": list(graph.get("top_invoice_risks") or [])[:100],
                "nodes": list(graph.get("nodes") or [])[:150],
                "edges": list(graph.get("edges") or [])[:250],
                "timestamp": time.time(),
            }
            context.last_vat_snapshot = snap
            self._memory.save_vat_snapshot(session_id, snap)

    def _inject_session_followup_rows(
        self,
        *,
        session_id: str,
        message: str,
        context,
        nl_results: dict[str, Any],
    ) -> None:
        """Prefer rows from uploaded risk/VAT snapshots when user mentions MST in chat."""
        tc = _first_mst_from_message(message)
        if not tc:
            return

        bd = getattr(context, "last_batch_data", None) or self._memory.get_batch_data(session_id) or {}
        index = bd.get("company_index") if isinstance(bd, dict) else None
        matched_from_index = False
        if isinstance(index, dict):
            matched = index.get(tc)
            if isinstance(matched, dict):
                nl_results["_session_upload_row"] = {
                    "status": "matched",
                    "tax_code": tc,
                    "row": matched,
                    "source_filename": bd.get("filename"),
                    "canonical_batch_id": bd.get("canonical_batch_id"),
                }
                matched_from_index = True
        rows = [] if matched_from_index else (bd.get("companies") if isinstance(bd, dict) else None)
        if isinstance(rows, list) and rows:
            matched = None
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if _normalize_mst(row.get("tax_code") or row.get("mst")) == tc:
                    matched = row
                    break
            if matched is not None:
                nl_results["_session_upload_row"] = {
                    "status": "matched",
                    "tax_code": tc,
                    "row": matched,
                    "source_filename": bd.get("filename"),
                }

        vs = (
            getattr(context, "last_vat_snapshot", None)
            or self._memory.get_vat_snapshot(session_id)
        )
        if isinstance(vs, dict):
            inv = vs.get("top_invoice_risks") if isinstance(vs.get("top_invoice_risks"), list) else []
            matched_inv = [
                r for r in inv
                if isinstance(r, dict)
                and (
                    _normalize_mst(r.get("seller_tax_code") or r.get("source")) == tc
                    or _normalize_mst(r.get("buyer_tax_code") or r.get("target")) == tc
                )
            ]
            if matched_inv:
                nl_results["_vat_session_focus"] = {
                    "status": "matched",
                    "tax_code": tc,
                    "source_filename": vs.get("filename"),
                    "invoices": matched_inv[:40],
                    "batch_id": vs.get("batch_id"),
                }

    def _build_mode_mismatch_answer(
        self,
        *,
        model_mode: str,
        routing_decision,
        attachment_analysis: dict[str, Any] | None = None,
    ) -> str:
        labels = {
            "full": "Toàn diện",
            "legal": "Pháp lý",
            "fraud": "Gian lận",
            "vat": "VAT & Hóa đơn",
            "delinquency": "Dự báo nợ",
            "macro": "Vĩ mô",
        }
        selected = labels.get(model_mode, model_mode)
        suggested = routing_decision.suggested_mode or routing_decision.requested_domain
        suggested_label = labels.get(suggested, suggested or "Toàn diện")
        domain_label = labels.get(routing_decision.requested_domain, routing_decision.requested_domain)
        filename = ""
        if attachment_analysis:
            filename = str(attachment_analysis.get("filename") or "").strip()
        file_note = f" Tệp `{filename}` được nhận diện thuộc miền {domain_label}." if filename else ""
        return (
            f"Chế độ {selected} không phù hợp với yêu cầu hiện tại.{file_note}\n\n"
            f"Hệ thống đã dừng trước khi chạy model để tránh phân tích sai miền. "
            f"Nên chuyển sang chế độ {suggested_label}, hoặc chọn Toàn diện để Auto Orchestrator tự chọn model phù hợp.\n\n"
            f"Bundle được đề xuất: {', '.join(routing_decision.selected_model_bundle or [])}."
        )


    def _legal_contradiction_review(self, tool_results: dict[str, Any]) -> dict[str, Any]:
        """Turn GraphRAG legal quality signals into debate/adjudication inputs."""
        ks = tool_results.get("knowledge_search") or {}
        graph_context = ks.get("graph_context") or {}
        disagreements: list[dict[str, Any]] = []

        effective = graph_context.get("effective_status") or {}
        if effective.get("has_non_usable"):
            disagreements.append({
                "topic": "legal_effective_status",
                "severity": "major",
                "stance_a": "Use only current, applicable legal documents.",
                "stance_b": "Retrieved subgraph contains expired/repealed/pending documents.",
                "evidence": effective,
                "recommendation": "Retry GraphRAG with effective-date query and escalate if unresolved.",
            })

        scope = graph_context.get("official_letter_scope") or {}
        if scope.get("warnings"):
            disagreements.append({
                "topic": "official_letter_scope",
                "severity": "minor",
                "stance_a": "Official letters can support interpretation.",
                "stance_b": "Official letters should not be treated as general normative rules.",
                "evidence": scope,
                "recommendation": "State the scope limitation and prefer law/decree/circular when conflicting.",
            })

        authority_path = graph_context.get("authority_path") or []
        has_official_letter = any(
            str(item.get("entity_type", "")).lower() == "official_letter"
            for item in authority_path
        )
        has_higher_authority = any(
            str(item.get("entity_type", "")).lower() in {"law", "decree", "circular"}
            for item in authority_path
        )
        if has_official_letter and has_higher_authority:
            disagreements.append({
                "topic": "authority_priority",
                "severity": "minor",
                "stance_a": "Higher-authority documents define the governing rule.",
                "stance_b": "Official-letter evidence is present and may be case-specific.",
                "evidence": {"authority_path": authority_path[:6]},
                "recommendation": "Use official letter as interpretive support only.",
            })

        penalty = 0.0
        for item in disagreements:
            penalty += 0.18 if item.get("severity") == "major" else 0.07

        return {
            "status": "review" if disagreements else "clear",
            "disagreements": disagreements,
            "confidence_penalty": round(min(0.35, penalty), 4),
            "summary": (
                "Legal GraphRAG review found authority/effective-date concerns."
                if disagreements else
                "Legal GraphRAG review found no authority conflict."
            ),
        }

    def _merge_legal_review_into_debate(
        self,
        debate_result: dict[str, Any] | None,
        legal_review: dict[str, Any],
    ) -> dict[str, Any]:
        disagreements = list(legal_review.get("disagreements") or [])
        if not debate_result:
            consensus = round(max(0.45, 0.86 - float(legal_review.get("confidence_penalty", 0.0))), 4)
            return {
                "consensus_score": consensus,
                "consensus_label": "legal_review_required" if consensus < 0.7 else "legal_grounded",
                "consensus_stance": "Prefer current higher-authority law; limit official letters by scope.",
                "disagreements": disagreements,
                "recommendation": "Verify legal effect, authority priority, and citation spans before final action.",
                "summary": legal_review.get("summary"),
                "source": "legal_graphrag_review",
            }

        merged = dict(debate_result)
        merged_disagreements = list(merged.get("disagreements") or [])
        merged_disagreements.extend(disagreements)
        penalty = float(legal_review.get("confidence_penalty", 0.0) or 0.0)
        consensus = float(merged.get("consensus_score", 1.0) or 1.0)
        merged["consensus_score"] = round(max(0.05, consensus - penalty), 4)
        merged["disagreements"] = merged_disagreements
        merged["legal_review"] = legal_review
        if legal_review.get("summary"):
            merged["summary"] = f"{merged.get('summary', '')} {legal_review['summary']}".strip()
        return merged


    def _build_react_tool_requests(
        self,
        actions: list[Any],
        *,
        plan,
        context,
        session_id: str,
        top_k: int,
    ) -> list[Any]:
        """Convert ReAct actions into executable tool requests."""
        from ml_engine.tax_agent_tools import ToolCallRequest

        step_by_tool = {s.tool_name: s for s in plan.steps}
        requests = []
        seen: set[str] = set()
        for action in actions:
            action_name = getattr(getattr(action, "action", None), "value", str(getattr(action, "action", "")))
            if action_name not in {"retry_tool", "add_tool"}:
                continue
            tool_name = getattr(action, "tool_name", None)
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)

            step = step_by_tool.get(tool_name)
            inputs = dict(getattr(step, "tool_inputs", {}) or {})
            inputs.update(getattr(action, "params", {}) or {})
            if not inputs:
                inputs = self._default_tool_inputs(
                    tool_name,
                    context=context,
                    session_id=session_id,
                    top_k=top_k,
                )
            if tool_name == "knowledge_search":
                requested_top_k = int(inputs.get("top_k") or top_k)
                inputs.update({
                    "session_id": session_id,
                    "request_id": f"react-{uuid.uuid4().hex[:8]}",
                    "top_k": max(top_k, requested_top_k),
                    "entity_scope": {
                        "tax_code": context.active_tax_code,
                        "tax_period": context.active_tax_period,
                    },
                })

            requests.append(ToolCallRequest(
                tool_name=tool_name,
                inputs=inputs,
                request_id=f"react-{uuid.uuid4().hex[:8]}",
                timeout_override=(getattr(step, "timeout_ms", 10000) if step else 10000) / 1000.0,
                max_retries_override=getattr(step, "max_retries", 1) if step else 1,
            ))
        return requests

    def _default_tool_inputs(self, tool_name: str, *, context, session_id: str, top_k: int) -> dict[str, Any]:
        if tool_name == "knowledge_search":
            return {"query": "", "intent": "general_tax_query", "top_k": top_k, "session_id": session_id}
        if context.active_tax_code:
            return {"tax_code": context.active_tax_code}
        return {}

    def _debate_requires_escalation(self, debate_result: dict[str, Any] | None) -> bool:
        if not debate_result:
            return False
        consensus = float(debate_result.get("consensus_score", 1.0) or 1.0)
        if consensus < 0.58:
            return True
        severe = {"major", "critical"}
        return any(
            str(d.get("severity", "")).lower() in severe
            for d in debate_result.get("disagreements", [])
        )

    def _confidence_after_debate(self, confidence: float, debate_result: dict[str, Any]) -> float:
        consensus = float(debate_result.get("consensus_score", 1.0) or 1.0)
        severe_count = sum(
            1 for d in debate_result.get("disagreements", [])
            if str(d.get("severity", "")).lower() in {"major", "critical"}
        )
        penalty = (1.0 - consensus) * 0.25 + min(0.2, severe_count * 0.08)
        return round(max(0.05, min(0.98, float(confidence) - penalty)), 4)

    def _persist_route_event(
        self,
        db,
        *,
        session_id: str,
        turn_id: int,
        dialogue_act: str,
        intent: str,
        model_mode: str,
        selected_tools: list[str],
        routing_decision,
    ) -> None:
        nested = None
        try:
            if hasattr(db, "begin_nested"):
                nested = db.begin_nested()
            selected = list(selected_tools or [])
            suppressed = list(getattr(routing_decision, "suppressed_tools", set()) or [])
            db.execute(sql_text("""
                INSERT INTO agent_route_events
                (session_id, turn_id, dialogue_act, intent, answer_contract, model_mode,
                 selected_tools_json, suppressed_tools_json, requested_domain,
                 selected_model_bundle_json, mode_validation_json, mode_mismatch,
                 suggested_mode, suppressed_domains_json, route_confidence,
                 focus_score, route_violation)
                VALUES
                (:session_id, :turn_id, :dialogue_act, :intent, :answer_contract, :model_mode,
                 CAST(:selected_tools_json AS jsonb), CAST(:suppressed_tools_json AS jsonb),
                 :requested_domain, CAST(:selected_model_bundle_json AS jsonb),
                 CAST(:mode_validation_json AS jsonb), :mode_mismatch,
                 :suggested_mode, CAST(:suppressed_domains_json AS jsonb),
                 :route_confidence, :focus_score, :route_violation)
            """), {
                "session_id": session_id,
                "turn_id": turn_id,
                "dialogue_act": dialogue_act,
                "intent": intent,
                "answer_contract": getattr(routing_decision.answer_contract, "value", str(routing_decision.answer_contract)),
                "model_mode": model_mode,
                "selected_tools_json": _json_dumps_safe(selected),
                "suppressed_tools_json": _json_dumps_safe(suppressed),
                "requested_domain": getattr(routing_decision, "requested_domain", None),
                "selected_model_bundle_json": _json_dumps_safe(getattr(routing_decision, "selected_model_bundle", []) or []),
                "mode_validation_json": _json_dumps_safe(getattr(routing_decision, "mode_validation", {}) or {}),
                "mode_mismatch": bool(getattr(routing_decision, "mode_mismatch", False)),
                "suggested_mode": getattr(routing_decision, "suggested_mode", None),
                "suppressed_domains_json": _json_dumps_safe(getattr(routing_decision, "suppressed_domains", []) or []),
                "route_confidence": float(getattr(routing_decision, "route_confidence", 0.0) or 0.0),
                "focus_score": float(getattr(routing_decision, "focus_score", 1.0) or 1.0),
                "route_violation": bool(getattr(routing_decision, "route_violation", False)),
            })
            if nested is not None:
                nested.commit()
        except Exception as exc:
            if nested is not None:
                try:
                    nested.rollback()
                except Exception:
                    pass
            logger.debug("[Orchestrator] route event persist skipped: %s", exc)

    def _persist_execution_plan(
        self,
        db,
        *,
        plan,
        session_id: str,
        turn_id: int,
        query_text: str,
        intent: str,
        tool_results: dict[str, Any],
        synthesis_result,
        compliance,
        latency_breakdown: dict[str, float],
        final_escalate: bool,
    ) -> None:
        nested = None
        try:
            if hasattr(db, "begin_nested"):
                nested = db.begin_nested()
            db.execute(sql_text("""
                INSERT INTO agent_execution_plans
                (plan_id, session_id, turn_id, query_text, intent, complexity,
                 reasoning_trace, budget_ms, max_react_iterations, retry_policy_json,
                 evidence_contract_json, steps_json, tool_results_json, synthesis_json,
                 compliance_json, latency_ms, latency_breakdown)
                VALUES
                (:plan_id, :session_id, :turn_id, :query_text, :intent, :complexity,
                 :reasoning_trace, :budget_ms, :max_react_iterations,
                 CAST(:retry_policy_json AS jsonb), CAST(:evidence_contract_json AS jsonb),
                 CAST(:steps_json AS jsonb), CAST(:tool_results_json AS jsonb),
                 CAST(:synthesis_json AS jsonb), CAST(:compliance_json AS jsonb),
                 :latency_ms, CAST(:latency_breakdown AS jsonb))
                ON CONFLICT (plan_id) DO NOTHING
            """), {
                "plan_id": plan.plan_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "query_text": query_text,
                "intent": intent,
                "complexity": plan.complexity.value,
                "reasoning_trace": plan.reasoning,
                "budget_ms": getattr(plan, "budget_ms", None),
                "max_react_iterations": getattr(plan, "max_react_iterations", None),
                "retry_policy_json": _json_dumps_safe(getattr(plan, "retry_policy", {}) or {}),
                "evidence_contract_json": _json_dumps_safe(getattr(plan, "evidence_contract", {}) or {}),
                "steps_json": _json_dumps_safe([
                    {
                        "step_id": s.step_id,
                        "tool_name": s.tool_name,
                        "description": s.description,
                        "depends_on": s.depends_on,
                        "priority": s.priority,
                        "optional": s.optional,
                        "timeout_ms": getattr(s, "timeout_ms", None),
                        "max_retries": getattr(s, "max_retries", None),
                        "evidence_contract": getattr(s, "evidence_contract", {}),
                    }
                    for s in plan.steps
                ], default=str),
                "tool_results_json": _json_dumps_safe(tool_results),
                "synthesis_json": _json_dumps_safe({
                    "summary": synthesis_result.summary,
                    "confidence": synthesis_result.confidence,
                    "tier": synthesis_result.synthesis_tier,
                    "evidence_count": len(synthesis_result.evidence),
                    "verification": getattr(synthesis_result, "verification", {}),
                    "clarification_needed": getattr(synthesis_result, "clarification_needed", False),
                }, default=str),
                "compliance_json": _json_dumps_safe({
                    "decision": compliance.overall_decision.value,
                    "abstain": compliance.abstain,
                    "escalate": final_escalate,
                }, default=str),
                "latency_ms": sum(float(v or 0) for v in latency_breakdown.values()),
                "latency_breakdown": _json_dumps_safe(latency_breakdown),
            })
            if nested is not None:
                nested.commit()
        except Exception as exc:
            if nested is not None:
                try:
                    nested.rollback()
                except Exception:
                    pass
            logger.debug("[Orchestrator] execution plan persist skipped: %s", exc)

    def _persist_debate_adjudication(
        self,
        db,
        *,
        session_id: str,
        turn_id: int,
        tax_code: str | None,
        debate_result: dict[str, Any] | None,
        final_escalate: bool,
    ) -> None:
        if not debate_result:
            return
        nested = None
        try:
            if hasattr(db, "begin_nested"):
                nested = db.begin_nested()
            entity_id = tax_code or f"{session_id}:{turn_id}"
            status = "open" if final_escalate else "resolved"
            dispute_reason = "; ".join(
                f"{d.get('severity')}:{d.get('topic')}"
                for d in debate_result.get("disagreements", [])[:5]
            )
            db.execute(sql_text("""
                INSERT INTO adjudication_cases
                (entity_type, entity_id, model_name, model_version, model_label,
                 human_label, final_label, status, dispute_reason, resolution_notes)
                VALUES
                ('tax_agent_debate', :entity_id, 'tax_agent_orchestrator',
                 :model_version, :model_label, NULL, :final_label, :status,
                 :dispute_reason, :resolution_notes)
            """), {
                "entity_id": entity_id,
                "model_version": "multi_agent_v2",
                "model_label": debate_result.get("consensus_stance"),
                "final_label": debate_result.get("consensus_label"),
                "status": status,
                "dispute_reason": dispute_reason or None,
                "resolution_notes": _json_dumps_safe({
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "consensus_score": debate_result.get("consensus_score"),
                    "recommendation": debate_result.get("recommendation"),
                    "summary": debate_result.get("summary"),
                }, ensure_ascii=False, default=str),
            })
            if nested is not None:
                nested.commit()
        except Exception as exc:
            if nested is not None:
                try:
                    nested.rollback()
                except Exception:
                    pass
            logger.debug("[Orchestrator] debate adjudication persist skipped: %s", exc)

    def _persist_agent_workspace(
        self,
        db,
        *,
        session_id: str,
        turn_id: int,
        query_text: str,
        intent: str,
        tool_results: dict[str, Any],
        synthesis_result,
        react_reflections: list[dict[str, Any]],
        debate_result: dict[str, Any] | None,
        legal_review: dict[str, Any],
        final_escalate: bool,
        escalation_domain: str,
    ) -> None:
        nested = None
        try:
            if hasattr(db, "begin_nested"):
                nested = db.begin_nested()
            ks = tool_results.get("knowledge_search") or {}
            facts = {
                "query": query_text,
                "intent": intent,
                "tools_used": list(tool_results.keys()),
                "retrieval_hits": len(ks.get("hits", []) or []),
                "graph_context": ks.get("graph_context") or {},
            }
            assumptions = []
            if legal_review.get("disagreements"):
                assumptions.append({
                    "type": "legal_quality",
                    "detail": "Authority/effective-date/scope issues may affect the answer.",
                    "review": legal_review,
                })
            open_questions = list(getattr(synthesis_result, "clarification_questions", []) or [])
            verification = getattr(synthesis_result, "verification", {}) or {}
            for claim in verification.get("unsupported_claims", [])[:6]:
                open_questions.append(f"Verify unsupported claim: {claim.get('claim', '')[:180]}")
            citations = []
            for ev in getattr(synthesis_result, "evidence", [])[:8]:
                if getattr(ev, "source_type", "") == "legal":
                    citations.append({
                        "citation_key": getattr(ev, "citation_key", ""),
                        "title": getattr(ev, "title", ""),
                        "score": getattr(ev, "score", 0.0),
                        "metadata": getattr(ev, "metadata", {}) or {},
                    })
            escalation_reason = None
            if final_escalate:
                escalation_reason = (
                    f"domain={escalation_domain}; verification={verification.get('status')}; "
                    f"legal_review={legal_review.get('status')}; "
                    f"debate={debate_result.get('consensus_score') if debate_result else 'none'}"
                )
            db.execute(sql_text("""
                INSERT INTO agent_case_workspace
                (session_id, turn_id, facts_json, assumptions_json, open_questions_json,
                 citations_json, claim_verification_json, escalation_reason)
                VALUES
                (:session_id, :turn_id, CAST(:facts_json AS jsonb),
                 CAST(:assumptions_json AS jsonb), CAST(:open_questions_json AS jsonb),
                 CAST(:citations_json AS jsonb), CAST(:claim_verification_json AS jsonb),
                 :escalation_reason)
            """), {
                "session_id": session_id,
                "turn_id": turn_id,
                "facts_json": _json_dumps_safe(facts, ensure_ascii=False),
                "assumptions_json": _json_dumps_safe(assumptions, ensure_ascii=False),
                "open_questions_json": _json_dumps_safe(open_questions, ensure_ascii=False),
                "citations_json": _json_dumps_safe(citations, ensure_ascii=False),
                "claim_verification_json": _json_dumps_safe({
                    "verification": verification,
                    "react": react_reflections,
                    "debate": debate_result,
                    "legal_review": legal_review,
                }, ensure_ascii=False, default=str),
                "escalation_reason": escalation_reason,
            })
            if nested is not None:
                nested.commit()
        except Exception as exc:
            if nested is not None:
                try:
                    nested.rollback()
                except Exception:
                    pass
            logger.debug("[Orchestrator] agent workspace persist skipped: %s", exc)

    def _persist_legal_claim_verifications(
        self,
        db,
        *,
        session_id: str,
        turn_id: int,
        synthesis_result,
    ) -> None:
        verification = getattr(synthesis_result, "verification", {}) or {}
        claims = []
        for item in verification.get("verified_claims", []) or []:
            claims.append((item, "supported"))
        for item in verification.get("unsupported_claims", []) or []:
            claims.append((item, "unsupported"))
        if not claims:
            return

        nested = None
        try:
            if hasattr(db, "begin_nested"):
                nested = db.begin_nested()
            for item, status in claims[:32]:
                db.execute(sql_text("""
                    INSERT INTO legal_claim_verifications
                    (session_id, turn_id, claim_text, support_score, evidence_ref,
                     status, metadata_json)
                    VALUES
                    (:session_id, :turn_id, :claim_text, :support_score, :evidence_ref,
                     :status, CAST(:metadata_json AS jsonb))
                """), {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "claim_text": str(item.get("claim", ""))[:1000],
                    "support_score": float(item.get("support_score", 0.0) or 0.0),
                    "evidence_ref": (
                        None if item.get("evidence_index") in (None, -1)
                        else str(item.get("evidence_index"))
                    ),
                    "status": status,
                    "metadata_json": _json_dumps_safe({
                        "verifier": "legal-faithfulness-v1",
                        "synthesis_tier": getattr(synthesis_result, "synthesis_tier", ""),
                    }, ensure_ascii=False, default=str),
                })
            if nested is not None:
                nested.commit()
        except Exception as exc:
            if nested is not None:
                try:
                    nested.rollback()
                except Exception:
                    pass
            logger.debug("[Orchestrator] legal claim verification persist skipped: %s", exc)

    def _enrich_with_sub_agents(
        self,
        answer: str,
        sub_agent_analysis: dict[str, Any],
    ) -> str:
        """Enrich the synthesized answer with sub-agent insights."""
        additions = []

        # Analytics enrichment
        analytics = sub_agent_analysis.get("analytics")
        if analytics:
            risk_level = analytics.get("risk_level", "unknown")
            score = analytics.get("composite_risk_score", 0)
            trend = analytics.get("risk_trend", "stable")
            additions.append(
                f"\n### Phân tích rủi ro tổng hợp\n"
                f"Mức rủi ro: **{risk_level.upper()}** (điểm: {score:.0%}, xu hướng: {trend})"
            )

        # Investigation enrichment
        investigation = sub_agent_analysis.get("investigation")
        if investigation:
            suspicion = investigation.get("suspicion_level", "clear")
            patterns = investigation.get("patterns_count", 0)
            escalation = investigation.get("escalation_level", "routine")
            additions.append(
                f"\n### Kết quả điều tra\n"
                f"Mức nghi vấn: **{suspicion.upper()}** "
                f"({patterns} mẫu đáng ngờ, escalation: {escalation})"
            )
            if investigation.get("recommended_actions"):
                for action in investigation["recommended_actions"][:2]:
                    additions.append(f"- {action}")

        # Legal enrichment
        legal = sub_agent_analysis.get("legal_research")
        if legal and legal.get("applicable_laws"):
            top_laws = legal["applicable_laws"][:3]
            law_refs = [l.get("reference", "") for l in top_laws if l.get("type") == "primary_law"]
            if law_refs:
                additions.append(
                    f"\n### Cơ sở pháp lý bổ sung\n"
                    + "\n".join(f"- {ref}" for ref in law_refs[:3])
                )

        if additions:
            answer += "\n" + "\n".join(additions)

        return answer

    # ─── Helpers ──────────────────────────────────────────────────────────

    INTENT_RULES = {
        "vat_refund_risk": [
            "hoan thue", "vat", "ho so hoan", "refund", "đề nghị hoàn",
            "hoàn thuế", "thuế gtgt", "thuế giá trị gia tăng", "giảm thuế",
            "thuế suất 8", "thuế suất 10", "khấu trừ", "đầu vào", "đầu ra",
            "nghị định 72", "nd 72", "72/2024", "thuế suất", "phương pháp khấu trừ",
            "tỷ lệ % trên doanh thu",
        ],
        "invoice_risk": [
            "hoa don", "invoice", "xuat hoa don", "mua vao", "ban ra",
            "hóa đơn", "hóa đơn điện tử", "máy tính tiền", "xuất hóa đơn",
            "hóa đơn không hợp pháp", "hóa đơn bất hợp pháp", "hóa đơn giả",
            "thông tư 78", "nghị định 123",
        ],
        "delinquency": [
            "no dong", "cham nop", "delinquency", "qua han", "thu no",
            "nợ đọng", "chậm nộp", "quá hạn", "thu nợ", "cưỡng chế",
            "tiền chậm nộp", "tiền phạt", "nhắc nợ",
        ],
        "osint_ownership": [
            "offshore", "so huu", "ubo", "phoenix", "cong ty me",
            "sở hữu", "công ty mẹ", "cấu trúc sở hữu", "người hưởng lợi",
            "pháp nhân nước ngoài", "singapore", "bvi", "cayman",
        ],
        "transfer_pricing": [
            "chuyen gia", "transfer pricing", "gia giao dich lien ket", "mispricing",
            "chuyển giá", "giá giao dịch liên kết", "giao dịch liên kết",
            "nghị định 132", "bên liên kết", "arm's length",
        ],
        "audit_selection": [
            "thanh tra", "audit", "kiem tra", "xep hang ho so",
            "kiểm tra", "xếp hạng hồ sơ", "chọn thanh tra",
        ],
        "general_tax_query": [
            "thuế tndn", "thuế thu nhập doanh nghiệp", "ưu đãi thuế",
            "miễn thuế", "đầu tư mở rộng", "dự án đầu tư",
            "khu công nghiệp", "luật thuế", "quy định", "chính sách thuế",
            "quản lý thuế", "công văn",
            "thuế tncn", "tncn", "lương gross", "khấu trừ thuế",
            "người phụ thuộc", "hoàn thuế", "đóng thừa",
            "hộ kinh doanh", "bán hàng online", "shopee", "tiktok",
            "thương mại điện tử", "cho thuê nhà", "thuế môn bài",
            "nộp tờ khai", "kê khai", "chi phí được trừ", "tiếp khách",
            "thanh toán tiền mặt", "freelancer", "cá nhân kinh doanh",
        ],
    }

    def _rule_based_intent(self, message: str) -> tuple[str, float]:
        """Keyword-based intent fallback."""
        normalized = message.lower()
        plain = normalize_vietnamese_text(message)
        if (
            "top" in plain
            or "danh sach" in plain
            or "liet ke" in plain
            or "cao nhat" in plain
            or "xep hang" in plain
        ) and ("doanh" in plain or "cong ty" in plain or "dn" in plain):
            return "top_n_query", 0.9
        best = ("general_tax_query", 0.15)
        for intent, keywords in self.INTENT_RULES.items():
            score = sum(1 for kw in keywords if kw in normalized or normalize_vietnamese_text(kw) in plain)
            if score > best[1]:
                best = (intent, float(score))
        conf = min(0.95, 0.25 + 0.15 * best[1])
        if best[0] == "general_tax_query":
            conf = 0.22
        return best[0], conf

    def _build_analysis_blocks(
        self,
        *,
        tool_results: dict[str, dict[str, Any]],
        routing_decision,
        synthesis_result,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        contract = getattr(getattr(routing_decision, "answer_contract", None), "value", "")
        if contract == "fraud_analysis" or tool_results.get("_batch_results"):
            batch = tool_results.get("_batch_results") or tool_results.get("_attachment_analysis") or {}
            blocks.append({
                "type": "fraud_analysis",
                "title": "Fraud Risk Analysis",
                "summary": getattr(synthesis_result, "summary", ""),
                "metrics": {
                    "total_records": batch.get("total") or len(batch.get("assessments", []) or []),
                    "risk_distribution": batch.get("by_level", {}),
                    "top_records": (batch.get("top_5") or batch.get("top_risky") or batch.get("assessments") or [])[:10],
                },
                "next_steps": getattr(synthesis_result, "recommendations", []) or [],
            })
        if contract == "vat_graph" or tool_results.get("_vat_graph_batch_results") or tool_results.get("_ocr_document_results"):
            vat = tool_results.get("_vat_graph_batch_results") or tool_results.get("_attachment_analysis") or {}
            graph = vat.get("graph") or {}
            blocks.append({
                "type": "vat_graph",
                "title": "VAT Network Analysis",
                "summary": getattr(synthesis_result, "summary", ""),
                "metrics": {
                    "processed_rows": vat.get("processed_rows") or vat.get("row_count"),
                    "suspect_value": (vat.get("summary") or {}).get("suspect_value"),
                    "rings": graph.get("rings") or graph.get("ring_findings") or [],
                    "top_invoice_risks": graph.get("top_invoice_risks") or [],
                },
                "next_steps": getattr(synthesis_result, "recommendations", []) or [],
            })
        # Multi-agent-only: cross-domain dossier summary (works even if frontend doesn't special-case it)
        try:
            selected_bundle = list(getattr(routing_decision, "selected_model_bundle", []) or [])
        except Exception:
            selected_bundle = []
        if len(selected_bundle) >= 2 or (tool_results.get("_batch_results") and tool_results.get("_vat_graph_batch_results")):
            blocks.append({
                "type": "investigation_dossier",
                "title": "Dossier điều tra (Multi-Agent)",
                "summary": "Hợp nhất đa miền để ưu tiên hành động và chuỗi bằng chứng theo cùng một phiên phân tích.",
                "metrics": {
                    "domains": selected_bundle,
                    "tools": list(tool_results.keys())[:12],
                    "evidence_sources": int(bool(tool_results.get("_batch_results"))) + int(bool(tool_results.get("_vat_graph_batch_results"))) + int(bool(tool_results.get("knowledge_search"))),
                },
                "next_steps": [
                    "Ưu tiên điều tra theo nhóm rủi ro cao và đường đi bằng chứng (graph paths / red flags).",
                    "Nếu kết luận còn mơ hồ: bật vòng lặp tự-bổ-sung bằng chứng (mở rộng subgraph / truy vấn MST / what-if).",
                    "Xuất dossier để handover: giả định, giới hạn dữ liệu, và khuyến nghị thanh tra/thu nợ theo thứ tự.",
                ],
            })
        return blocks

    def _build_fraud_visualization_payload(
        self,
        tool_results: dict[str, dict[str, Any]],
        sub_agent_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        batch = tool_results.get("_batch_results") or {}
        attachment = tool_results.get("_attachment_analysis") or {}
        if not batch and attachment.get("analysis_type") == "risk_csv":
            batch = attachment
        top_n = tool_results.get("top_n_risky_companies") or {}
        companies = list(batch.get("assessments", []) or batch.get("companies", []) or top_n.get("companies", []) or [])
        if not companies and not sub_agent_analysis.get("analytics"):
            return {}

        scores: list[float] = []
        for row in companies:
            try:
                scores.append(float(row.get("risk_score") or row.get("score") or 0.0))
            except Exception:
                scores.append(0.0)
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        max_score = round(max(scores), 2) if scores else 0.0
        analytics = sub_agent_analysis.get("analytics", {})
        if analytics and not max_score:
            max_score = round(float(analytics.get("composite_risk_score", 0.0)) * 100, 2)
            avg_score = max_score

        by_level = dict(batch.get("by_level") or {})
        if not by_level and companies:
            by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for row in companies:
                level = str(row.get("risk_level") or "low").lower()
                by_level[level] = by_level.get(level, 0) + 1

        years: dict[str, list[float]] = {}
        scatter = []
        for row in companies[:200]:
            year = str(row.get("year") or row.get("tax_year") or "")
            if year:
                try:
                    years.setdefault(year, []).append(float(row.get("risk_score") or 0.0))
                except Exception:
                    pass
            try:
                scatter.append({
                    "x": float(row.get("revenue") or row.get("total_revenue") or 0.0),
                    "y": float(row.get("risk_score") or 0.0),
                    "label": row.get("company_name") or row.get("tax_code") or "",
                    "industry": row.get("industry") or "",
                })
            except Exception:
                continue
        yearly = [
            {"year": year, "avg_risk": round(sum(vals) / len(vals), 2), "count": len(vals)}
            for year, vals in sorted(years.items())
        ]
        sorted_scores = sorted(scores, reverse=True)
        total_risk = sum(sorted_scores) or 1.0
        cumulative = []
        running = 0.0
        for idx, score in enumerate(sorted_scores[:50], 1):
            running += score
            cumulative.append({"rank": idx, "cumulative_pct": round(running / total_risk * 100, 2)})

        return {
            "summary": {
                "total": int(batch.get("total") or top_n.get("total") or len(companies)),
                "avg_risk": avg_score,
                "max_risk": max_score,
                "source": batch.get("filename") or top_n.get("source") or "agent",
            },
            "risk_gauge": {
                "score": max_score,
                "level": "critical" if max_score >= 90 else "high" if max_score >= 70 else "medium" if max_score >= 40 else "low",
                "color": "#DC2626" if max_score >= 90 else "#F97316" if max_score >= 70 else "#EAB308" if max_score >= 40 else "#16A34A",
                "confidence": 85,
            },
            "radar": {
                "labels": ["Compliance", "Financial", "VAT", "Network"],
                "values": [
                    round(min(100.0, avg_score * 0.95 + 5), 2),
                    round(min(100.0, avg_score * 1.05), 2),
                    round(min(100.0, max_score * 0.9), 2),
                    round(min(100.0, max_score), 2),
                ],
            },
            "yearly_trend": yearly,
            "revenue_risk_scatter": scatter[:120],
            "risk_distribution": by_level,
            "cumulative_risk": cumulative,
            "top_companies": companies[:10],
            "case_narrative": analytics.get("summary") or "",
            "cross_model_consensus": {
                "analytics": sub_agent_analysis.get("analytics", {}),
                "investigation": sub_agent_analysis.get("investigation", {}),
            },
        }

    def _build_vat_visualization_payload(self, tool_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        vat = tool_results.get("_vat_graph_batch_results") or {}
        attachment = tool_results.get("_attachment_analysis") or {}
        if not vat and attachment.get("analysis_type") == "vat_graph_csv":
            vat = attachment
        ocr = tool_results.get("_ocr_document_results") or {}
        if not vat and not ocr:
            return {}
        graph = vat.get("graph") or {}
        summary = vat.get("summary") or {}
        nodes = graph.get("nodes") or graph.get("graph_nodes") or []
        edges = graph.get("edges") or graph.get("graph_edges") or []
        top_invoice_risks = graph.get("top_invoice_risks") or []
        rings = graph.get("rings") or graph.get("ring_findings") or graph.get("motifs") or []
        logs = graph.get("forensic_logs") or graph.get("audit_log") or graph.get("findings") or []
        return {
            "summary": {
                "batch_id": vat.get("batch_id"),
                "row_count": vat.get("row_count"),
                "processed_rows": vat.get("processed_rows"),
                "suspect_value": summary.get("suspect_value") or summary.get("total_suspicious_amount"),
                "warnings": vat.get("warnings") or [],
            },
            "graph": {"nodes": nodes[:300], "edges": edges[:500]},
            "timeline": graph.get("timeline") or graph.get("monthly_edges") or [],
            "risk_bars": graph.get("risk_bars") or top_invoice_risks[:20],
            "model_intelligence": graph.get("gnn_scores") or graph.get("model_intelligence") or {},
            "ring_scoring": {"rings": rings[:20], "count": len(rings)},
            "ownership_summary": graph.get("ownership_summary") or graph.get("ownership") or {},
            "forensic_logs": logs[:30] if isinstance(logs, list) else logs,
            "evidence_paths": graph.get("evidence_paths") or graph.get("paths") or [],
            "cross_border_signals": graph.get("cross_border_signals") or graph.get("offshore_signals") or {},
            "top_invoice_risks": top_invoice_risks[:20],
            "ocr_invoice": ocr if ocr else {},
        }

    def _build_visualization_data(
        self,
        tool_results: dict[str, dict[str, Any]],
        sub_agent_analysis: dict[str, Any],
        plan,
        latency_breakdown: dict[str, float],
    ) -> dict[str, Any]:
        """Build chart-ready visualization data for the frontend."""
        viz: dict[str, Any] = {}
        fraud_payload = self._build_fraud_visualization_payload(tool_results, sub_agent_analysis)
        if fraud_payload:
            viz["fraud"] = fraud_payload
        vat_payload = self._build_vat_visualization_payload(tool_results)
        if vat_payload:
            viz["vat"] = vat_payload

        # 1. Risk Gauge — from analytics sub-agent
        analytics = sub_agent_analysis.get("analytics", {})
        if analytics:
            score = float(analytics.get("composite_risk_score", 0))
            level = str(analytics.get("risk_level", "unknown"))
            color_map = {"critical": "#DC2626", "high": "#F97316",
                         "moderate": "#EAB308", "low": "#22C55E", "minimal": "#06B6D4"}
            viz["risk_gauge"] = {
                "score": round(score * 100, 1),
                "level": level,
                "color": color_map.get(level, "#64748B"),
                "confidence": round(float(analytics.get("confidence", 0)) * 100, 1),
            }

        # 2. Delinquency Timeline — ML vs DL comparison
        ml_dq = tool_results.get("delinquency_check", {})
        dl_dq = tool_results.get("temporal_delinquency_deep", {})
        if ml_dq.get("status") == "analyzed" or dl_dq.get("status") == "analyzed":
            viz["delinquency_timeline"] = {
                "labels": ["30 ngay", "60 ngay", "90 ngay"],
                "ml_values": [
                    round(float(ml_dq.get("prob_30d", 0)) * 100, 1),
                    round(float(ml_dq.get("prob_60d", 0)) * 100, 1),
                    round(float(ml_dq.get("prob_90d", 0)) * 100, 1),
                ] if ml_dq.get("status") == "analyzed" else [],
                "dl_values": [
                    round(float(dl_dq.get("prob_30d", 0)) * 100, 1),
                    round(float(dl_dq.get("prob_60d", 0)) * 100, 1),
                    round(float(dl_dq.get("prob_90d", 0)) * 100, 1),
                ] if dl_dq.get("status") == "analyzed" else [],
                "dl_architecture": dl_dq.get("architecture", ""),
            }
            # Sequence features for detail chart
            if dl_dq.get("sequence_features"):
                viz["delinquency_timeline"]["sequence_features"] = dl_dq["sequence_features"]

        # 3. Anomaly Scatter — VAE results
        vae = tool_results.get("vae_anomaly_scan", {})
        if vae.get("status") == "analyzed":
            viz["anomaly_scatter"] = {
                "total": vae.get("total_invoices", 0),
                "anomaly_count": vae.get("anomaly_count", 0),
                "anomaly_ratio": round(float(vae.get("anomaly_ratio", 0)) * 100, 1),
                "threshold": vae.get("threshold", 0),
                "top_anomalies": vae.get("top_anomalies", [])[:8],
                "distribution": vae.get("error_distribution", {}),
                "architecture": vae.get("architecture", ""),
            }

        # 4. Network Mini Graph — HeteroGNN
        hgnn = tool_results.get("hetero_gnn_risk", {})
        if hgnn.get("status") == "analyzed":
            nodes = [{"id": hgnn.get("tax_code", ""), "type": "company",
                      "risk": hgnn.get("fraud_probability", 0), "label": "Target"}]
            edges = []
            for i, nb in enumerate(hgnn.get("neighbor_risk_summary", [])[:6]):
                nodes.append({"id": nb["tax_code"], "type": "company",
                              "risk": 0.3, "label": f"Neighbor {i+1}"})
                edges.append({"source": hgnn.get("tax_code", ""),
                              "target": nb["tax_code"],
                              "weight": nb.get("invoices", 1)})
            viz["network_graph"] = {
                "nodes": nodes,
                "edges": edges,
                "node_type_scores": hgnn.get("node_type_scores", {}),
                "architecture": hgnn.get("architecture", ""),
            }

        # 5. Uplift Action Bars — Causal Inference
        uplift = tool_results.get("causal_uplift_recommend", {})
        if uplift.get("status") == "analyzed":
            viz["uplift_actions"] = {
                "cate_score": uplift.get("cate_score", 0),
                "recommended": uplift.get("recommended_action", ""),
                "actions": uplift.get("action_ranking", []),
                "architecture": uplift.get("architecture", ""),
            }

        # 6. Model Comparison Table
        models_used = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get("model"):
                models_used.append({
                    "tool": tool_name,
                    "model": result.get("model", ""),
                    "architecture": result.get("architecture", ""),
                    "risk_level": result.get("risk_level", ""),
                    "status": result.get("status", ""),
                })
        if models_used:
            viz["model_comparison"] = models_used

        macro = tool_results.get("macro_forecast", {})
        if isinstance(macro, dict) and macro.get("status") in {"analyzed", "success"}:
            macro_result = self._plain_payload(macro.get("result") or macro.get("baseline") or {})
            if isinstance(macro_result, dict):
                quarterly = macro_result.get("quarterly_projection") or []
                industries = macro_result.get("industry_impacts") or []
                viz["macro"] = {
                    "action": macro.get("action") or "run",
                    "scenario_name": macro_result.get("scenario_name") or "Macro simulation",
                    "parameters": macro_result.get("parameters") or {},
                    "kpis": {
                        "baseline_total_companies": macro_result.get("baseline_total_companies"),
                        "baseline_high_risk_count": macro_result.get("baseline_high_risk_count"),
                        "simulated_high_risk_count": macro_result.get("simulated_high_risk_count"),
                        "delta_high_risk": macro_result.get("delta_high_risk"),
                        "baseline_delinquency_rate": macro_result.get("baseline_delinquency_rate"),
                        "simulated_delinquency_rate": macro_result.get("simulated_delinquency_rate"),
                        "baseline_estimated_loss": macro_result.get("baseline_estimated_loss"),
                        "simulated_estimated_loss": macro_result.get("simulated_estimated_loss"),
                        "delta_estimated_loss": macro_result.get("delta_estimated_loss"),
                        "baseline_total_revenue": macro_result.get("baseline_total_revenue"),
                        "simulated_total_revenue": macro_result.get("simulated_total_revenue"),
                        "delta_revenue": macro_result.get("delta_revenue"),
                        "delta_revenue_pct": macro_result.get("delta_revenue_pct"),
                        "scenario_health_score": macro_result.get("scenario_health_score"),
                    },
                    "quarterly_projection": quarterly[:60] if isinstance(quarterly, list) else [],
                    "industry_impacts": industries[:30] if isinstance(industries, list) else [],
                    "risk_distribution": macro_result.get("risk_distribution") or {},
                    "generated_at": macro_result.get("generated_at"),
                }
                viz["macro_kpis"] = viz["macro"]["kpis"]

        # 6.5 OCR Document Extraction
        ocr_result = tool_results.get("_ocr_document_results")
        if ocr_result and isinstance(ocr_result, dict):
            viz["ocr_extraction"] = {
                "tables": ocr_result.get("tables", []),
                "extracted_fields": ocr_result.get("extracted_fields", {}),
                "table_extraction_method": ocr_result.get("table_extraction_method", "none"),
                "confidence": ocr_result.get("confidence", 0),
            }

        # 7. Tool Execution Timeline
        timeline = []
        for step in plan.steps:
            result = tool_results.get(step.tool_name, {})
            timeline.append({
                "tool": step.tool_name,
                "description": step.description,
                "status": result.get("status", "skipped"),
                "latency_ms": round(float(result.get("_latency_ms", 0) or 0), 0),
            })
        viz["tool_timeline"] = timeline

        # 8. Top-N Companies Table — from NL query
        top_n_data = tool_results.get("top_n_risky_companies", {})
        if top_n_data.get("companies"):
            viz["top_companies"] = {
                "columns": ["stt", "tax_code", "company_name", "industry", "risk_score", "risk_level"],
                "rows": top_n_data["companies"],
                "total": top_n_data.get("total", 0),
                "sort_by": "risk_score",
                "clickable": True,
            }

        # 9. Batch Analysis Summary — from CSV upload
        batch_data = tool_results.get("_batch_results", {})
        if batch_data and isinstance(batch_data, dict) and batch_data.get("total"):
            by_level = batch_data.get("by_level", {})
            viz["batch_summary"] = {
                "total": batch_data["total"],
                "by_level": by_level,
                "top_5": batch_data.get("top_5", []),
                "filename": batch_data.get("filename", ""),
            }
            viz["batch_risk_distribution"] = {
                "labels": ["Rất cao", "Cao", "Trung bình", "An toàn"],
                "values": [
                    by_level.get("critical", 0),
                    by_level.get("high", 0),
                    by_level.get("medium", 0),
                    by_level.get("low", 0),
                ],
                "colors": ["#DC2626", "#EA580C", "#EAB308", "#16A34A"],
            }

        vat_batch = tool_results.get("_vat_graph_batch_results", {})
        if vat_batch and isinstance(vat_batch, dict):
            summary = vat_batch.get("summary", {}) if isinstance(vat_batch.get("summary"), dict) else {}
            viz["vat_graph_batch"] = {
                "batch_id": vat_batch.get("batch_id"),
                "filename": vat_batch.get("filename", ""),
                "processed_rows": vat_batch.get("processed_rows", 0),
                "companies": summary.get("companies"),
                "invoices": summary.get("invoices"),
                "cycles": summary.get("cycles"),
                "total_suspicious_amount": summary.get("total_suspicious_amount"),
                "top_edges": summary.get("top_edges", []),
                "top_nodes": summary.get("top_nodes", []),
            }

        ocr_data = tool_results.get("_ocr_document_results", {})
        if ocr_data and isinstance(ocr_data, dict):
            viz["ocr_extraction"] = {
                "filename": ocr_data.get("filename", ""),
                "confidence": ocr_data.get("confidence", 0.0),
                "extracted_fields": ocr_data.get("extracted_fields", {}),
                "tables": ocr_data.get("tables", []),
                "table_extraction_method": ocr_data.get("table_extraction_method", "none"),
                "invoice_risk": ocr_data.get("invoice_risk", {}),
                "graph_linkage_candidates": ocr_data.get("graph_linkage_candidates", []),
                "warnings": ocr_data.get("warnings", []),
            }

        # 10. Company Name Search Results
        name_search = tool_results.get("company_name_search", {})
        if name_search.get("matches"):
            viz["company_search_results"] = {
                "matches": name_search["matches"],
                "query": name_search.get("query", ""),
            }

        # 11. XAI Explainability — SHAP waterfall, VAE breakdown, counterfactual
        try:
            from ml_engine.tax_agent_xai import XAIExplainer
            xai = XAIExplainer()
            xai_data = xai.explain_all(tool_results, top_k=8)
            if xai_data:
                viz.update(xai_data)
        except Exception as exc:
            logger.debug("[Orchestrator] XAI skipped: %s", exc)

        # 12. Knowledge Graph Citation Subgraph — from GraphRAG
        ks_result = tool_results.get("knowledge_search", {})
        graph_context = ks_result.get("graph_context")
        if graph_context and isinstance(graph_context, dict):
            subgraph = graph_context.get("subgraph", {})
            if subgraph.get("nodes"):
                viz["knowledge_graph"] = {
                    "nodes": subgraph.get("nodes", []),
                    "edges": subgraph.get("edges", []),
                    "anchor_entities": graph_context.get("anchor_entities", []),
                    "traversal_path": graph_context.get("traversal_path", []),
                    "expansion_depth": graph_context.get("expansion_depth", 0),
                    "total_entities": graph_context.get("total_entities", 0),
                    "total_relations": graph_context.get("total_relations", 0),
                    "latency_ms": graph_context.get("latency_ms", 0),
                    "retrieval_tier": ks_result.get("retrieval_tier", ""),
                }

        return viz

    def _ensure_session(self, db, *, session_id: str, user_id: int | None) -> None:
        """Ensure session exists in DB."""
        row = db.execute(
            sql_text("SELECT 1 FROM agent_sessions WHERE session_id = :session_id"),
            {"session_id": session_id},
        ).fetchone()
        if row:
            return
        db.execute(
            sql_text("""
                INSERT INTO agent_sessions (session_id, user_id, channel, status, metadata_json)
                VALUES (:session_id, :user_id, 'chat', 'active', CAST(:metadata_json AS jsonb))
            """),
            {
                "session_id": session_id,
                "user_id": user_id,
                "metadata_json": _json_dumps_safe({"source": "multi_agent_orchestrator_v2"}),
            },
        )


# ─── Singleton ────────────────────────────────────────────────────────────────

_orchestrator_instance: TaxAgentOrchestrator | None = None


def get_orchestrator() -> TaxAgentOrchestrator:
    """Get or create the singleton orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TaxAgentOrchestrator()
    return _orchestrator_instance
