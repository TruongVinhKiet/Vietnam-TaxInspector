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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Model Mode Profiles — tool/sub-agent selection per mode
# ═══════════════════════════════════════════════════════════════════════

MODE_TOOL_PROFILES: dict[str, dict[str, Any]] = {
    "fraud": {
        "required_tools": [
            "knowledge_search", "company_risk_lookup", "invoice_risk_scan",
            "gnn_analysis", "hetero_gnn_risk", "vae_anomaly_scan",
            "motif_detection", "ownership_analysis",
            "nlp_red_flag_scan",
        ],
        "optional_tools": ["temporal_delinquency_deep", "ring_scoring",
                           "entity_resolution_check"],
        "sub_agents": ["legal", "analytics", "investigation"],
        "label": "🔍 Phân tích Gian lận",
    },
    "vat": {
        "required_tools": [
            "knowledge_search", "company_risk_lookup", "invoice_risk_scan",
            "vat_refund_risk", "vae_anomaly_scan",
            "nlp_red_flag_scan",
        ],
        "optional_tools": ["gnn_analysis"],
        "sub_agents": ["legal", "analytics"],
        "label": "📄 Rủi ro VAT",
    },
    "delinquency": {
        "required_tools": [
            "knowledge_search", "company_risk_lookup", "delinquency_check",
            "temporal_delinquency_deep", "causal_uplift_recommend",
            "revenue_forecast",
        ],
        "optional_tools": [],
        "sub_agents": ["legal", "analytics"],
        "label": "📊 Dự báo Nợ động",
    },
    "macro": {
        "required_tools": ["knowledge_search", "macro_forecast"],
        "optional_tools": ["revenue_forecast"],
        "sub_agents": ["legal"],
        "label": "🌐 Mô phỏng Vĩ mô",
    },
    "legal": {
        "required_tools": ["knowledge_search", "company_risk_lookup"],
        "optional_tools": ["nlp_red_flag_scan"],
        "sub_agents": ["legal"],
        "label": "⚖️ Tư vấn Pháp lý",
    },
    "full": {
        "required_tools": None,  # Use all tools from planner
        "optional_tools": [],
        "sub_agents": ["legal", "analytics", "investigation"],
        "label": "⚡ Toàn diện",
    },
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
        # Phase 3: Sub-agents
        self._legal_agent = None
        self._analytics_agent = None
        self._investigation_agent = None
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

        self._initialized = True
        logger.info(
            "[Orchestrator] ✓ Initialized all components "
            "(tools=%d, embedding=%s, sub_agents=3)",
            self._tool_registry.count(),
            self._embedding_engine.model_tier,
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
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  STREAMING VERSION — yields SSE events at each pipeline step
    # ═══════════════════════════════════════════════════════════════════════

    def process_streaming(
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
        latency_breakdown: dict[str, float] = {}

        self._ensure_initialized(db)
        self._compliance_gate.db = db
        self._memory.db = db

        # ─── Step 1: Context ────────────────────────────────────────────
        yield {"event": "thinking", "data": {"step": "context", "detail": "Đang xây dựng ngữ cảnh hội thoại..."}}
        t0 = time.perf_counter()
        self._ensure_session(db, session_id=session_id, user_id=user_id)

        turn_index_row = db.execute(
            sql_text(
                "SELECT COALESCE(MAX(turn_index), 0) FROM agent_turns "
                "WHERE session_id = :session_id"
            ),
            {"session_id": session_id},
        ).fetchone()
        turn_index = int(turn_index_row[0] or 0) + 1

        db.execute(
            sql_text("""
                INSERT INTO agent_turns (session_id, turn_index, role, message_text)
                VALUES (:session_id, :turn_index, 'user', :message_text)
            """),
            {"session_id": session_id, "turn_index": turn_index, "message_text": message},
        )

        context = self._memory.build_context(session_id, turn_index, message)
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
            )
            if conv_intel_result.resolved_message != message:
                message = conv_intel_result.resolved_message
                yield {"event": "thinking", "data": {
                    "step": "conv_intel", "detail": f"Đã giải quyết ngữ cảnh: {message[:60]}...",
                }}
            if conv_intel_result.is_ambiguous and conv_intel_result.clarification_prompt:
                # Return clarification instead of running full pipeline
                yield {"event": "text_chunk", "data": {"chunk": conv_intel_result.clarification_prompt}}
                yield {"event": "done", "data": {
                    "session_id": session_id, "answer": conv_intel_result.clarification_prompt,
                    "intent": "clarification", "is_ambiguous": True,
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

        yield {"event": "thinking", "data": {
            "step": "intent_done",
            "detail": f"Intent: {intent} ({intent_conf:.0%})",
            "intent": intent,
            "confidence": round(intent_conf, 4),
        }}

        # ─── Step 2.5: NL Query Fast Paths ──────────────────────────────
        t0 = time.perf_counter()
        nl_results = {}
        try:
            from ml_engine.tax_agent_nl_query import NLQueryExecutor
            nl_executor = NLQueryExecutor()

            if intent == "top_n_query":
                yield {"event": "thinking", "data": {"step": "nl_query", "detail": "Đang truy vấn top DN rủi ro..."}}
                quantity = 10
                if multi_intent_result and multi_intent_result.extracted_entities:
                    for ent in multi_intent_result.extracted_entities:
                        if ent.get("type") == "quantity":
                            quantity = min(50, max(1, int(ent["value"])))
                nl_results = nl_executor.execute_top_n(db, n=quantity, sort_by="risk_score", mode=model_mode)

            elif intent == "company_name_lookup":
                company_name = ""
                if multi_intent_result and multi_intent_result.extracted_entities:
                    for ent in multi_intent_result.extracted_entities:
                        if ent.get("type") == "company_name":
                            company_name = ent["value"]
                if company_name:
                    yield {"event": "thinking", "data": {"step": "nl_query", "detail": f"Đang tìm DN: {company_name}..."}}
                    nl_results = nl_executor.execute_company_name_search(db, name=company_name)
                    matches = nl_results.get("matches", [])
                    if len(matches) == 1:
                        context.active_tax_code = matches[0]["tax_code"]

            if attachment_analysis:
                analysis_type = str(attachment_analysis.get("analysis_type") or attachment_analysis.get("detected_schema", "attachment"))
                yield {"event": "thinking", "data": {"step": "attachment", "detail": f"Da phan tich tep dinh kem: {analysis_type}"}}
                nl_results["_attachment_analysis"] = attachment_analysis
                if analysis_type == "risk_csv":
                    intent = "batch_analysis"
                    nl_results["_batch_results"] = attachment_analysis
                elif analysis_type == "vat_graph_csv":
                    intent = "vat_network_analysis"
                    nl_results["_vat_graph_batch_results"] = attachment_analysis
                elif analysis_type == "ocr_invoice":
                    intent = "invoice_risk"
                    nl_results["_ocr_document_results"] = attachment_analysis

            if csv_attachment:
                intent = "batch_analysis"
                yield {"event": "thinking", "data": {"step": "batch", "detail": f"Đang phân tích file {csv_attachment.get('filename', 'CSV')}..."}}
                batch_result = nl_executor.execute_batch_inline(
                    db, csv_content=csv_attachment["content"], filename=csv_attachment["filename"],
                )
                nl_results["_batch_results"] = batch_result
        except Exception as exc:
            logger.warning("[Orchestrator:stream] NL query error: %s", exc)
        latency_breakdown["nl_query"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 3: Planning ───────────────────────────────────────────
        yield {"event": "thinking", "data": {"step": "planning", "detail": "Đang lập kế hoạch phân tích..."}}
        t0 = time.perf_counter()

        mode_profile = MODE_TOOL_PROFILES.get(model_mode, MODE_TOOL_PROFILES["full"])
        allowed_tools = mode_profile.get("required_tools")
        allowed_optional = set(mode_profile.get("optional_tools", []))
        allowed_sub_agents = set(mode_profile.get("sub_agents", ["legal", "analytics", "investigation"]))
        plan = self._planner.plan(
            query=message,
            intent=intent,
            intent_confidence=intent_conf,
            tax_code=context.active_tax_code,
            tax_period=context.active_tax_period,
            context_intents=context.active_intent_history,
        )
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
                if step.optional and intent_conf <= 0.6:
                    continue
                if allowed_tools is not None and step.tool_name not in allowed_tools and step.tool_name not in allowed_optional:
                    continue
                tool_inputs = dict(step.tool_inputs)
                request_id = f"req-{uuid.uuid4().hex[:8]}"
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
            if ks_hits and "legal" in allowed_sub_agents:
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

            if context.active_tax_code and plan.complexity.value in ("moderate", "complex", "investigation") and "analytics" in allowed_sub_agents:
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

            if context.active_tax_code and plan.complexity.value in ("complex", "investigation") and "investigation" in allowed_sub_agents:
                if intent in ("osint_ownership", "invoice_risk", "vat_refund_risk"):
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
        turn_row = db.execute(
            sql_text("""
                INSERT INTO agent_turns (session_id, turn_index, role, message_text)
                VALUES (:session_id, :turn_index, 'assistant', '') RETURNING id
            """),
            {"session_id": session_id, "turn_index": turn_index + 1},
        ).fetchone()
        assistant_turn_id = int(turn_row[0]) if turn_row else 0

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
            "citations_json": _json.dumps(citations),
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
            "evidence_json": _json.dumps({
                "plan": {
                    "complexity": plan.complexity.value,
                    "tools": [s.tool_name for s in plan.steps],
                    "budget_ms": getattr(plan, "budget_ms", None),
                    "retry_policy": getattr(plan, "retry_policy", {}),
                    "evidence_contract": getattr(plan, "evidence_contract", {}),
                },
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
                "tool_input": _json.dumps(next((s.tool_inputs for s in plan.steps if s.tool_name == tool_name), {})),
                "tool_output": _json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, default=str),
                "status": result.get("status", "unknown"), "latency_ms": result.get("_latency_ms"),
            })

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

        legal_workspace = {
            "facts": facts,
            "assumptions": assumptions,
            "open_questions": open_questions,
            "verifications": verifications,
            "escalations": escalations,
        }

        yield {"event": "done", "data": {
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
            "plan_budget_ms": budget_ms,
            "retry_policy": getattr(plan, "retry_policy", {}),
            "evidence_contract": getattr(plan, "evidence_contract", {}),
            "react_reflections": react_reflections,
            "debate": debate_result_dict,
            "legal_review": legal_review,
            "legal_workspace": legal_workspace,
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
                "retry_policy_json": json.dumps(getattr(plan, "retry_policy", {}) or {}, default=str),
                "evidence_contract_json": json.dumps(getattr(plan, "evidence_contract", {}) or {}, default=str),
                "steps_json": json.dumps([
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
                "tool_results_json": json.dumps(tool_results, default=str),
                "synthesis_json": json.dumps({
                    "summary": synthesis_result.summary,
                    "confidence": synthesis_result.confidence,
                    "tier": synthesis_result.synthesis_tier,
                    "evidence_count": len(synthesis_result.evidence),
                    "verification": getattr(synthesis_result, "verification", {}),
                    "clarification_needed": getattr(synthesis_result, "clarification_needed", False),
                }, default=str),
                "compliance_json": json.dumps({
                    "decision": compliance.overall_decision.value,
                    "abstain": compliance.abstain,
                    "escalate": final_escalate,
                }, default=str),
                "latency_ms": sum(float(v or 0) for v in latency_breakdown.values()),
                "latency_breakdown": json.dumps(latency_breakdown, default=str),
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
                "resolution_notes": json.dumps({
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
                "facts_json": json.dumps(facts, ensure_ascii=False, default=str),
                "assumptions_json": json.dumps(assumptions, ensure_ascii=False, default=str),
                "open_questions_json": json.dumps(open_questions, ensure_ascii=False, default=str),
                "citations_json": json.dumps(citations, ensure_ascii=False, default=str),
                "claim_verification_json": json.dumps({
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
                    "metadata_json": json.dumps({
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
        ],
    }

    def _rule_based_intent(self, message: str) -> tuple[str, float]:
        """Keyword-based intent fallback."""
        normalized = message.lower()
        best = ("general_tax_query", 0.15)
        for intent, keywords in self.INTENT_RULES.items():
            score = sum(1 for kw in keywords if kw in normalized)
            if score > best[1]:
                best = (intent, float(score))
        conf = min(0.95, 0.25 + 0.15 * best[1])
        if best[0] == "general_tax_query":
            conf = 0.22
        return best[0], conf

    def _build_visualization_data(
        self,
        tool_results: dict[str, dict[str, Any]],
        sub_agent_analysis: dict[str, Any],
        plan,
        latency_breakdown: dict[str, float],
    ) -> dict[str, Any]:
        """Build chart-ready visualization data for the frontend."""
        viz: dict[str, Any] = {}

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
            viz["ocr_invoice"] = {
                "filename": ocr_data.get("filename", ""),
                "confidence": ocr_data.get("confidence", 0.0),
                "extracted_fields": ocr_data.get("extracted_fields", {}),
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
                "metadata_json": json.dumps({"source": "multi_agent_orchestrator_v2"}),
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
