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
    # Full traces for audit
    policy_traces: list[dict[str, Any]]
    plan_steps: list[dict[str, Any]]
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
        self._tool_executor = ToolExecutor(self._tool_registry)

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
        t_total_start = time.perf_counter()
        latency_breakdown: dict[str, float] = {}

        self._ensure_initialized(db)

        # Update compliance gate's DB reference
        self._compliance_gate.db = db
        self._memory.db = db

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
                    "multi_intent": multi_intent_result.all_intents[:3],
                    "classifier_tier": multi_intent_result.tier,
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
            intent=intent,
            intent_confidence=intent_conf,
            entities={"tax_code": context.active_tax_code, "tax_period": context.active_tax_period},
            context_summary=context.session_summary or "",
            model_mode=model_mode,
        )
        latency_breakdown["planning"] = (time.perf_counter() - t0) * 1000.0

        yield {"event": "thinking", "data": {
            "step": "plan_done",
            "detail": f"Kế hoạch: {plan.complexity.value} — {len(plan.steps)} tools",
            "tools": [s.tool_name for s in plan.steps],
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
                requests.append(ToolCallRequest(
                    tool_name=step.tool_name,
                    inputs=step.tool_inputs,
                    request_id=f"req-{uuid.uuid4().hex[:8]}",
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

        latency_breakdown["tools"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 4.5: Sub-Agent Dispatch ───────────────────────────────
        t0 = time.perf_counter()
        sub_agent_analysis = {}

        try:
            ks_hits = all_tool_results.get("knowledge_search", {}).get("hits", [])
            if ks_hits and "legal" in allowed_sub_agents:
                yield {"event": "sub_agent", "data": {"agent": "legal", "status": "running", "detail": "Phân tích pháp lý..."}}
                legal_opinion = self._legal_agent.research(
                    query=message, retrieval_results=ks_hits, intent=intent, tax_code=context.active_tax_code,
                )
                sub_agent_analysis["legal_research"] = {
                    "analysis": legal_opinion.analysis, "conclusion": legal_opinion.conclusion,
                    "citation_chain": legal_opinion.citation_chain[:5], "authority_score": legal_opinion.authority_score,
                    "confidence": legal_opinion.confidence, "caveats": legal_opinion.caveats,
                    "applicable_laws": legal_opinion.applicable_laws[:5],
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

        # ─── Step 5: Synthesis ──────────────────────────────────────────
        yield {"event": "thinking", "data": {"step": "synthesis", "detail": "Đang tổng hợp câu trả lời..."}}
        t0 = time.perf_counter()

        enriched_tool_results = dict(all_tool_results)
        for agent_name, analysis in sub_agent_analysis.items():
            enriched_tool_results[f"_sub_agent_{agent_name}"] = analysis

        synthesis_result = self._synthesizer.synthesize(
            query=message, intent=intent, tool_results=all_tool_results,
            reasoning_trace=plan.reasoning, tax_code=context.active_tax_code,
        )
        answer = self._synthesizer.format_response_text(synthesis_result)
        answer = self._enrich_with_sub_agents(answer, sub_agent_analysis)
        latency_breakdown["synthesis"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 5.5: ReAct Self-Reflection ─────────────────────────────
        react_reflections = []
        try:
            from ml_engine.tax_agent_react import ReActEngine
            react = ReActEngine()
            planned_tool_names = [s.tool_name for s in plan.steps]
            reflection = react.reflect(
                tool_results=all_tool_results,
                planned_tools=planned_tool_names,
                intent=intent,
                iteration=0,
                sub_agent_analysis=sub_agent_analysis,
            )
            react_reflections.append(reflection.to_dict())
            if reflection.observations:
                yield {"event": "thinking", "data": {
                    "step": "react", "detail": reflection.summary,
                }}
        except Exception as exc:
            logger.debug("[Orchestrator:stream] ReAct error: %s", exc)

        # Stream answer text in chunks
        chunk_size = 80
        for i in range(0, len(answer), chunk_size):
            yield {"event": "text_chunk", "data": {"chunk": answer[i:i + chunk_size]}}

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

        latency_breakdown["compliance"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 7: Audit ──────────────────────────────────────────────
        t0 = time.perf_counter()
        citations = []
        for ev in synthesis_result.evidence[:5]:
            if ev.source_type == "legal":
                citations.append({
                    "chunk_key": ev.metadata.get("chunk_key", ""),
                    "title": ev.title, "score": round(ev.score, 4), "citation_key": ev.citation_key,
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
            "abstained": compliance.abstain, "escalation_required": compliance.escalate,
            "evidence_json": _json.dumps({
                "plan": {"complexity": plan.complexity.value, "tools": [s.tool_name for s in plan.steps]},
                "compliance": {"decision": compliance.overall_decision.value},
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

        self._memory.persist_entities(session_id, context.active_entities)
        db.commit()
        latency_breakdown["audit"] = (time.perf_counter() - t0) * 1000.0
        total_latency = (time.perf_counter() - t_total_start) * 1000.0

        # Visualization data
        viz_data = self._build_visualization_data(all_tool_results, sub_agent_analysis, plan, latency_breakdown)
        if debate_result_dict:
            viz_data["agent_debate"] = debate_result_dict
        if viz_data:
            yield {"event": "viz_data", "data": viz_data}

        # ─── Final done event with full response ────────────────────────
        retrieval_context = all_tool_results.get("knowledge_search", {}).get("hits", [])
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
            "escalation_required": compliance.escalate,
            "active_tax_code": context.active_tax_code,
            "latency_ms": round(total_latency, 1),
            "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
            "synthesis_tier": synthesis_result.synthesis_tier,
            "visualization_data": viz_data,
            "model_mode": model_mode,
            "plan_steps": [{"tool": s.tool_name, "description": s.description} for s in plan.steps],
        }}

        # ─── Step 1: Session & Context ────────────────────────────────────
        t0 = time.perf_counter()
        self._ensure_session(db, session_id=session_id, user_id=user_id)

        # Get current turn index
        turn_index_row = db.execute(
            sql_text(
                "SELECT COALESCE(MAX(turn_index), 0) FROM agent_turns "
                "WHERE session_id = :session_id"
            ),
            {"session_id": session_id},
        ).fetchone()
        turn_index = int(turn_index_row[0] or 0) + 1

        # Log user turn
        db.execute(
            sql_text("""
                INSERT INTO agent_turns (session_id, turn_index, role, message_text)
                VALUES (:session_id, :turn_index, 'user', :message_text)
            """),
            {"session_id": session_id, "turn_index": turn_index, "message_text": message},
        )

        # Build conversation context
        context = self._memory.build_context(session_id, turn_index, message)
        latency_breakdown["context"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 2: Intent Classification (Phase 4 Enhanced) ─────────────
        t0 = time.perf_counter()
        multi_intent_result = None
        intent_meta = {}

        # Try enhanced classifier first (semantic + multi-intent)
        if self._enhanced_intent is not None:
            try:
                multi_intent_result = self._enhanced_intent.classify(
                    message,
                    context_intents=context.active_intent_history,
                )
                intent = multi_intent_result.primary_intent
                intent_conf = multi_intent_result.primary_confidence
                intent_meta = {
                    "source": multi_intent_result.classification_source,
                    "is_multi_intent": multi_intent_result.is_multi_intent,
                    "secondary_intents": [
                        {"intent": si.intent, "score": si.score}
                        for si in multi_intent_result.secondary_intents
                    ],
                    "all_scores": multi_intent_result.all_scores,
                }
                # Use extracted entities to enrich context
                for ent in multi_intent_result.extracted_entities:
                    if ent["type"] == "tax_code" and not context.active_tax_code:
                        context.active_tax_code = ent["value"]
                    elif ent["type"] == "tax_period" and not context.active_tax_period:
                        context.active_tax_period = ent["value"]
            except Exception as exc:
                logger.warning("[Orchestrator] Enhanced intent failed: %s", exc)
                multi_intent_result = None

        # Fallback to legacy intent model
        if multi_intent_result is None:
            intent, intent_conf, intent_meta = self._intent_model.predict(message)
            if intent_conf < 0.45:
                rule_intent, rule_conf = self._rule_based_intent(message)
                if rule_conf > intent_conf:
                    intent, intent_conf = rule_intent, rule_conf
                    intent_meta["source"] = "keyword_rules"

        intent_conf = min(0.95, max(0.15, float(intent_conf)))
        latency_breakdown["intent"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 2.5: NL Query Fast Paths ─────────────────────────────────
        t0 = time.perf_counter()
        nl_results = {}
        try:
            from ml_engine.tax_agent_nl_query import NLQueryExecutor
            nl_executor = NLQueryExecutor()

            # Fast path: top_n_query
            if intent == "top_n_query":
                quantity = 10  # default
                if multi_intent_result and multi_intent_result.extracted_entities:
                    for ent in multi_intent_result.extracted_entities:
                        if ent.get("type") == "quantity":
                            quantity = min(50, max(1, int(ent["value"])))
                nl_results = nl_executor.execute_top_n(
                    db, n=quantity, sort_by="risk_score", mode=model_mode,
                )

            # Fast path: company_name_lookup
            elif intent == "company_name_lookup":
                company_name = ""
                if multi_intent_result and multi_intent_result.extracted_entities:
                    for ent in multi_intent_result.extracted_entities:
                        if ent.get("type") == "company_name":
                            company_name = ent["value"]
                if company_name:
                    nl_results = nl_executor.execute_company_name_search(
                        db, name=company_name,
                    )
                    # If single match found, set tax_code for pipeline
                    matches = nl_results.get("matches", [])
                    if len(matches) == 1:
                        context.active_tax_code = matches[0]["tax_code"]

            # CSV attachment → batch analysis
            if csv_attachment:
                intent = "batch_analysis"
                batch_result = nl_executor.execute_batch_inline(
                    db,
                    csv_content=csv_attachment["content"],
                    filename=csv_attachment["filename"],
                )
                nl_results["_batch_results"] = batch_result
        except Exception as exc:
            logger.warning("[Orchestrator] NL query fast-path error: %s", exc)
        latency_breakdown["nl_query"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 3: Planning (mode-aware) ─────────────────────────────────
        t0 = time.perf_counter()
        plan = self._planner.plan(
            query=message,
            intent=intent,
            intent_confidence=intent_conf,
            tax_code=context.active_tax_code,
            tax_period=context.active_tax_period,
            context_intents=context.active_intent_history,
            model_mode=model_mode,
        )
        latency_breakdown["planning"] = (time.perf_counter() - t0) * 1000.0

        # Get mode profile for tool filtering
        mode_profile = MODE_TOOL_PROFILES.get(model_mode, MODE_TOOL_PROFILES["full"])
        allowed_tools = mode_profile.get("required_tools")  # None = all
        allowed_optional = set(mode_profile.get("optional_tools", []))
        allowed_sub_agents = set(mode_profile.get("sub_agents", []))

        # ─── Step 4: Tool Execution (mode-filtered) ───────────────────────
        t0 = time.perf_counter()
        from ml_engine.tax_agent_tools import ToolCallRequest

        stages = plan.get_stages()
        all_tool_results: dict[str, dict[str, Any]] = {}

        # Merge NL query results into tool results
        if nl_results:
            for k, v in nl_results.items():
                all_tool_results[k] = v if isinstance(v, dict) else {"data": v, "status": "success"}

        for stage in stages:
            requests = []
            for step in stage:
                if step.optional and intent_conf <= 0.6:
                    continue
                # Mode filter: skip tools not in allowed list
                if allowed_tools is not None and step.tool_name not in allowed_tools and step.tool_name not in allowed_optional:
                    continue
                requests.append(
                    ToolCallRequest(
                        tool_name=step.tool_name,
                        inputs=step.tool_inputs,
                        request_id=f"req-{uuid.uuid4().hex[:8]}",
                    )
                )

            results = self._tool_executor.execute_parallel(requests, db=db)
            for result in results:
                all_tool_results[result.tool_name] = {
                    "status": result.status.value,
                    **(result.outputs or {}),
                    "_latency_ms": result.latency_ms,
                    "_error": result.error,
                }

        latency_breakdown["tools"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 4.5: Sub-Agent Dispatch (mode-filtered) ──────────────────
        t0 = time.perf_counter()
        sub_agent_analysis = {}

        try:
            # Legal Research Agent — runs if 'legal' in allowed_sub_agents
            ks_hits = all_tool_results.get("knowledge_search", {}).get("hits", [])
            if ks_hits and "legal" in allowed_sub_agents:
                legal_opinion = self._legal_agent.research(
                    query=message,
                    retrieval_results=ks_hits,
                    intent=intent,
                    tax_code=context.active_tax_code,
                )
                sub_agent_analysis["legal_research"] = {
                    "analysis": legal_opinion.analysis,
                    "conclusion": legal_opinion.conclusion,
                    "citation_chain": legal_opinion.citation_chain[:5],
                    "authority_score": legal_opinion.authority_score,
                    "confidence": legal_opinion.confidence,
                    "caveats": legal_opinion.caveats,
                    "applicable_laws": legal_opinion.applicable_laws[:5],
                }

            # Analytics Agent — runs when we have company data and 'analytics' in allowed
            if context.active_tax_code and plan.complexity.value in ("moderate", "complex", "investigation") and "analytics" in allowed_sub_agents:
                analytics_report = self._analytics_agent.analyze(
                    tax_code=context.active_tax_code,
                    tool_results=all_tool_results,
                    intent=intent,
                )
                sub_agent_analysis["analytics"] = {
                    "composite_risk_score": analytics_report.composite_risk_score,
                    "risk_level": analytics_report.risk_level.value,
                    "summary": analytics_report.summary,
                    "detailed_analysis": analytics_report.detailed_analysis,
                    "recommendations": analytics_report.recommendations,
                    "risk_trend": analytics_report.risk_trend,
                    "confidence": analytics_report.confidence,
                }

            # Investigation Agent — runs for complex/investigation queries if allowed
            if context.active_tax_code and plan.complexity.value in ("complex", "investigation") and "investigation" in allowed_sub_agents:
                if intent in ("osint_ownership", "invoice_risk", "vat_refund_risk"):
                    inv_report = self._investigation_agent.investigate(
                        tax_code=context.active_tax_code,
                        tool_results=all_tool_results,
                        intent=intent,
                    )
                    sub_agent_analysis["investigation"] = {
                        "suspicion_level": inv_report.suspicion_level.value,
                        "overall_score": inv_report.overall_score,
                        "executive_summary": inv_report.executive_summary,
                        "detailed_findings": inv_report.detailed_findings,
                        "patterns_count": len(inv_report.suspicious_patterns),
                        "escalation_level": inv_report.escalation_level,
                        "recommended_actions": inv_report.recommended_actions,
                        "confidence": inv_report.confidence,
                    }

        except Exception as exc:
            logger.warning("[Orchestrator] Sub-agent dispatch error: %s", exc)

        latency_breakdown["sub_agents"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 4.5: Multi-Agent Debate ────────────────────────────────
        debate_result_dict = None
        if len(sub_agent_analysis) >= 2:
            try:
                from ml_engine.tax_agent_debate import AgentDebateProtocol
                debate = AgentDebateProtocol()
                debate_result = debate.run_debate(sub_agent_analysis, all_tool_results)
                debate_result_dict = debate_result.to_dict()
            except Exception as exc:
                logger.warning("[Orchestrator] Debate error: %s", exc)

        # ─── Step 5: Synthesis (enriched with sub-agent analysis) ──────────
        t0 = time.perf_counter()

        # Merge sub-agent analysis into tool results for richer synthesis
        enriched_tool_results = dict(all_tool_results)
        for agent_name, analysis in sub_agent_analysis.items():
            enriched_tool_results[f"_sub_agent_{agent_name}"] = analysis

        synthesis_result = self._synthesizer.synthesize(
            query=message,
            intent=intent,
            tool_results=all_tool_results,
            reasoning_trace=plan.reasoning,
            tax_code=context.active_tax_code,
        )

        # Enrich answer with sub-agent insights
        answer = self._synthesizer.format_response_text(synthesis_result)
        answer = self._enrich_with_sub_agents(answer, sub_agent_analysis)

        latency_breakdown["synthesis"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 6: Compliance Gate ──────────────────────────────────────
        t0 = time.perf_counter()

        # Create assistant turn for logging
        turn_row = db.execute(
            sql_text("""
                INSERT INTO agent_turns
                (session_id, turn_index, role, message_text)
                VALUES (:session_id, :turn_index, 'assistant', '')
                RETURNING id
            """),
            {"session_id": session_id, "turn_index": turn_index + 1},
        ).fetchone()
        assistant_turn_id = int(turn_row[0]) if turn_row else 0

        retrieval_hits = len(
            all_tool_results.get("knowledge_search", {}).get("hits", [])
        )
        compliance = self._compliance_gate.evaluate(
            query=message,
            intent=intent,
            intent_confidence=intent_conf,
            retrieval_hits=retrieval_hits,
            response_text=answer,
            tool_results=all_tool_results,
            session_id=session_id,
            turn_id=assistant_turn_id,
        )

        # If compliance blocks, use abstain response
        if compliance.abstain:
            synthesis_result = self._synthesizer.synthesize(
                query=message,
                intent=intent,
                tool_results=all_tool_results,
                reasoning_trace=plan.reasoning,
                abstained=True,
            )
            answer = self._synthesizer.format_response_text(synthesis_result)

        latency_breakdown["compliance"] = (time.perf_counter() - t0) * 1000.0

        # ─── Step 7: Audit Trail & Memory ─────────────────────────────────
        t0 = time.perf_counter()

        # Build citations
        citations = []
        for ev in synthesis_result.evidence[:5]:
            if ev.source_type == "legal":
                citations.append({
                    "chunk_key": ev.metadata.get("chunk_key", ""),
                    "title": ev.title,
                    "score": round(ev.score, 4),
                    "citation_key": ev.citation_key,
                })

        # Update assistant turn
        db.execute(
            sql_text("""
                UPDATE agent_turns
                SET message_text = :message_text,
                    normalized_intent = :normalized_intent,
                    confidence = :confidence,
                    citations_json = CAST(:citations_json AS jsonb)
                WHERE id = :turn_id
            """),
            {
                "turn_id": assistant_turn_id,
                "message_text": answer,
                "normalized_intent": intent,
                "confidence": intent_conf,
                "citations_json": json.dumps(citations),
            },
        )

        # Log decision trace
        db.execute(
            sql_text("""
                INSERT INTO agent_decision_traces
                (session_id, turn_id, intent, selected_track, confidence,
                 abstained, escalation_required, evidence_json, answer_text)
                VALUES
                (:session_id, :turn_id, :intent, :selected_track, :confidence,
                 :abstained, :escalation_required, CAST(:evidence_json AS jsonb), :answer_text)
            """),
            {
                "session_id": session_id,
                "turn_id": assistant_turn_id,
                "intent": intent,
                "selected_track": plan.complexity.value,
                "confidence": synthesis_result.confidence,
                "abstained": compliance.abstain,
                "escalation_required": compliance.escalate,
                "evidence_json": json.dumps({
                    "plan": {
                        "complexity": plan.complexity.value,
                        "tools": [s.tool_name for s in plan.steps],
                        "reasoning": plan.reasoning,
                    },
                    "tool_results_summary": {
                        k: {"status": v.get("status")}
                        for k, v in all_tool_results.items()
                    },
                    "synthesis": {
                        "confidence": synthesis_result.confidence,
                        "evidence_count": len(synthesis_result.evidence),
                        "tier": synthesis_result.synthesis_tier,
                    },
                    "compliance": {
                        "decision": compliance.overall_decision.value,
                        "warnings": compliance.warnings,
                    },
                }),
                "answer_text": answer,
            },
        )

        # Log tool calls
        for tool_name, result in all_tool_results.items():
            db.execute(
                sql_text("""
                    INSERT INTO agent_tool_calls
                    (session_id, turn_id, tool_name, tool_input, tool_output, status, latency_ms)
                    VALUES (:session_id, :turn_id, :tool_name,
                            CAST(:tool_input AS jsonb), CAST(:tool_output AS jsonb),
                            :status, :latency_ms)
                """),
                {
                    "session_id": session_id,
                    "turn_id": assistant_turn_id,
                    "tool_name": tool_name,
                    "tool_input": json.dumps(
                        next(
                            (s.tool_inputs for s in plan.steps if s.tool_name == tool_name),
                            {},
                        )
                    ),
                    "tool_output": json.dumps(
                        {k: v for k, v in result.items() if not k.startswith("_")},
                        default=str,
                    ),
                    "status": result.get("status", "unknown"),
                    "latency_ms": result.get("_latency_ms"),
                },
            )

        # Persist entity memory
        self._memory.persist_entities(session_id, context.active_entities)

        db.commit()

        latency_breakdown["audit"] = (time.perf_counter() - t0) * 1000.0
        total_latency = (time.perf_counter() - t_total_start) * 1000.0

        # Log inference
        from ml_engine.model_registry import ModelRegistryService, AuditContext
        registry = ModelRegistryService(db)
        registry.log_inference(
            model_name="tax_agent_orchestrator",
            model_version="multi-agent-v2",
            entity_type="session",
            entity_id=session_id,
            input_features={
                "intent": intent,
                "intent_confidence": intent_conf,
                "query_len": len(message),
                "complexity": plan.complexity.value,
                "tools_count": len(all_tool_results),
            },
            outputs={
                "abstained": compliance.abstain,
                "escalation_required": compliance.escalate,
                "confidence": synthesis_result.confidence,
                "citations_count": len(citations),
                "tools_used": list(all_tool_results.keys()),
            },
            latency_ms=total_latency,
            ctx=AuditContext(request_id=f"orch-{uuid.uuid4().hex[:12]}"),
        )

        logger.info(
            "[Orchestrator] ✓ Processed query: intent=%s complexity=%s "
            "tools=%d confidence=%.2f latency=%.0fms",
            intent, plan.complexity.value,
            len(all_tool_results), synthesis_result.confidence,
            total_latency,
        )

        # Retrieval context for backward compatibility
        retrieval_context = all_tool_results.get("knowledge_search", {}).get("hits", [])

        return OrchestratorResponse(
            session_id=session_id,
            intent=intent,
            intent_confidence=round(intent_conf, 4),
            complexity=plan.complexity.value,
            reasoning_trace=plan.reasoning,
            tools_used=list(all_tool_results.keys()),
            answer=answer,
            summary=synthesis_result.summary,
            citations=citations,
            recommendations=synthesis_result.recommendations,
            confidence=synthesis_result.confidence,
            abstained=compliance.abstain,
            escalation_required=compliance.escalate,
            escalation_domain=compliance.escalation_domain.value,
            compliance_warnings=compliance.warnings,
            active_tax_code=context.active_tax_code,
            active_tax_period=context.active_tax_period,
            latency_ms=round(total_latency, 1),
            latency_breakdown={k: round(v, 1) for k, v in latency_breakdown.items()},
            tool_results={
                k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in all_tool_results.items()
            },
            synthesis_tier=synthesis_result.synthesis_tier,
            policy_traces=[
                {
                    "rule_key": t.rule_key,
                    "decision": t.decision.value,
                    "score": t.score,
                    "reason": t.reason,
                }
                for t in compliance.traces
            ],
            plan_steps=[
                {
                    "step_id": s.step_id,
                    "tool": s.tool_name,
                    "description": s.description,
                    "priority": s.priority,
                    "optional": s.optional,
                }
                for s in plan.steps
            ],
            visualization_data=(lambda vd: (vd.update({"agent_debate": debate_result_dict}) or vd) if debate_result_dict else vd)(
                self._build_visualization_data(
                    all_tool_results, sub_agent_analysis, plan, latency_breakdown,
                )
            ),
            model_mode=model_mode,
        )

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
        "vat_refund_risk": ["hoan thue", "vat", "ho so hoan", "refund", "đề nghị hoàn"],
        "invoice_risk": ["hoa don", "invoice", "xuat hoa don", "mua vao", "ban ra"],
        "delinquency": ["no dong", "cham nop", "delinquency", "qua han", "thu no"],
        "osint_ownership": ["offshore", "so huu", "ubo", "phoenix", "cong ty me"],
        "transfer_pricing": ["chuyen gia", "transfer pricing", "gia giao dich lien ket"],
        "audit_selection": ["thanh tra", "audit", "kiem tra", "xep hang ho so"],
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
