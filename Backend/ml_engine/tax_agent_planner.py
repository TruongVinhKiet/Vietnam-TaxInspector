"""
tax_agent_planner.py – Intent & Planning Agent (Phase 2)
=========================================================
Chain-of-Thought task decomposition for tax intelligence queries.

Responsibilities:
    1. Classify query complexity (simple lookup vs deep investigation)
    2. Decompose complex queries into sub-tasks (execution DAG)
    3. Select which tools to invoke and in what order
    4. Generate execution plan with dependency tracking

Architecture:
    - Rule-based planner (deterministic, auditable)
    - Template-based CoT reasoning
    - Future: LLM-based planning when custom model is ready
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    """Classification of query complexity."""
    SIMPLE = "simple"           # Direct lookup, single tool
    MODERATE = "moderate"       # 2-3 tools, some reasoning
    COMPLEX = "complex"         # Multiple tools, cross-referencing
    INVESTIGATION = "investigation"  # Deep dive, full pipeline


class PlanStep(str, Enum):
    """Individual step types in an execution plan."""
    RETRIEVE_LEGAL = "retrieve_legal"
    LOOKUP_COMPANY = "lookup_company"
    CHECK_DELINQUENCY = "check_delinquency"
    SCAN_INVOICES = "scan_invoices"
    RUN_GNN = "run_gnn"
    DETECT_MOTIFS = "detect_motifs"
    ANALYZE_OWNERSHIP = "analyze_ownership"
    SYNTHESIZE = "synthesize"


@dataclass
class SubTask:
    """A single sub-task in the execution plan."""
    step_id: int
    step_type: PlanStep
    tool_name: str
    tool_inputs: dict[str, Any]
    description: str
    depends_on: list[int] = field(default_factory=list)  # step_ids this depends on
    priority: int = 5
    optional: bool = False
    timeout_ms: int = 10000
    max_retries: int = 1
    evidence_contract: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """
    A DAG of sub-tasks to execute.
    Steps with no dependencies can run in parallel.
    """
    plan_id: str
    query: str
    intent: str
    complexity: QueryComplexity
    reasoning: str                    # Chain-of-thought explanation
    steps: list[SubTask]
    estimated_latency_ms: float = 0.0
    budget_ms: int = 30000
    max_react_iterations: int = 2
    retry_policy: dict[str, Any] = field(default_factory=dict)
    evidence_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_stages(self) -> list[list[SubTask]]:
        """
        Convert the DAG into sequential stages for execution.
        Steps in the same stage can run in parallel.
        """
        if not self.steps:
            return []

        # Topological sort into stages
        completed: set[int] = set()
        stages: list[list[SubTask]] = []
        remaining = list(self.steps)

        max_iterations = len(remaining) + 1
        iteration = 0
        while remaining and iteration < max_iterations:
            iteration += 1
            current_stage = []
            still_remaining = []

            for step in remaining:
                deps_met = all(dep_id in completed for dep_id in step.depends_on)
                if deps_met:
                    current_stage.append(step)
                else:
                    still_remaining.append(step)

            if not current_stage:
                # Deadlock — add remaining as final stage
                stages.append(still_remaining)
                break

            stages.append(current_stage)
            completed.update(s.step_id for s in current_stage)
            remaining = still_remaining

        return stages


class TaxAgentPlanner:
    """
    Deterministic query planner with Chain-of-Thought reasoning.

    Flow:
    1. Classify query complexity
    2. Identify required tools based on intent + entities
    3. Build execution plan (DAG)
    4. Return plan with reasoning trace

    Future: Replace rules with custom LLM-based planning.
    """

    # Intent → default tools mapping
    INTENT_TOOL_MAP: dict[str, list[str]] = {
        "vat_refund_risk": [
            "knowledge_search", "company_risk_lookup", "invoice_risk_scan",
            "nlp_red_flag_scan", "vae_anomaly_scan",
        ],
        "invoice_risk": [
            "knowledge_search", "invoice_risk_scan", "motif_detection",
            "nlp_red_flag_scan", "vae_anomaly_scan",
        ],
        "delinquency": [
            "knowledge_search", "company_risk_lookup", "delinquency_check",
            "revenue_forecast", "temporal_delinquency_deep", "causal_uplift_recommend",
        ],
        "osint_ownership": [
            "knowledge_search", "ownership_analysis", "company_risk_lookup", "gnn_analysis",
            "entity_resolution_check", "hetero_gnn_risk",
        ],
        "transfer_pricing": [
            "knowledge_search", "company_risk_lookup", "invoice_risk_scan",
        ],
        "audit_selection": [
            "knowledge_search", "company_risk_lookup", "delinquency_check",
            "invoice_risk_scan", "nlp_red_flag_scan",
        ],
        "top_n_query": [
            "top_n_risky_companies",
        ],
        "company_name_lookup": [
            "company_name_search",
        ],
        "batch_analysis": [
            "knowledge_search",
        ],
        "general_tax_query": [
            "knowledge_search",
        ],
    }

    # Keywords that upgrade complexity
    INVESTIGATION_KEYWORDS = [
        "điều tra", "investigation", "deep dive", "phân tích sâu",
        "toàn diện", "comprehensive", "full analysis", "đánh giá tổng thể",
    ]
    COMPLEX_KEYWORDS = [
        "so sánh", "compare", "liên quan", "related", "kết hợp",
        "đồng thời", "cùng lúc", "nhiều", "multiple",
    ]

    def plan(
        self,
        query: str,
        intent: str,
        intent_confidence: float,
        *,
        tax_code: str | None = None,
        tax_period: str | None = None,
        context_intents: list[str] | None = None,
        model_mode: str = "full",
    ) -> ExecutionPlan:
        """
        Generate an execution plan for the given query.

        Args:
            query: User's query text
            intent: Classified intent
            intent_confidence: Intent classifier confidence
            tax_code: Extracted tax code (if any)
            tax_period: Extracted tax period (if any)
            context_intents: Previous intents in the session

        Returns:
            ExecutionPlan with steps and reasoning
        """
        t0 = time.perf_counter()

        # Step 1: Classify complexity
        complexity = self._classify_complexity(
            query, intent, intent_confidence,
            has_tax_code=bool(tax_code),
            context_intents=context_intents,
        )

        # Step 2: Select tools
        tools = self._select_tools(
            intent, complexity,
            has_tax_code=bool(tax_code),
        )

        # Step 3: Build execution steps
        steps = self._build_steps(
            tools, intent,
            tax_code=tax_code,
            tax_period=tax_period,
            query=query,
        )
        budget_ms = self._estimate_budget_ms(complexity, steps)
        retry_policy = self._build_retry_policy(complexity, intent_confidence)
        evidence_contract = self._build_plan_evidence_contract(intent, tools)

        # Step 4: Generate reasoning trace
        reasoning = self._generate_reasoning(
            query, intent, complexity, tools, tax_code,
        )

        plan = ExecutionPlan(
            plan_id=f"plan-{int(time.time() * 1000)}",
            query=query,
            intent=intent,
            complexity=complexity,
            reasoning=reasoning,
            steps=steps,
            estimated_latency_ms=self._estimate_latency(steps),
            budget_ms=budget_ms,
            max_react_iterations=retry_policy["max_react_iterations"],
            retry_policy=retry_policy,
            evidence_contract=evidence_contract,
            metadata={
                "intent_confidence": intent_confidence,
                "tax_code": tax_code,
                "tax_period": tax_period,
                "tools_selected": [t for t in tools],
                "planning_latency_ms": (time.perf_counter() - t0) * 1000.0,
                "budget_ms": budget_ms,
                "retry_policy": retry_policy,
                "evidence_contract": evidence_contract,
            },
        )

        logger.info(
            "[Planner] Plan generated: complexity=%s, steps=%d, tools=%s",
            complexity.value, len(steps), [s.tool_name for s in steps],
        )

        return plan

    def _classify_complexity(
        self,
        query: str,
        intent: str,
        intent_confidence: float,
        *,
        has_tax_code: bool,
        context_intents: list[str] | None,
    ) -> QueryComplexity:
        """Classify query complexity."""
        query_lower = query.lower()

        # Investigation keywords → INVESTIGATION
        if any(kw in query_lower for kw in self.INVESTIGATION_KEYWORDS):
            return QueryComplexity.INVESTIGATION

        # Complex keywords + tax code → COMPLEX
        if any(kw in query_lower for kw in self.COMPLEX_KEYWORDS) and has_tax_code:
            return QueryComplexity.COMPLEX

        # Tax code present → at least MODERATE
        if has_tax_code:
            # OSINT/audit queries with tax code are always complex
            if intent in ("osint_ownership", "audit_selection"):
                return QueryComplexity.COMPLEX
            return QueryComplexity.MODERATE

        # Low confidence or general query → SIMPLE
        if intent == "general_tax_query" or intent_confidence < 0.4:
            return QueryComplexity.SIMPLE

        # Multi-intent context → MODERATE
        if context_intents and len(set(context_intents)) >= 2:
            return QueryComplexity.MODERATE

        return QueryComplexity.SIMPLE

    def _select_tools(
        self,
        intent: str,
        complexity: QueryComplexity,
        *,
        has_tax_code: bool,
    ) -> list[str]:
        """Select tools based on intent and complexity."""
        # Get default tools for intent
        default_tools = list(self.INTENT_TOOL_MAP.get(intent, ["knowledge_search"]))

        if complexity == QueryComplexity.SIMPLE:
            # Preserve direct-data tools for simple lookup intents.
            return default_tools[:1]

        if complexity == QueryComplexity.MODERATE:
            # Primary tools only
            if has_tax_code:
                return default_tools[:3]
            return default_tools[:2]

        if complexity == QueryComplexity.COMPLEX:
            # All default tools
            return default_tools

        if complexity == QueryComplexity.INVESTIGATION:
            # Full toolkit
            investigation_tools = list(default_tools)
            for extra in ["gnn_analysis", "motif_detection", "ownership_analysis"]:
                if extra not in investigation_tools:
                    investigation_tools.append(extra)
            return investigation_tools

        return default_tools

    def _build_steps(
        self,
        tools: list[str],
        intent: str,
        *,
        tax_code: str | None,
        tax_period: str | None,
        query: str,
    ) -> list[SubTask]:
        """Build execution steps with dependency tracking."""
        steps: list[SubTask] = []
        step_id = 0

        # Stage 1: Knowledge retrieval (always first, no dependencies)
        if "knowledge_search" in tools:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.RETRIEVE_LEGAL,
                tool_name="knowledge_search",
                tool_inputs={
                    "query": query,
                    "intent": intent,
                    "top_k": 5,
                },
                description="Tìm kiếm cơ sở pháp lý liên quan",
                priority=1,
            ))
            retrieval_step_id = step_id
            step_id += 1

        # Stage 2: Analytics tools (parallel, depend on nothing or retrieval)
        analytics_step_ids = []

        if "top_n_risky_companies" in tools:
            n = 10
            quantity_match = re.search(r"\b(?:top\s*)?(\d{1,2})\b", query.lower())
            if quantity_match:
                try:
                    n = min(50, max(1, int(quantity_match.group(1))))
                except Exception:
                    n = 10
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.LOOKUP_COMPANY,
                tool_name="top_n_risky_companies",
                tool_inputs={"n": n, "sort_by": "risk_score"},
                description=f"Tra cứu top {n} doanh nghiệp rủi ro cao nhất",
                priority=1,
            ))
            analytics_step_ids.append(step_id)
            step_id += 1

        if "company_name_search" in tools:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.LOOKUP_COMPANY,
                tool_name="company_name_search",
                tool_inputs={"name": query},
                description="Tra cứu doanh nghiệp theo tên",
                priority=1,
            ))
            analytics_step_ids.append(step_id)
            step_id += 1

        if "company_risk_lookup" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.LOOKUP_COMPANY,
                tool_name="company_risk_lookup",
                tool_inputs={"tax_code": tax_code},
                description=f"Tra cứu hồ sơ rủi ro DN {tax_code}",
                priority=2,
            ))
            analytics_step_ids.append(step_id)
            step_id += 1

        if "delinquency_check" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.CHECK_DELINQUENCY,
                tool_name="delinquency_check",
                tool_inputs={"tax_code": tax_code},
                description=f"Kiểm tra rủi ro nợ đọng DN {tax_code}",
                priority=3,
            ))
            analytics_step_ids.append(step_id)
            step_id += 1

        if "invoice_risk_scan" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.SCAN_INVOICES,
                tool_name="invoice_risk_scan",
                tool_inputs={"tax_code": tax_code},
                description=f"Quét rủi ro hóa đơn DN {tax_code}",
                priority=3,
            ))
            analytics_step_ids.append(step_id)
            step_id += 1

        # Stage 3: Investigation tools (may depend on analytics)
        if "gnn_analysis" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.RUN_GNN,
                tool_name="gnn_analysis",
                tool_inputs={"tax_code": tax_code},
                description=f"Phân tích GNN đồ thị giao dịch DN {tax_code}",
                depends_on=[],  # Can run parallel with analytics
                priority=4,
            ))
            step_id += 1

        if "motif_detection" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.DETECT_MOTIFS,
                tool_name="motif_detection",
                tool_inputs={"tax_code": tax_code},
                description=f"Phát hiện mẫu giao dịch đáng ngờ DN {tax_code}",
                depends_on=[],
                priority=5,
                optional=True,
            ))
            step_id += 1

        if "ownership_analysis" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.ANALYZE_OWNERSHIP,
                tool_name="ownership_analysis",
                tool_inputs={"tax_code": tax_code},
                description=f"Phân tích cấu trúc sở hữu DN {tax_code}",
                depends_on=[],
                priority=5,
            ))
            step_id += 1

        # ═══ NEW ML TOOLS ═══

        if "nlp_red_flag_scan" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.SCAN_INVOICES,
                tool_name="nlp_red_flag_scan",
                tool_inputs={"tax_code": tax_code},
                description=f"Phân tích NLP mô tả hóa đơn DN {tax_code}",
                depends_on=[],
                priority=4,
            ))
            step_id += 1

        if "revenue_forecast" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.CHECK_DELINQUENCY,
                tool_name="revenue_forecast",
                tool_inputs={"tax_code": tax_code},
                description=f"Dự báo doanh thu quý tới DN {tax_code}",
                depends_on=[],
                priority=4,
            ))
            step_id += 1

        if "entity_resolution_check" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.ANALYZE_OWNERSHIP,
                tool_name="entity_resolution_check",
                tool_inputs={"tax_code": tax_code},
                description=f"Kiểm tra trùng lặp thực thể DN {tax_code}",
                depends_on=[],
                priority=5,
                optional=True,
            ))
            step_id += 1

        if "temporal_delinquency_deep" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.CHECK_DELINQUENCY,
                tool_name="temporal_delinquency_deep",
                tool_inputs={"tax_code": tax_code},
                description=f"Deep temporal delinquency model for {tax_code}",
                depends_on=[],
                priority=4,
                optional=True,
            ))
            step_id += 1

        if "hetero_gnn_risk" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.RUN_GNN,
                tool_name="hetero_gnn_risk",
                tool_inputs={"tax_code": tax_code},
                description=f"Heterogeneous graph risk analysis for {tax_code}",
                depends_on=[],
                priority=4,
                optional=True,
            ))
            step_id += 1

        if "vae_anomaly_scan" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.SCAN_INVOICES,
                tool_name="vae_anomaly_scan",
                tool_inputs={"tax_code": tax_code},
                description=f"VAE anomaly scan for invoices of {tax_code}",
                depends_on=[],
                priority=4,
                optional=True,
            ))
            step_id += 1

        if "causal_uplift_recommend" in tools and tax_code:
            steps.append(SubTask(
                step_id=step_id,
                step_type=PlanStep.CHECK_DELINQUENCY,
                tool_name="causal_uplift_recommend",
                tool_inputs={"tax_code": tax_code},
                description=f"Causal next-best-action uplift estimate for {tax_code}",
                depends_on=[],
                priority=5,
                optional=True,
            ))
            step_id += 1

        for step in steps:
            step.timeout_ms = self._tool_timeout_ms(step.tool_name)
            step.max_retries = self._tool_max_retries(step.tool_name)
            step.evidence_contract = self._tool_evidence_contract(step.tool_name)

        return steps

    def _generate_reasoning(
        self,
        query: str,
        intent: str,
        complexity: QueryComplexity,
        tools: list[str],
        tax_code: str | None,
    ) -> str:
        """
        Generate Chain-of-Thought reasoning trace.
        This is a deterministic, rule-based version.
        Future: Replace with LLM-generated reasoning.
        """
        parts = []

        # Step 1: Intent analysis
        parts.append(
            f"1. PHÂN TÍCH Ý ĐỊNH: Câu hỏi thuộc loại '{intent}' "
            f"(độ phức tạp: {complexity.value})."
        )

        # Step 2: Entity identification
        if tax_code:
            parts.append(
                f"2. NHẬN DIỆN THỰC THỂ: MST '{tax_code}' được đề cập → "
                f"cần tra cứu dữ liệu cụ thể của doanh nghiệp."
            )
        else:
            parts.append(
                "2. NHẬN DIỆN THỰC THỂ: Không có MST cụ thể → "
                "tập trung vào tra cứu pháp lý chung."
            )

        # Step 3: Tool selection reasoning
        tool_reasons = {
            "knowledge_search": "tra cứu cơ sở pháp lý",
            "company_risk_lookup": "đánh giá rủi ro tổng thể",
            "delinquency_check": "dự báo khả năng vi phạm thời hạn",
            "invoice_risk_scan": "phân tích bất thường hóa đơn",
            "gnn_analysis": "phân tích đồ thị giao dịch bằng AI",
            "motif_detection": "phát hiện mẫu gian lận",
            "ownership_analysis": "phân tích cấu trúc sở hữu/UBO",
            "nlp_red_flag_scan": "phân tích NLP mô tả hóa đơn phát hiện gian lận",
            "revenue_forecast": "dự báo doanh thu quý tới bằng ML",
            "entity_resolution_check": "phát hiện doanh nghiệp trùng lặp/phượng hoàng",
            "ocr_document_process": "trích xuất thông tin từ ảnh/PDF hóa đơn",
        }
        reasons = [
            f"{tool_reasons.get(t, t)}"
            for t in tools
        ]
        parts.append(
            f"3. LỰA CHỌN CÔNG CỤ: Cần {len(tools)} công cụ: "
            f"{', '.join(reasons)}."
        )

        # Step 4: Execution strategy
        if complexity in (QueryComplexity.COMPLEX, QueryComplexity.INVESTIGATION):
            parts.append(
                "4. CHIẾN LƯỢC: Thực thi song song các công cụ phân tích, "
                "sau đó tổng hợp kết quả với cross-reference."
            )
        elif complexity == QueryComplexity.MODERATE:
            parts.append(
                "4. CHIẾN LƯỢC: Tra cứu pháp lý trước, "
                "sau đó bổ sung dữ liệu phân tích."
            )
        else:
            parts.append(
                "4. CHIẾN LƯỢC: Tra cứu tri thức chuyên môn, "
                "trả lời dựa trên citations."
            )

        return "\n".join(parts)

    def _estimate_latency(self, steps: list[SubTask]) -> float:
        """Estimate total latency based on steps."""
        # Rough estimates per tool (ms)
        latency_estimates = {
            "knowledge_search": 500,
            "company_risk_lookup": 100,
            "delinquency_check": 300,
            "invoice_risk_scan": 200,
            "gnn_analysis": 400,
            "motif_detection": 800,
            "ownership_analysis": 300,
            "nlp_red_flag_scan": 500,
            "revenue_forecast": 400,
            "entity_resolution_check": 600,
            "ocr_document_process": 2000,
            "temporal_delinquency_deep": 900,
            "hetero_gnn_risk": 1200,
            "vae_anomaly_scan": 900,
            "causal_uplift_recommend": 700,
        }

        # Simple estimate: sum of sequential stages
        # (parallel steps counted once)
        plan = ExecutionPlan(
            plan_id="", query="", intent="",
            complexity=QueryComplexity.SIMPLE,
            reasoning="", steps=steps,
        )
        stages = plan.get_stages()
        total = 0.0
        for stage in stages:
            stage_max = max(
                latency_estimates.get(s.tool_name, 200)
                for s in stage
            ) if stage else 0.0
            total += stage_max

        return total

    def _estimate_budget_ms(self, complexity: QueryComplexity, steps: list[SubTask]) -> int:
        base = {
            QueryComplexity.SIMPLE: 12000,
            QueryComplexity.MODERATE: 25000,
            QueryComplexity.COMPLEX: 45000,
            QueryComplexity.INVESTIGATION: 70000,
        }[complexity]
        return max(base, int(self._estimate_latency(steps) * 1.8))

    def _build_retry_policy(self, complexity: QueryComplexity, intent_confidence: float) -> dict[str, Any]:
        max_iterations = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MODERATE: 1,
            QueryComplexity.COMPLEX: 2,
            QueryComplexity.INVESTIGATION: 3,
        }[complexity]
        if intent_confidence < 0.55:
            max_iterations = min(3, max_iterations + 1)
        return {
            "max_react_iterations": max_iterations,
            "retry_missing_required": True,
            "retry_failed_optional": complexity in (QueryComplexity.COMPLEX, QueryComplexity.INVESTIGATION),
            "escalate_on_contradiction": True,
            "stop_when_budget_exhausted": True,
        }

    def _build_plan_evidence_contract(self, intent: str, tools: list[str]) -> dict[str, Any]:
        required_tools = [
            t for t in tools
            if t in {
                "knowledge_search",
                "company_risk_lookup",
                "invoice_risk_scan",
                "delinquency_check",
                "ownership_analysis",
            }
        ]
        return {
            "intent": intent,
            "required_tools": required_tools,
            "min_legal_hits": 1 if "knowledge_search" in tools else 0,
            "require_citation_spans": "knowledge_search" in tools,
            "require_numeric_scores": any(t in tools for t in [
                "company_risk_lookup",
                "invoice_risk_scan",
                "delinquency_check",
                "gnn_analysis",
                "hetero_gnn_risk",
                "vae_anomaly_scan",
            ]),
        }

    def _tool_timeout_ms(self, tool_name: str) -> int:
        return {
            "knowledge_search": 15000,
            "company_risk_lookup": 5000,
            "delinquency_check": 10000,
            "invoice_risk_scan": 10000,
            "gnn_analysis": 15000,
            "motif_detection": 20000,
            "ownership_analysis": 15000,
            "temporal_delinquency_deep": 15000,
            "hetero_gnn_risk": 15000,
            "vae_anomaly_scan": 15000,
            "causal_uplift_recommend": 10000,
            "nlp_red_flag_scan": 15000,
            "revenue_forecast": 10000,
            "entity_resolution_check": 15000,
        }.get(tool_name, 10000)

    def _tool_max_retries(self, tool_name: str) -> int:
        if tool_name in {"knowledge_search", "company_risk_lookup"}:
            return 2
        return 1

    def _tool_evidence_contract(self, tool_name: str) -> dict[str, Any]:
        contracts = {
            "knowledge_search": {
                "required_fields": ["hits", "retrieval_tier", "query_embedding_hash"],
                "min_hits": 1,
                "quality_fields": ["score", "components", "citation_spans"],
            },
            "company_risk_lookup": {"required_fields": ["status", "risk_score", "risk_level"]},
            "delinquency_check": {"required_fields": ["status", "prob_30d", "prob_60d", "prob_90d"]},
            "invoice_risk_scan": {"required_fields": ["status", "total_invoices", "risk_ratio"]},
            "gnn_analysis": {"required_fields": ["status", "gnn_outputs"]},
            "hetero_gnn_risk": {"required_fields": ["status", "fraud_probability"]},
            "vae_anomaly_scan": {"required_fields": ["status", "anomaly_count", "anomaly_ratio"]},
            "ownership_analysis": {"required_fields": ["status", "summary"]},
            "motif_detection": {"required_fields": ["status", "summary"]},
            "causal_uplift_recommend": {"required_fields": ["status", "recommended_action"]},
        }
        return contracts.get(tool_name, {"required_fields": ["status"]})
