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
        ],
        "invoice_risk": [
            "knowledge_search", "invoice_risk_scan", "motif_detection",
        ],
        "delinquency": [
            "knowledge_search", "company_risk_lookup", "delinquency_check",
        ],
        "osint_ownership": [
            "knowledge_search", "ownership_analysis", "company_risk_lookup", "gnn_analysis",
        ],
        "transfer_pricing": [
            "knowledge_search", "company_risk_lookup", "invoice_risk_scan",
        ],
        "audit_selection": [
            "knowledge_search", "company_risk_lookup", "delinquency_check",
            "invoice_risk_scan",
        ],
        "top_n_query": [
            "knowledge_search",
        ],
        "company_name_lookup": [
            "knowledge_search", "company_risk_lookup",
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
            metadata={
                "intent_confidence": intent_confidence,
                "tax_code": tax_code,
                "tax_period": tax_period,
                "tools_selected": [t for t in tools],
                "planning_latency_ms": (time.perf_counter() - t0) * 1000.0,
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
            # Only retrieval for simple queries
            return ["knowledge_search"]

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
