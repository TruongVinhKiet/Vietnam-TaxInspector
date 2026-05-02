"""
tax_agent_react.py – ReAct Reasoning Loop (Self-Reflection Engine)
====================================================================
Implements the ReAct (Reason + Act) pattern for the orchestrator.

After the initial tool execution, the ReAct engine evaluates the quality
of the results and decides whether to:
    1. Accept the results and proceed to synthesis
    2. Retry with additional/different tools
    3. Trigger deeper investigation via sub-agents
    4. Flag contradictions for transparent reporting

The loop is bounded (max_iterations=3) to prevent infinite loops while
still allowing meaningful self-correction.

Architecture:
    Tool Results
         ↓
    [ReActEngine.reflect()]
         ↓
    Reflection = {observations, contradictions, actions}
         ↓
    if actions → execute additional tools → re-reflect
    else → proceed to synthesis

Design Decisions:
    - Deterministic: Rule-based reflection, no LLM calls, fully auditable
    - Bounded: Hard cap at MAX_ITERATIONS with diminishing returns logic
    - Observable: Every reflection step is logged as an SSE event
    - Non-destructive: Original results are preserved; retries add to them

Reference:
    Yao et al., "ReAct: Synergizing Reasoning and Acting in LLMs", ICLR 2023
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════

MAX_ITERATIONS = 3                # Hard cap on reflection loops
CONTRADICTION_THRESHOLD = 30.0    # Score gap (0-100) to flag contradiction
LOW_CONFIDENCE_THRESHOLD = 0.4    # Below this → suggest retry
MISSING_DATA_KEYWORDS = [         # Tool status values that indicate missing data
    "not_found", "no_data", "error", "timeout", "empty", "skipped",
]


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

class ReflectionType(str, Enum):
    """Types of observations the engine can make."""
    CONTRADICTION = "contradiction"         # Two tools disagree significantly
    LOW_CONFIDENCE = "low_confidence"       # Overall confidence is too low
    MISSING_DATA = "missing_data"           # A critical tool returned no data
    ANOMALY_DETECTED = "anomaly_detected"   # Unusual pattern found
    INCOMPLETE_PLAN = "incomplete_plan"     # Not enough tools were run
    CONSISTENT = "consistent"               # Everything looks good


class ReflectionAction(str, Enum):
    """Actions the engine can recommend."""
    PROCEED = "proceed"                     # Accept results, go to synthesis
    RETRY_TOOL = "retry_tool"              # Re-run a specific tool with different params
    ADD_TOOL = "add_tool"                  # Run an additional tool not in the original plan
    TRIGGER_INVESTIGATION = "trigger_investigation"  # Escalate to investigation agent
    SUGGEST_FOLLOWUP = "suggest_followup"  # Suggest a follow-up question to the user


@dataclass
class Observation:
    """A single observation about the tool results."""
    observation_type: ReflectionType
    severity: str               # "info", "warning", "critical"
    detail: str                 # Human-readable description (Vietnamese)
    source_tools: list[str]     # Which tools this observation concerns
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.observation_type.value,
            "severity": self.severity,
            "detail": self.detail,
            "source_tools": self.source_tools,
        }


@dataclass
class ReflectionAction_:
    """A recommended action based on observations."""
    action: ReflectionAction
    tool_name: str | None = None    # For retry/add actions
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "tool": self.tool_name,
            "params": self.params,
            "reason": self.reason,
        }


@dataclass
class ReflectionResult:
    """Complete result of one reflection iteration."""
    iteration: int
    observations: list[Observation]
    actions: list[ReflectionAction_]
    should_retry: bool
    summary: str                    # Vietnamese narrative

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "observations": [o.to_dict() for o in self.observations],
            "actions": [a.to_dict() for a in self.actions],
            "should_retry": self.should_retry,
            "summary": self.summary,
        }


# ════════════════════════════════════════════════════════════════
#  Contradiction Detectors
# ════════════════════════════════════════════════════════════════

def _check_risk_score_contradiction(tool_results: dict) -> list[Observation]:
    """Check if risk scores from different models contradict each other."""
    observations = []

    # Collect all risk-like scores
    scores: dict[str, float] = {}

    # XGBoost risk score (0-100)
    risk_lookup = tool_results.get("company_risk_lookup", {})
    if risk_lookup.get("status") in ("found", "success", "analyzed"):
        scores["XGBoost"] = float(risk_lookup.get("risk_score", 0))

    # GNN fraud probability (0-1 → 0-100)
    gnn = tool_results.get("gnn_analysis", {})
    if gnn.get("status") in ("success", "analyzed"):
        scores["GNN"] = float(gnn.get("fraud_probability", 0)) * 100

    # HeteroGNN fraud probability
    hgnn = tool_results.get("hetero_gnn_risk", {})
    if hgnn.get("status") in ("success", "analyzed"):
        scores["HeteroGNN"] = float(hgnn.get("fraud_probability", 0)) * 100

    # VAE anomaly score
    vae = tool_results.get("vae_anomaly_scan", {})
    if vae.get("status") in ("success", "analyzed"):
        anomaly_ratio = float(vae.get("anomaly_ratio", 0)) * 100
        scores["VAE"] = anomaly_ratio

    # Pairwise comparison
    score_items = list(scores.items())
    for i in range(len(score_items)):
        for j in range(i + 1, len(score_items)):
            name_a, val_a = score_items[i]
            name_b, val_b = score_items[j]
            gap = abs(val_a - val_b)

            if gap > CONTRADICTION_THRESHOLD:
                severity = "critical" if gap > 50 else "warning"
                higher = name_a if val_a > val_b else name_b
                lower = name_b if val_a > val_b else name_a
                observations.append(Observation(
                    observation_type=ReflectionType.CONTRADICTION,
                    severity=severity,
                    detail=(
                        f"Mâu thuẫn: {higher} báo rủi ro cao "
                        f"({max(val_a, val_b):.0f}%) nhưng {lower} "
                        f"báo thấp ({min(val_a, val_b):.0f}%). "
                        f"Chênh lệch: {gap:.0f} điểm."
                    ),
                    source_tools=[name_a, name_b],
                    data={"gap": gap, "scores": {name_a: val_a, name_b: val_b}},
                ))

    return observations


def _check_missing_data(
    tool_results: dict,
    planned_tools: list[str],
    evidence_contracts: dict[str, dict[str, Any]] | None = None,
) -> list[Observation]:
    """Check if any planned tools returned missing/error data."""
    observations = []
    evidence_contracts = evidence_contracts or {}

    for tool_name in planned_tools:
        result = tool_results.get(tool_name, {})
        status = str(result.get("status", "")).lower()
        contract = evidence_contracts.get(tool_name, {}) or {}

        if tool_name == "knowledge_search" and result and not result.get("hits"):
            observations.append(Observation(
                observation_type=ReflectionType.MISSING_DATA,
                severity="warning",
                detail="knowledge_search không trả về citation/hit phù hợp. Cần retry hoặc mở rộng truy vấn.",
                source_tools=[tool_name],
                data={"status": status, "hits": 0, "reason": "empty_hits"},
            ))
            continue

        if status in MISSING_DATA_KEYWORDS or not result:
            observations.append(Observation(
                observation_type=ReflectionType.MISSING_DATA,
                severity="warning",
                detail=f"Tool '{tool_name}' không trả kết quả hợp lệ (status: {status}). Cần xem xét retry hoặc bỏ qua.",
                source_tools=[tool_name],
                data={"status": status},
            ))
            continue

        missing_fields = [
            field_name
            for field_name in contract.get("required_fields", [])
            if result.get(field_name) in (None, "", [], {})
        ]
        min_hits = int(contract.get("min_hits", 0) or 0)
        if tool_name == "knowledge_search" and min_hits and len(result.get("hits", []) or []) < min_hits:
            missing_fields.append("hits")
        if missing_fields:
            observations.append(Observation(
                observation_type=ReflectionType.MISSING_DATA,
                severity="warning",
                detail=(
                    f"Tool '{tool_name}' khong dat evidence contract "
                    f"(missing: {', '.join(sorted(set(missing_fields)))})"
                ),
                source_tools=[tool_name],
                data={"status": status, "missing_fields": sorted(set(missing_fields))},
            ))

    return observations


def _check_anomaly_signals(tool_results: dict) -> list[Observation]:
    """Check if anomaly detection tools found concerning patterns."""
    observations = []

    # VAE anomalies
    vae = tool_results.get("vae_anomaly_scan", {})
    if vae.get("status") in ("success", "analyzed"):
        anomaly_count = int(vae.get("anomaly_count", 0))
        if anomaly_count >= 5:
            observations.append(Observation(
                observation_type=ReflectionType.ANOMALY_DETECTED,
                severity="critical" if anomaly_count >= 10 else "warning",
                detail=f"VAE phát hiện {anomaly_count} hóa đơn bất thường. Cần kích hoạt điều tra chuyên sâu.",
                source_tools=["vae_anomaly_scan"],
                data={"anomaly_count": anomaly_count},
            ))

    # Motif detection
    motif = tool_results.get("motif_detection", {})
    if motif.get("status") in ("success", "analyzed"):
        summary = motif.get("summary", {})
        total_motifs = sum(
            summary.get(k, 0)
            for k in ("carousel_count", "shell_count", "layering_count")
        )
        if total_motifs > 0:
            observations.append(Observation(
                observation_type=ReflectionType.ANOMALY_DETECTED,
                severity="critical",
                detail=f"Phát hiện {total_motifs} mẫu giao dịch đáng ngờ (carousel/shell/layering). Đề xuất mở rộng điều tra.",
                source_tools=["motif_detection"],
                data={"total_motifs": total_motifs},
            ))

    return observations


def _check_completeness(
    tool_results: dict,
    intent: str,
) -> list[Observation]:
    """Check if enough tools were run given the user's intent."""
    observations = []

    # Map intents to expected essential tools
    essential_tools: dict[str, list[str]] = {
        "vat_refund_risk": ["company_risk_lookup", "invoice_risk_scan", "vae_anomaly_scan"],
        "invoice_risk": ["company_risk_lookup", "invoice_risk_scan"],
        "delinquency": ["company_risk_lookup", "delinquency_check"],
        "osint_ownership": ["company_risk_lookup", "ownership_analysis"],
        "general_risk": ["company_risk_lookup"],
    }

    expected = essential_tools.get(intent, [])
    run_tools = set(tool_results.keys())
    missing = [t for t in expected if t not in run_tools]

    if missing:
        observations.append(Observation(
            observation_type=ReflectionType.INCOMPLETE_PLAN,
            severity="warning",
            detail=f"Thiếu {len(missing)} tool quan trọng cho intent '{intent}': {', '.join(missing)}.",
            source_tools=missing,
        ))

    return observations


# ════════════════════════════════════════════════════════════════
#  Action Generator
# ════════════════════════════════════════════════════════════════

def _generate_actions(
    observations: list[Observation],
    tool_results: dict,
    iteration: int,
) -> list[ReflectionAction_]:
    """Generate recommended actions based on observations."""
    actions: list[ReflectionAction_] = []

    # Diminishing returns: fewer retries in later iterations
    max_new_actions = max(1, 3 - iteration)

    for obs in observations:
        if len(actions) >= max_new_actions:
            break

        if obs.observation_type == ReflectionType.MISSING_DATA:
            # Retry the missing tool
            for tool_name in obs.source_tools:
                result = tool_results.get(tool_name, {})
                status = str(result.get("status", "")).lower()
                no_hits = tool_name == "knowledge_search" and not result.get("hits")
                contract_missing = bool(obs.data.get("missing_fields"))
                if tool_name not in tool_results or status in MISSING_DATA_KEYWORDS or no_hits or contract_missing:
                    params = {}
                    if tool_name == "knowledge_search":
                        params = {"intent": "general_tax_query", "top_k": 10}
                    actions.append(ReflectionAction_(
                        action=ReflectionAction.RETRY_TOOL,
                        tool_name=tool_name,
                        params=params,
                        reason=f"Tool '{tool_name}' trả về dữ liệu không hợp lệ, thử lại.",
                    ))

        elif obs.observation_type == ReflectionType.CONTRADICTION:
            if obs.severity == "critical":
                # Trigger investigation for critical contradictions
                actions.append(ReflectionAction_(
                    action=ReflectionAction.TRIGGER_INVESTIGATION,
                    reason=obs.detail,
                ))
            else:
                # Add complementary tool for moderate contradictions
                if "vae_anomaly_scan" not in tool_results:
                    actions.append(ReflectionAction_(
                        action=ReflectionAction.ADD_TOOL,
                        tool_name="vae_anomaly_scan",
                        reason="Thêm VAE scan để kiểm tra chéo giữa các model mâu thuẫn.",
                    ))

        elif obs.observation_type == ReflectionType.ANOMALY_DETECTED:
            if obs.severity == "critical":
                actions.append(ReflectionAction_(
                    action=ReflectionAction.TRIGGER_INVESTIGATION,
                    reason=obs.detail,
                ))

        elif obs.observation_type == ReflectionType.INCOMPLETE_PLAN:
            for tool_name in obs.source_tools[:max_new_actions]:
                actions.append(ReflectionAction_(
                    action=ReflectionAction.ADD_TOOL,
                    tool_name=tool_name,
                    reason=f"Bổ sung tool '{tool_name}' cho phân tích toàn diện.",
                ))

    # Default: proceed if no actions generated
    if not actions:
        actions.append(ReflectionAction_(
            action=ReflectionAction.PROCEED,
            reason="Kết quả nhất quán, đủ dữ liệu để tổng hợp.",
        ))

    return actions


# ════════════════════════════════════════════════════════════════
#  Main ReAct Engine
# ════════════════════════════════════════════════════════════════

class ReActEngine:
    """
    ReAct-style reasoning loop for the TaxInspector orchestrator.

    Evaluates tool results, detects problems, and recommends corrective
    actions. Designed to be called iteratively by the orchestrator.

    Usage:
        react = ReActEngine()
        reflection = react.reflect(
            tool_results=tool_results,
            planned_tools=["company_risk_lookup", "gnn_analysis"],
            intent="vat_refund_risk",
            iteration=0,
        )
        if reflection.should_retry:
            # Execute recommended tools and re-reflect
            ...
        else:
            # Proceed to synthesis
            ...
    """

    def reflect(
        self,
        tool_results: dict[str, Any],
        planned_tools: list[str],
        intent: str,
        iteration: int = 0,
        sub_agent_analysis: dict[str, Any] | None = None,
        evidence_contracts: dict[str, dict[str, Any]] | None = None,
    ) -> ReflectionResult:
        """
        Perform one iteration of reflection on tool results.

        Args:
            tool_results: Dict of tool_name → result_dict from tool execution.
            planned_tools: List of tool names that were planned to execute.
            intent: The classified user intent.
            iteration: Current iteration number (0-indexed).
            sub_agent_analysis: Optional sub-agent results for cross-checking.

        Returns:
            ReflectionResult with observations, actions, and retry decision.
        """
        t0 = time.perf_counter()
        observations: list[Observation] = []

        # Guard: max iterations
        if iteration >= MAX_ITERATIONS:
            return ReflectionResult(
                iteration=iteration,
                observations=[],
                actions=[ReflectionAction_(
                    action=ReflectionAction.PROCEED,
                    reason=f"Đã đạt giới hạn {MAX_ITERATIONS} vòng suy luận. Tiến hành tổng hợp.",
                )],
                should_retry=False,
                summary=f"Đã suy luận {MAX_ITERATIONS} vòng. Dừng lại để tổng hợp kết quả.",
            )

        # 1. Check for risk score contradictions
        observations.extend(_check_risk_score_contradiction(tool_results))

        # 2. Check for missing/failed data
        observations.extend(_check_missing_data(tool_results, planned_tools, evidence_contracts))

        # 3. Check for anomaly signals
        observations.extend(_check_anomaly_signals(tool_results))

        # 4. Check plan completeness
        observations.extend(_check_completeness(tool_results, intent))

        # 5. If no issues found, mark as consistent
        if not observations:
            observations.append(Observation(
                observation_type=ReflectionType.CONSISTENT,
                severity="info",
                detail="Tất cả kết quả nhất quán. Không phát hiện vấn đề.",
                source_tools=list(tool_results.keys()),
            ))

        # 6. Generate actions
        actions = _generate_actions(observations, tool_results, iteration)

        # 7. Determine if retry is needed
        should_retry = any(
            a.action in (ReflectionAction.RETRY_TOOL, ReflectionAction.ADD_TOOL)
            for a in actions
        )

        # 8. Generate summary
        summary = self._generate_summary(observations, actions, iteration)

        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            "[ReAct] Iteration %d: %d observations, %d actions, retry=%s (%.1fms)",
            iteration, len(observations), len(actions), should_retry, latency,
        )

        return ReflectionResult(
            iteration=iteration,
            observations=observations,
            actions=actions,
            should_retry=should_retry,
            summary=summary,
        )

    def _generate_summary(
        self,
        observations: list[Observation],
        actions: list[ReflectionAction_],
        iteration: int,
    ) -> str:
        """Generate Vietnamese narrative summary of this reflection."""
        parts = [f"Vòng suy luận #{iteration + 1}:"]

        # Count by severity
        critical = sum(1 for o in observations if o.severity == "critical")
        warning = sum(1 for o in observations if o.severity == "warning")
        info = sum(1 for o in observations if o.severity == "info")

        if critical:
            parts.append(f"🔴 {critical} vấn đề nghiêm trọng.")
        if warning:
            parts.append(f"🟡 {warning} cảnh báo.")
        if info and not critical and not warning:
            parts.append("🟢 Không phát hiện vấn đề.")

        # Actions
        retry_actions = [a for a in actions if a.action != ReflectionAction.PROCEED]
        if retry_actions:
            action_strs = [a.reason[:60] for a in retry_actions[:3]]
            parts.append(f"Hành động: {'; '.join(action_strs)}.")
        else:
            parts.append("Kết quả đủ tốt — tiến hành tổng hợp.")

        return " ".join(parts)
