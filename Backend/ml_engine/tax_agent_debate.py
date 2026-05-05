"""
tax_agent_debate.py – Multi-Agent Debate Protocol
===================================================
Implements structured debate between sub-agents for consensus-building
and disagreement transparency.

Architecture (inspired by Du et al., 2023 — "Multiagent Debate"):
    1. Each sub-agent provides an independent assessment
    2. Pairwise disagreements are detected and scored
    3. Cross-examination: agents respond to conflicting views
    4. Final consensus via weighted voting + minority opinion extraction

Key Design Decisions:
    - Deterministic: No LLM calls in the debate itself — purely rule-based
      comparison of structured agent outputs. This ensures reproducibility
      and auditability for tax authority compliance.
    - Transparent: Every disagreement and resolution step is logged and
      returned to the frontend for display.
    - Non-blocking: Debate runs in <50ms since it operates on pre-computed
      agent results (no I/O).

Sub-Agents involved:
    - Legal Research Agent: Evaluates regulatory compliance
    - Analytics Agent: Computes composite risk score from ML models
    - Investigation Agent: Identifies suspicious patterns and red flags

Reference:
    Du et al., "Improving Factuality and Reasoning in Language Models
    through Multiagent Debate", ICML 2023
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

class StanceType(str, Enum):
    """Agent's stance on risk level."""
    SAFE = "safe"
    CAUTIOUS = "cautious"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


class DisagreementSeverity(str, Enum):
    """How serious the disagreement is."""
    MINOR = "minor"         # Same general direction, small score gap
    MODERATE = "moderate"   # Different risk levels but adjacent
    MAJOR = "major"         # Completely opposite conclusions
    CRITICAL = "critical"   # Direct contradiction on key finding


@dataclass
class AgentStance:
    """Structured representation of one agent's position."""
    agent_name: str
    agent_label: str           # Vietnamese display name
    agent_icon: str            # FontAwesome icon class
    stance: StanceType
    confidence: float          # 0.0 - 1.0
    risk_score: float          # 0.0 - 100.0 (normalized)
    key_findings: list[str]    # Top findings supporting stance
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "label": self.agent_label,
            "icon": self.agent_icon,
            "stance": self.stance.value,
            "confidence": round(self.confidence, 2),
            "risk_score": round(self.risk_score, 1),
            "findings": self.key_findings[:5],
        }


@dataclass
class Disagreement:
    """A detected disagreement between two agents."""
    agent_a: str
    agent_b: str
    severity: DisagreementSeverity
    topic: str                      # What they disagree about
    agent_a_position: str           # Agent A's stance (brief)
    agent_b_position: str           # Agent B's stance (brief)
    resolution: str | None = None   # How it was resolved

    def to_dict(self) -> dict:
        return {
            "agents": [self.agent_a, self.agent_b],
            "severity": self.severity.value,
            "topic": self.topic,
            "positions": {
                self.agent_a: self.agent_a_position,
                self.agent_b: self.agent_b_position,
            },
            "resolution": self.resolution,
        }


@dataclass
class DebateResult:
    """Complete debate outcome."""
    consensus_score: float          # 0.0 - 1.0 (how much agents agree)
    consensus_stance: StanceType    # Overall consensus
    consensus_label: str            # Vietnamese description
    stances: list[AgentStance]      # Each agent's position
    disagreements: list[Disagreement]
    minority_opinions: list[dict[str, Any]]
    recommendation: str             # Final actionable recommendation
    debate_summary: str             # Vietnamese narrative summary
    adjudicator_verdict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "consensus_score": round(self.consensus_score, 2),
            "consensus_pct": round(self.consensus_score * 100, 1),
            "consensus_stance": self.consensus_stance.value,
            "consensus_label": self.consensus_label,
            "stances": [s.to_dict() for s in self.stances],
            "disagreements": [d.to_dict() for d in self.disagreements],
            "minority_opinions": self.minority_opinions,
            "recommendation": self.recommendation,
            "summary": self.debate_summary,
            "agent_count": len(self.stances),
        }
        if self.adjudicator_verdict:
            result["adjudicator_verdict"] = self.adjudicator_verdict
        return result


# ════════════════════════════════════════════════════════════════
#  Stance Extraction (per agent type)
# ════════════════════════════════════════════════════════════════

def _extract_legal_stance(data: dict) -> AgentStance | None:
    """Extract structured stance from Legal Research Agent output."""
    if not data:
        return None

    confidence = float(data.get("confidence", 0))
    authority = float(data.get("authority_score", 0))

    # Legal agent: high authority + high confidence = safe
    # Low authority or caveats = cautious/suspicious
    caveats = data.get("caveats", [])
    conclusion = str(data.get("conclusion", ""))

    # Determine risk score (inverse of compliance)
    risk_score = 100 - (confidence * 50 + authority * 50)
    if caveats:
        risk_score += len(caveats) * 8
    risk_score = min(100, max(0, risk_score))

    # Determine stance
    if risk_score < 25:
        stance = StanceType.SAFE
    elif risk_score < 50:
        stance = StanceType.CAUTIOUS
    elif risk_score < 75:
        stance = StanceType.SUSPICIOUS
    else:
        stance = StanceType.DANGEROUS

    findings = []
    if conclusion:
        findings.append(conclusion[:120])
    for caveat in caveats[:3]:
        findings.append(f"⚠️ {caveat}")
    laws = data.get("applicable_laws", [])
    if laws:
        findings.append(f"📋 {len(laws)} điều luật liên quan")

    return AgentStance(
        agent_name="legal",
        agent_label="Pháp lý",
        agent_icon="fa-solid fa-scale-balanced",
        stance=stance,
        confidence=confidence,
        risk_score=risk_score,
        key_findings=findings,
        raw_data=data,
    )


def _extract_analytics_stance(data: dict) -> AgentStance | None:
    """Extract structured stance from Analytics Agent output."""
    if not data:
        return None

    risk_score = float(data.get("composite_risk_score", 0)) * 100
    confidence = float(data.get("confidence", 0))
    risk_level = str(data.get("risk_level", ""))

    # Determine stance from risk level
    stance_map = {
        "critical": StanceType.DANGEROUS,
        "high": StanceType.SUSPICIOUS,
        "moderate": StanceType.CAUTIOUS,
        "low": StanceType.SAFE,
        "minimal": StanceType.SAFE,
    }
    stance = stance_map.get(risk_level, StanceType.CAUTIOUS)

    findings = []
    summary = data.get("summary", "")
    if summary:
        findings.append(summary[:120])
    recommendations = data.get("recommendations", [])
    for rec in recommendations[:2]:
        findings.append(f"💡 {rec}")
    trend = data.get("risk_trend", "")
    if trend:
        findings.append(f"📈 Xu hướng: {trend}")

    return AgentStance(
        agent_name="analytics",
        agent_label="Phân tích",
        agent_icon="fa-solid fa-chart-pie",
        stance=stance,
        confidence=confidence,
        risk_score=risk_score,
        key_findings=findings,
        raw_data=data,
    )


def _extract_investigation_stance(data: dict) -> AgentStance | None:
    """Extract structured stance from Investigation Agent output."""
    if not data:
        return None

    overall_score = float(data.get("overall_score", 0))
    risk_score = overall_score * 100
    confidence = float(data.get("confidence", 0))
    suspicion_level = str(data.get("suspicion_level", ""))

    stance_map = {
        "critical": StanceType.DANGEROUS,
        "high": StanceType.SUSPICIOUS,
        "medium": StanceType.CAUTIOUS,
        "low": StanceType.SAFE,
    }
    stance = stance_map.get(suspicion_level, StanceType.CAUTIOUS)

    findings = []
    exec_summary = data.get("executive_summary", "")
    if exec_summary:
        findings.append(exec_summary[:120])
    patterns_count = data.get("patterns_count", 0)
    if patterns_count > 0:
        findings.append(f"🔍 Phát hiện {patterns_count} mẫu giao dịch đáng ngờ")
    escalation = data.get("escalation_level", "")
    if escalation:
        findings.append(f"⚡ Mức leo thang: {escalation}")
    actions = data.get("recommended_actions", [])
    for action in actions[:2]:
        findings.append(f"🎯 {action}")

    return AgentStance(
        agent_name="investigation",
        agent_label="Điều tra",
        agent_icon="fa-solid fa-magnifying-glass",
        stance=stance,
        confidence=confidence,
        risk_score=risk_score,
        key_findings=findings,
        raw_data=data,
    )


# ════════════════════════════════════════════════════════════════
#  Disagreement Detection
# ════════════════════════════════════════════════════════════════

STANCE_ORDER = {
    StanceType.SAFE: 0,
    StanceType.CAUTIOUS: 1,
    StanceType.SUSPICIOUS: 2,
    StanceType.DANGEROUS: 3,
}


def _detect_disagreements(stances: list[AgentStance]) -> list[Disagreement]:
    """Detect pairwise disagreements between agents."""
    disagreements = []

    for i in range(len(stances)):
        for j in range(i + 1, len(stances)):
            a, b = stances[i], stances[j]
            gap = abs(STANCE_ORDER[a.stance] - STANCE_ORDER[b.stance])
            score_gap = abs(a.risk_score - b.risk_score)

            if gap == 0 and score_gap < 20:
                continue  # They agree — skip

            # Determine severity
            if gap >= 3 or score_gap > 60:
                severity = DisagreementSeverity.CRITICAL
            elif gap >= 2 or score_gap > 40:
                severity = DisagreementSeverity.MAJOR
            elif gap >= 1 or score_gap > 25:
                severity = DisagreementSeverity.MODERATE
            else:
                severity = DisagreementSeverity.MINOR

            # Determine topic
            topic = _identify_topic(a, b)

            # Generate resolution
            resolution = _resolve_disagreement(a, b, severity)

            disagreements.append(Disagreement(
                agent_a=a.agent_label,
                agent_b=b.agent_label,
                severity=severity,
                topic=topic,
                agent_a_position=f"{a.stance.value} ({a.risk_score:.0f}%)",
                agent_b_position=f"{b.stance.value} ({b.risk_score:.0f}%)",
                resolution=resolution,
            ))

    return disagreements


def _identify_topic(a: AgentStance, b: AgentStance) -> str:
    """Identify the primary topic of disagreement."""
    if a.agent_name == "legal" and b.agent_name == "investigation":
        return "Tuân thủ pháp lý vs. Dấu hiệu điều tra"
    elif a.agent_name == "legal" and b.agent_name == "analytics":
        return "Pháp lý vs. Phân tích rủi ro số liệu"
    elif a.agent_name == "analytics" and b.agent_name == "investigation":
        return "Chỉ số rủi ro vs. Mẫu giao dịch đáng ngờ"
    return "Đánh giá rủi ro tổng thể"


def _resolve_disagreement(
    a: AgentStance, b: AgentStance, severity: DisagreementSeverity,
) -> str:
    """Generate resolution recommendation."""
    higher_risk = a if a.risk_score > b.risk_score else b
    lower_risk = b if a.risk_score > b.risk_score else a

    if severity == DisagreementSeverity.CRITICAL:
        return (
            f"Mâu thuẫn nghiêm trọng: {higher_risk.agent_label} đánh giá rủi ro rất cao "
            f"({higher_risk.risk_score:.0f}%) trong khi {lower_risk.agent_label} đánh giá thấp "
            f"({lower_risk.risk_score:.0f}%). Khuyến nghị: THANH TRA THỰC ĐỊA để xác minh."
        )
    elif severity == DisagreementSeverity.MAJOR:
        return (
            f"Bất đồng lớn giữa {a.agent_label} và {b.agent_label}. "
            f"Áp dụng nguyên tắc thận trọng — ưu tiên đánh giá của {higher_risk.agent_label}."
        )
    elif severity == DisagreementSeverity.MODERATE:
        return (
            f"Khác biệt vừa phải. Cần thu thập thêm dữ liệu để khẳng định."
        )
    else:
        return "Khác biệt nhỏ, không ảnh hưởng đáng kể đến kết luận."


# ════════════════════════════════════════════════════════════════
#  Consensus Computation
# ════════════════════════════════════════════════════════════════

# Agent weights reflect domain expertise relevance
AGENT_WEIGHTS = {
    "analytics": 0.45,      # ML models — highest weight
    "investigation": 0.35,  # Pattern detection
    "legal": 0.20,          # Regulatory compliance
}


def _compute_consensus(stances: list[AgentStance]) -> tuple[float, StanceType]:
    """
    Compute weighted consensus score and stance.

    Returns:
        (consensus_agreement, consensus_stance)
        consensus_agreement: 0.0 (total disagreement) to 1.0 (perfect agreement)
    """
    if not stances:
        return 0.0, StanceType.CAUTIOUS

    # Weighted risk score
    total_weight = 0.0
    weighted_score = 0.0
    for s in stances:
        w = AGENT_WEIGHTS.get(s.agent_name, 0.25)
        weighted_score += s.risk_score * w
        total_weight += w

    consensus_risk = weighted_score / max(total_weight, 0.01)

    # Agreement score: inverse of variance
    if len(stances) >= 2:
        scores = [s.risk_score for s in stances]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        max_variance = 50 ** 2  # Max possible variance (0 vs 100)
        agreement = 1.0 - min(1.0, variance / max_variance)
    else:
        agreement = 1.0

    # Consensus stance from weighted score
    if consensus_risk < 25:
        stance = StanceType.SAFE
    elif consensus_risk < 50:
        stance = StanceType.CAUTIOUS
    elif consensus_risk < 75:
        stance = StanceType.SUSPICIOUS
    else:
        stance = StanceType.DANGEROUS

    return round(agreement, 3), stance


# ════════════════════════════════════════════════════════════════
#  Main Debate Protocol
# ════════════════════════════════════════════════════════════════

class AgentDebateProtocol:
    """
    Multi-agent debate framework for TaxInspector.

    Orchestrates structured debate between Legal, Analytics, and
    Investigation sub-agents to produce a transparent consensus
    with full disagreement traceability.

    Usage:
        debate = AgentDebateProtocol()
        result = debate.run_debate(sub_agent_analysis)
        # result.to_dict() → ready for frontend
    """

    def run_debate(
        self,
        sub_agent_analysis: dict[str, Any],
        tool_results: dict[str, Any] | None = None,
    ) -> DebateResult:
        """
        Execute the debate protocol.

        Args:
            sub_agent_analysis: Dict with keys "legal_research",
                "analytics", "investigation" — each containing the
                agent's output dict.
            tool_results: Optional raw tool outputs for enrichment.

        Returns:
            DebateResult with full consensus data.
        """
        # Step 1: Extract structured stances from each agent
        stances: list[AgentStance] = []

        legal_stance = _extract_legal_stance(
            sub_agent_analysis.get("legal_research")
        )
        if legal_stance:
            stances.append(legal_stance)

        analytics_stance = _extract_analytics_stance(
            sub_agent_analysis.get("analytics")
        )
        if analytics_stance:
            stances.append(analytics_stance)

        investigation_stance = _extract_investigation_stance(
            sub_agent_analysis.get("investigation")
        )
        if investigation_stance:
            stances.append(investigation_stance)

        if len(stances) < 2:
            # Not enough agents for meaningful debate
            single = stances[0] if stances else None
            return DebateResult(
                consensus_score=1.0,
                consensus_stance=single.stance if single else StanceType.CAUTIOUS,
                consensus_label="Chỉ có 1 agent — không đủ để tranh luận",
                stances=stances,
                disagreements=[],
                minority_opinions=[],
                recommendation=single.key_findings[0] if single and single.key_findings else "Cần thêm dữ liệu.",
                debate_summary="Chỉ một agent tham gia phân tích. Không có tranh luận đa chiều.",
            )

        # Step 2: Detect pairwise disagreements
        disagreements = _detect_disagreements(stances)

        # Step 3: Compute consensus
        consensus_score, consensus_stance = _compute_consensus(stances)

        # Step 4: Extract minority opinions
        minority_opinions = self._extract_minority(stances, consensus_stance)

        # Step 5: Generate recommendation
        recommendation = self._generate_recommendation(
            stances, disagreements, consensus_stance, consensus_score,
        )

        # Step 6: Generate narrative summary
        summary = self._generate_summary(
            stances, disagreements, consensus_score, consensus_stance,
        )

        # Step 7: Consensus label
        consensus_label = self._consensus_label(consensus_score, consensus_stance)

        # Step 8: Adjudicator — triggers when consensus is low or critical disagreements exist
        adjudicator_verdict = {}
        _has_critical = any(
            d.severity in (DisagreementSeverity.CRITICAL, DisagreementSeverity.MAJOR)
            for d in disagreements
        )
        if consensus_score < 0.58 or _has_critical:
            adjudicator = AdjudicatorAgent()
            adjudicator_verdict = adjudicator.adjudicate(
                stances=stances,
                disagreements=disagreements,
                consensus_score=consensus_score,
                consensus_stance=consensus_stance,
                minority_opinions=minority_opinions,
            )
            # Override recommendation with adjudicator's if available
            if adjudicator_verdict.get("recommended_action"):
                recommendation = adjudicator_verdict["recommended_action"]

        return DebateResult(
            consensus_score=consensus_score,
            consensus_stance=consensus_stance,
            consensus_label=consensus_label,
            stances=stances,
            disagreements=disagreements,
            minority_opinions=minority_opinions,
            recommendation=recommendation,
            debate_summary=summary,
            adjudicator_verdict=adjudicator_verdict,
        )

    def _extract_minority(
        self,
        stances: list[AgentStance],
        consensus: StanceType,
    ) -> list[dict[str, Any]]:
        """Find agents whose stance differs from consensus."""
        minorities = []
        for s in stances:
            gap = abs(STANCE_ORDER[s.stance] - STANCE_ORDER[consensus])
            if gap >= 2:  # At least 2 levels away from consensus
                minorities.append({
                    "agent": s.agent_label,
                    "icon": s.agent_icon,
                    "stance": s.stance.value,
                    "risk_score": round(s.risk_score, 1),
                    "reason": s.key_findings[0] if s.key_findings else "Không rõ lý do",
                })
        return minorities

    def _generate_recommendation(
        self,
        stances: list[AgentStance],
        disagreements: list[Disagreement],
        consensus_stance: StanceType,
        consensus_score: float,
    ) -> str:
        """Generate actionable recommendation based on debate outcome."""
        critical_disagreements = [
            d for d in disagreements
            if d.severity in (DisagreementSeverity.CRITICAL, DisagreementSeverity.MAJOR)
        ]

        if critical_disagreements:
            return (
                "⚠️ Phát hiện mâu thuẫn nghiêm trọng giữa các agent. "
                "Khuyến nghị: Thanh tra thực địa hoặc yêu cầu bổ sung hồ sơ "
                "trước khi ra quyết định."
            )

        if consensus_stance == StanceType.DANGEROUS:
            return (
                "🔴 Đồng thuận cao: Doanh nghiệp có rủi ro nghiêm trọng. "
                "Đề xuất đưa vào danh sách ưu tiên thanh tra."
            )
        elif consensus_stance == StanceType.SUSPICIOUS:
            return (
                "🟠 Có dấu hiệu đáng ngờ. Đề xuất theo dõi chặt chẽ "
                "và lên kế hoạch thanh tra trong 6 tháng tới."
            )
        elif consensus_stance == StanceType.CAUTIOUS:
            return (
                "🟡 Mức rủi ro trung bình. Đề xuất tiếp tục giám sát "
                "theo quy trình thường xuyên."
            )
        else:
            return (
                "🟢 Các agent đồng thuận: Doanh nghiệp hoạt động bình thường. "
                "Không cần hành động đặc biệt."
            )

    def _generate_summary(
        self,
        stances: list[AgentStance],
        disagreements: list[Disagreement],
        consensus_score: float,
        consensus_stance: StanceType,
    ) -> str:
        """Generate Vietnamese narrative summary of the debate."""
        n_agents = len(stances)
        n_disagree = len(disagreements)
        pct = round(consensus_score * 100, 1)

        agent_names = ", ".join(s.agent_label for s in stances)

        parts = [
            f"Hội đồng {n_agents} agent ({agent_names}) đã tiến hành đánh giá đa chiều."
        ]

        if n_disagree == 0:
            parts.append(f"Mức đồng thuận: {pct}% — các agent hoàn toàn nhất trí.")
        elif n_disagree == 1:
            d = disagreements[0]
            parts.append(
                f"Mức đồng thuận: {pct}%. Có 1 bất đồng ({d.severity.value}) "
                f"giữa {d.agent_a} và {d.agent_b} về \"{d.topic}\"."
            )
        else:
            parts.append(
                f"Mức đồng thuận: {pct}%. Phát hiện {n_disagree} bất đồng."
            )

        stance_labels = {
            StanceType.SAFE: "An toàn",
            StanceType.CAUTIOUS: "Cần lưu ý",
            StanceType.SUSPICIOUS: "Đáng ngờ",
            StanceType.DANGEROUS: "Nguy hiểm",
        }
        parts.append(
            f"Kết luận đồng thuận: {stance_labels.get(consensus_stance, '?')}."
        )

        return " ".join(parts)

    def _consensus_label(self, score: float, stance: StanceType) -> str:
        """Short label for consensus state."""
        pct = round(score * 100)
        stance_vi = {
            StanceType.SAFE: "An toàn",
            StanceType.CAUTIOUS: "Cần lưu ý",
            StanceType.SUSPICIOUS: "Đáng ngờ",
            StanceType.DANGEROUS: "Nguy hiểm",
        }
        return f"{stance_vi.get(stance, '?')} — Đồng thuận {pct}%"


# ════════════════════════════════════════════════════════════════
#  Adjudicator Agent — Arbiter for Serious Disagreements
# ════════════════════════════════════════════════════════════════

class AdjudicatorAgent:
    """
    Trọng tài AI — chỉ kích hoạt khi agents bất đồng nghiêm trọng.

    When the debate consensus drops below 0.58 or critical/major
    disagreements are detected, the Adjudicator steps in with
    structured reasoning about WHY the final decision should lean
    one way or another.

    Design: Deterministic, rule-based — no LLM calls.
    Evaluates evidence weight per agent based on:
      - Citation count / authority score (legal)
      - ML model confidence (analytics)
      - Pattern detection count (investigation)
    """

    # Evidence quality weights for adjudication
    EVIDENCE_WEIGHTS = {
        "legal": {
            "citation_count": 0.25,
            "authority_score": 0.35,
            "confidence": 0.40,
        },
        "analytics": {
            "model_confidence": 0.50,
            "data_coverage": 0.30,
            "trend_signal": 0.20,
        },
        "investigation": {
            "patterns_found": 0.40,
            "confidence": 0.35,
            "escalation_weight": 0.25,
        },
    }

    def adjudicate(
        self,
        *,
        stances: list[AgentStance],
        disagreements: list[Disagreement],
        consensus_score: float,
        consensus_stance: StanceType,
        minority_opinions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Produce a structured adjudicator verdict.

        Returns:
            Dict with verdict, reasoning, evidence_weights, and recommended_action.
        """
        evidence_scores: dict[str, dict[str, Any]] = {}
        for stance in stances:
            evidence_scores[stance.agent_name] = self._evaluate_evidence(stance)

        # Determine which agent has strongest evidence
        agent_totals = {
            name: scores.get("total", 0.0)
            for name, scores in evidence_scores.items()
        }
        strongest_agent = max(agent_totals, key=agent_totals.get) if agent_totals else "analytics"
        strongest_stance = next(
            (s for s in stances if s.agent_name == strongest_agent), None
        )

        # Build reasoning chain
        reasoning_lines = []
        for stance in stances:
            score_info = evidence_scores.get(stance.agent_name, {})
            total = score_info.get("total", 0)
            reasoning_lines.append(
                f"{stance.agent_label} ({stance.stance.value}, "
                f"risk={stance.risk_score:.0f}%): "
                f"evidence weight = {total:.2f}"
            )

        # Disagreement analysis
        critical_topics = [
            d.topic for d in disagreements
            if d.severity in (DisagreementSeverity.CRITICAL, DisagreementSeverity.MAJOR)
        ]

        # Verdict
        if strongest_stance and strongest_stance.risk_score > 65:
            verdict = "high_risk_confirmed"
            verdict_vi = "Xác nhận rủi ro cao"
            action = (
                "⚖️ Trọng tài phán quyết: Bằng chứng mạnh nhất đến từ "
                f"{strongest_stance.agent_label} (weight={agent_totals[strongest_agent]:.2f}). "
                "Khuyến nghị: Ưu tiên thanh tra thực địa, thu thập thêm hồ sơ trước khi kết luận."
            )
        elif strongest_stance and strongest_stance.risk_score < 30:
            verdict = "low_risk_confirmed"
            verdict_vi = "Xác nhận rủi ro thấp"
            action = (
                "⚖️ Trọng tài phán quyết: Dù có bất đồng, bằng chứng chính từ "
                f"{strongest_stance.agent_label} cho thấy rủi ro thấp. "
                "Đề xuất: Giám sát theo quy trình thường xuyên."
            )
        else:
            verdict = "inconclusive"
            verdict_vi = "Chưa đủ bằng chứng để kết luận"
            action = (
                "⚖️ Trọng tài phán quyết: Bất đồng chưa giải quyết được. "
                "Cần thu thập thêm dữ liệu từ "
                + ", ".join(critical_topics[:2])
                + ". Chuyển hồ sơ cho chuyên gia cấp cao."
            )

        return {
            "verdict": verdict,
            "verdict_vi": verdict_vi,
            "strongest_agent": strongest_agent,
            "strongest_agent_label": strongest_stance.agent_label if strongest_stance else "N/A",
            "evidence_weights": evidence_scores,
            "agent_ranking": sorted(
                agent_totals.items(), key=lambda x: x[1], reverse=True,
            ),
            "reasoning": reasoning_lines,
            "critical_topics": critical_topics,
            "consensus_score": round(consensus_score, 3),
            "minority_count": len(minority_opinions),
            "recommended_action": action,
        }

    def _evaluate_evidence(self, stance: AgentStance) -> dict[str, Any]:
        """Score the quality of evidence behind this agent's stance."""
        raw = stance.raw_data or {}
        agent = stance.agent_name

        if agent == "legal":
            citations = len(raw.get("applicable_laws", []) or raw.get("citation_chain", []))
            authority = float(raw.get("authority_score", 0))
            conf = stance.confidence
            w = self.EVIDENCE_WEIGHTS["legal"]
            total = (
                min(1.0, citations / 5) * w["citation_count"]
                + authority * w["authority_score"]
                + conf * w["confidence"]
            )
            return {
                "citations": citations,
                "authority": round(authority, 3),
                "confidence": round(conf, 3),
                "total": round(total, 4),
            }

        elif agent == "analytics":
            conf = stance.confidence
            risk = stance.risk_score / 100.0
            trend = 0.5 if raw.get("risk_trend") else 0.3
            w = self.EVIDENCE_WEIGHTS["analytics"]
            total = conf * w["model_confidence"] + risk * w["data_coverage"] + trend * w["trend_signal"]
            return {
                "model_confidence": round(conf, 3),
                "risk_signal": round(risk, 3),
                "trend_signal": round(trend, 3),
                "total": round(total, 4),
            }

        elif agent == "investigation":
            patterns = int(raw.get("patterns_count", 0))
            conf = stance.confidence
            escalation = 0.8 if raw.get("escalation_level") in ("high", "critical") else 0.3
            w = self.EVIDENCE_WEIGHTS["investigation"]
            total = (
                min(1.0, patterns / 5) * w["patterns_found"]
                + conf * w["confidence"]
                + escalation * w["escalation_weight"]
            )
            return {
                "patterns_count": patterns,
                "confidence": round(conf, 3),
                "escalation": round(escalation, 3),
                "total": round(total, 4),
            }

        # Unknown agent
        return {"total": round(stance.confidence * 0.5, 4)}
