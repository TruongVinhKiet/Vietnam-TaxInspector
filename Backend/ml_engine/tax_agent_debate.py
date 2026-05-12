"""
tax_agent_debate.py – Multi-Agent Debate Engine (Enterprise v2)
================================================================
Implements structured adversarial debate between Inspector, Defense,
Legal, and Judge agents to reduce false positives and improve
analytical rigor before producing final risk conclusions.

Debate Roles:
    Inspector Agent  – presents risk evidence, anomalies, graph signals
    Defense Agent    – counter-argues with data gaps, industry norms, legal exceptions
    Legal Agent      – provides statutory grounding (only when legal context needed)
    Judge Agent      – synthesizes verdict, adjusts confidence, decides escalation

Trigger conditions:
    - Fraud/VAT risk score >= HIGH_RISK_THRESHOLD
    - Ring/motif detection with severity >= CRITICAL
    - Confidence below CONFIDENCE_FLOOR
    - Legal contradiction detected
    - User explicitly requests investigation report
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 70.0
CONFIDENCE_FLOOR = 0.55
MAX_DEBATE_ROUNDS = 3
DEBATE_BUDGET_MS = 5000


class DebateRole(str, Enum):
    INSPECTOR = "inspector"
    DEFENSE = "defense"
    LEGAL = "legal"
    JUDGE = "judge"


class DebateVerdict(str, Enum):
    CONFIRMED_HIGH_RISK = "confirmed_high_risk"
    DOWNGRADED = "downgraded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


@dataclass
class DebateArgument:
    """A single argument in a debate round."""
    role: DebateRole
    claim: str
    evidence: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # delta to apply to risk score
    cited_rules: list[str] = field(default_factory=list)
    counter_to: str | None = None


@dataclass
class DebateRound:
    """A round of debate with arguments from multiple agents."""
    round_number: int
    arguments: list[DebateArgument] = field(default_factory=list)
    round_verdict: str = ""
    confidence_after: float = 0.0


@dataclass
class DebateSession:
    """Complete debate session result."""
    session_id: str
    trigger_reason: str
    subject_tax_code: str | None = None
    initial_risk_score: float = 0.0
    final_risk_score: float = 0.0
    verdict: DebateVerdict = DebateVerdict.NO_ACTION
    verdict_reasoning: str = ""
    rounds: list[DebateRound] = field(default_factory=list)
    total_latency_ms: float = 0.0
    escalation_recommended: bool = False
    wording_guidance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trigger_reason": self.trigger_reason,
            "subject_tax_code": self.subject_tax_code,
            "initial_risk_score": self.initial_risk_score,
            "final_risk_score": round(self.final_risk_score, 2),
            "verdict": self.verdict.value,
            "verdict_reasoning": self.verdict_reasoning,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "arguments": [
                        {
                            "role": a.role.value,
                            "claim": a.claim,
                            "evidence": a.evidence,
                            "confidence_adjustment": a.confidence_adjustment,
                            "cited_rules": a.cited_rules,
                        }
                        for a in r.arguments
                    ],
                    "round_verdict": r.round_verdict,
                    "confidence_after": round(r.confidence_after, 2),
                }
                for r in self.rounds
            ],
            "total_latency_ms": round(self.total_latency_ms, 1),
            "escalation_recommended": self.escalation_recommended,
            "wording_guidance": self.wording_guidance,
        }


class MultiAgentDebateEngine:
    """
    Orchestrates structured adversarial debate for high-stakes tax decisions.
    
    The engine is deterministic and rule-based (no LLM dependency).
    It synthesizes existing tool outputs into a structured argument flow.
    """

    def should_trigger(
        self,
        *,
        risk_score: float = 0.0,
        confidence: float = 1.0,
        has_ring_motif: bool = False,
        ring_severity: str = "",
        has_legal_contradiction: bool = False,
        user_requests_report: bool = False,
        mode: str = "fraud",
    ) -> tuple[bool, str]:
        """Check whether a debate should be triggered. Returns (should_trigger, reason)."""
        if user_requests_report and risk_score > 40:
            return True, "user_report_request"
        if risk_score >= HIGH_RISK_THRESHOLD:
            return True, f"high_risk_score_{risk_score:.0f}"
        if has_ring_motif and ring_severity in ("critical", "high"):
            return True, f"ring_motif_{ring_severity}"
        if confidence < CONFIDENCE_FLOOR:
            return True, f"low_confidence_{confidence:.2f}"
        if has_legal_contradiction:
            return True, "legal_contradiction"
        return False, ""

    def run_debate(
        self,
        *,
        risk_score: float,
        risk_level: str = "unknown",
        tax_code: str | None = None,
        company_name: str | None = None,
        trigger_reason: str = "",
        tool_results: dict[str, Any] | None = None,
        batch_data: dict[str, Any] | None = None,
        vat_snapshot: dict[str, Any] | None = None,
        legal_facts: list[dict[str, Any]] | None = None,
        mode: str = "fraud",
    ) -> DebateSession:
        """
        Execute a multi-round debate and produce a verdict.
        
        This is a deterministic debate engine that uses structured rules
        to generate Inspector/Defense/Legal/Judge arguments from existing
        analytical outputs.
        """
        t0 = time.perf_counter()
        session_id = f"debate-{uuid.uuid4().hex[:12]}"
        tool_results = tool_results or {}
        batch_data = batch_data or {}
        vat_snapshot = vat_snapshot or {}
        legal_facts = legal_facts or []

        session = DebateSession(
            session_id=session_id,
            trigger_reason=trigger_reason,
            subject_tax_code=tax_code,
            initial_risk_score=risk_score,
            final_risk_score=risk_score,
        )

        current_score = risk_score

        # ── Round 1: Inspector presents evidence ──────────────────────
        round1 = DebateRound(round_number=1)
        inspector_args = self._build_inspector_arguments(
            risk_score=risk_score,
            risk_level=risk_level,
            tax_code=tax_code,
            tool_results=tool_results,
            batch_data=batch_data,
            vat_snapshot=vat_snapshot,
            mode=mode,
        )
        round1.arguments.extend(inspector_args)

        # ── Round 1: Defense counter-arguments ────────────────────────
        defense_args = self._build_defense_arguments(
            risk_score=risk_score,
            tool_results=tool_results,
            batch_data=batch_data,
            mode=mode,
        )
        round1.arguments.extend(defense_args)

        # Apply confidence adjustments from round 1
        for arg in round1.arguments:
            current_score += arg.confidence_adjustment
        current_score = max(0.0, min(100.0, current_score))
        round1.confidence_after = current_score
        round1.round_verdict = (
            "Inspector đã trình bày bằng chứng. Defense đã phản biện."
        )
        session.rounds.append(round1)

        # ── Round 2: Legal Agent (if legal facts present) ─────────────
        if legal_facts:
            round2 = DebateRound(round_number=2)
            legal_args = self._build_legal_arguments(legal_facts, risk_score)
            round2.arguments.extend(legal_args)
            for arg in round2.arguments:
                current_score += arg.confidence_adjustment
            current_score = max(0.0, min(100.0, current_score))
            round2.confidence_after = current_score
            round2.round_verdict = "Legal Agent đã cung cấp căn cứ pháp lý."
            session.rounds.append(round2)

        # ── Final Round: Judge verdict ────────────────────────────────
        judge_round = DebateRound(round_number=len(session.rounds) + 1)
        judge_arg, verdict = self._build_judge_verdict(
            initial_score=risk_score,
            current_score=current_score,
            inspector_args=inspector_args,
            defense_args=defense_args,
            legal_args=legal_facts,
        )
        judge_round.arguments.append(judge_arg)
        judge_round.confidence_after = current_score
        judge_round.round_verdict = verdict.value
        session.rounds.append(judge_round)

        session.final_risk_score = current_score
        session.verdict = verdict
        session.verdict_reasoning = judge_arg.claim
        session.escalation_recommended = verdict in (
            DebateVerdict.CONFIRMED_HIGH_RISK,
            DebateVerdict.ESCALATE,
        )
        session.wording_guidance = self._generate_wording_guidance(verdict, current_score)
        session.total_latency_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "[Debate] Session %s completed: verdict=%s, score %.1f→%.1f, rounds=%d",
            session_id, verdict.value, risk_score, current_score, len(session.rounds),
        )
        return session

    # ─── Inspector Agent ──────────────────────────────────────────────────────

    def _build_inspector_arguments(
        self,
        *,
        risk_score: float,
        risk_level: str,
        tax_code: str | None,
        tool_results: dict,
        batch_data: dict,
        vat_snapshot: dict,
        mode: str,
    ) -> list[DebateArgument]:
        args: list[DebateArgument] = []

        # Core risk score argument
        args.append(DebateArgument(
            role=DebateRole.INSPECTOR,
            claim=f"Doanh nghiệp {tax_code or 'N/A'} có điểm rủi ro {risk_score:.1f}/100, "
                  f"thuộc mức '{risk_level}'. Đây là mức cần giám sát chặt chẽ.",
            evidence=[
                f"risk_score={risk_score}",
                f"risk_level={risk_level}",
            ],
            confidence_adjustment=0.0,
        ))

        # Invoice anomalies
        companies = batch_data.get("companies", [])
        high_risk_count = sum(
            1 for c in companies
            if float(c.get("risk_score", 0)) >= HIGH_RISK_THRESHOLD
        )
        if high_risk_count > 0:
            args.append(DebateArgument(
                role=DebateRole.INSPECTOR,
                claim=f"Trong lô phân tích, {high_risk_count}/{len(companies)} "
                      f"doanh nghiệp có rủi ro >= {HIGH_RISK_THRESHOLD}.",
                evidence=[f"high_risk_count={high_risk_count}"],
                confidence_adjustment=min(5.0, high_risk_count * 0.5),
            ))

        # VAT ring/motif evidence
        rings = vat_snapshot.get("rings", [])
        if rings:
            args.append(DebateArgument(
                role=DebateRole.INSPECTOR,
                claim=f"Phát hiện {len(rings)} vòng lặp hóa đơn (ring/motif) "
                      f"trong mạng lưới giao dịch VAT.",
                evidence=[f"ring_count={len(rings)}"],
                confidence_adjustment=3.0,
            ))

        # Red flags from NLP scan
        red_flags = tool_results.get("nlp_red_flag_scan", {}).get("flags", [])
        if red_flags:
            args.append(DebateArgument(
                role=DebateRole.INSPECTOR,
                claim=f"Phát hiện {len(red_flags)} dấu hiệu cảnh báo (red flags) "
                      f"qua quét NLP trên dữ liệu kê khai.",
                evidence=[f"flag: {f.get('description', f)}" for f in red_flags[:5]],
                confidence_adjustment=2.0,
            ))

        return args

    # ─── Defense Agent ────────────────────────────────────────────────────────

    def _build_defense_arguments(
        self,
        *,
        risk_score: float,
        tool_results: dict,
        batch_data: dict,
        mode: str,
    ) -> list[DebateArgument]:
        args: list[DebateArgument] = []

        # Data completeness defense
        companies = batch_data.get("companies", [])
        total = len(companies)
        if total < 10:
            args.append(DebateArgument(
                role=DebateRole.DEFENSE,
                claim="Mẫu phân tích quá nhỏ (< 10 doanh nghiệp). "
                      "Điểm rủi ro có thể bị phóng đại do thiếu dữ liệu đối chứng.",
                evidence=[f"sample_size={total}"],
                confidence_adjustment=-5.0,
                counter_to="risk_score",
            ))

        # Industry seasonality defense
        if risk_score < 90:
            args.append(DebateArgument(
                role=DebateRole.DEFENSE,
                claim="Biến động doanh thu/chi phí có thể do yếu tố mùa vụ "
                      "(seasonal) hoặc đặc thù ngành, không nhất thiết là gian lận.",
                evidence=["seasonal_variation_possible"],
                confidence_adjustment=-2.0,
                counter_to="anomaly_detection",
            ))

        # Model uncertainty defense
        if risk_score >= 50 and risk_score < 85:
            args.append(DebateArgument(
                role=DebateRole.DEFENSE,
                claim="Điểm rủi ro ở vùng trung bình-cao (50-85), "
                      "vùng này có tỷ lệ False Positive cao nhất theo benchmark nội bộ.",
                evidence=["false_positive_zone=50-85"],
                confidence_adjustment=-3.0,
                counter_to="risk_classification",
            ))

        # Legal exception defense
        if mode == "vat":
            args.append(DebateArgument(
                role=DebateRole.DEFENSE,
                claim="Một số giao dịch VAT có thể thuộc diện hàng hóa miễn thuế, "
                      "gia công xuất khẩu, hoặc chế độ ưu đãi đầu tư theo NĐ 218/2013.",
                evidence=["vat_exemption_possible"],
                confidence_adjustment=-1.5,
                cited_rules=["NĐ 218/2013/NĐ-CP", "Luật Thuế GTGT Điều 5"],
            ))

        return args

    # ─── Legal Agent ──────────────────────────────────────────────────────────

    def _build_legal_arguments(
        self,
        legal_facts: list[dict],
        risk_score: float,
    ) -> list[DebateArgument]:
        args: list[DebateArgument] = []

        for fact in legal_facts[:3]:
            claim_text = fact.get("claim_text", "")
            value_json = fact.get("value_json", {})
            title = value_json.get("title", "") if isinstance(value_json, dict) else ""
            snippet = value_json.get("snippet", "") if isinstance(value_json, dict) else ""

            if claim_text:
                cited = []
                if title:
                    cited.append(title)

                args.append(DebateArgument(
                    role=DebateRole.LEGAL,
                    claim=claim_text,
                    evidence=[snippet] if snippet else [],
                    confidence_adjustment=0.0,  # Legal doesn't change score directly
                    cited_rules=cited,
                ))

        if not args:
            args.append(DebateArgument(
                role=DebateRole.LEGAL,
                claim="Chưa có căn cứ pháp lý cụ thể được trích dẫn cho trường hợp này. "
                      "Cần bổ sung tham chiếu Luật Quản lý thuế 2019 và các Nghị định liên quan.",
                evidence=[],
                confidence_adjustment=-1.0,
                cited_rules=["Luật Quản lý thuế 38/2019/QH14"],
            ))

        return args

    # ─── Judge Agent ──────────────────────────────────────────────────────────

    def _build_judge_verdict(
        self,
        *,
        initial_score: float,
        current_score: float,
        inspector_args: list[DebateArgument],
        defense_args: list[DebateArgument],
        legal_args: list[dict] | None,
    ) -> tuple[DebateArgument, DebateVerdict]:
        delta = current_score - initial_score
        inspector_count = len(inspector_args)
        defense_count = len(defense_args)

        if current_score >= 80:
            verdict = DebateVerdict.CONFIRMED_HIGH_RISK
            claim = (
                f"Sau {inspector_count} luận điểm buộc tội và {defense_count} phản biện, "
                f"điểm rủi ro hiệu chỉnh vẫn ở mức rất cao ({current_score:.1f}/100). "
                f"Kết luận: RỦI RO CAO – Khuyến nghị đưa vào danh sách thanh tra trọng điểm."
            )
        elif current_score >= 60:
            verdict = DebateVerdict.ESCALATE
            claim = (
                f"Điểm rủi ro hiệu chỉnh ({current_score:.1f}/100) ở mức cần escalation. "
                f"Inspector trình bày {inspector_count} bằng chứng, Defense phản biện {defense_count} điểm. "
                f"Kết luận: CẦN ĐÁNH GIÁ THÊM – Đề xuất giám sát và thu thập thêm dữ liệu."
            )
        elif delta < -10:
            verdict = DebateVerdict.DOWNGRADED
            claim = (
                f"Điểm rủi ro giảm đáng kể từ {initial_score:.1f} → {current_score:.1f} "
                f"(giảm {abs(delta):.1f} điểm) sau khi xem xét phản biện của Defense. "
                f"Kết luận: HẠ MỨC RỦI RO – Bằng chứng chưa đủ thuyết phục."
            )
        elif current_score < 40:
            verdict = DebateVerdict.NO_ACTION
            claim = (
                f"Điểm rủi ro hiệu chỉnh ({current_score:.1f}/100) ở mức thấp. "
                f"Kết luận: KHÔNG CẦN HÀNH ĐỘNG – Tiếp tục giám sát định kỳ."
            )
        else:
            verdict = DebateVerdict.INSUFFICIENT_EVIDENCE
            claim = (
                f"Điểm rủi ro hiệu chỉnh ({current_score:.1f}/100) ở mức trung bình. "
                f"Cả Inspector ({inspector_count} điểm) và Defense ({defense_count} điểm) "
                f"đều có lý lẽ hợp lệ. Kết luận: CHƯA ĐỦ BẰNG CHỨNG – Cần bổ sung dữ liệu."
            )

        judge_arg = DebateArgument(
            role=DebateRole.JUDGE,
            claim=claim,
            evidence=[
                f"initial_score={initial_score:.1f}",
                f"final_score={current_score:.1f}",
                f"delta={delta:.1f}",
                f"inspector_arguments={inspector_count}",
                f"defense_arguments={defense_count}",
            ],
            confidence_adjustment=0.0,
        )

        return judge_arg, verdict

    # ─── Wording Guidance ─────────────────────────────────────────────────────

    def _generate_wording_guidance(self, verdict: DebateVerdict, score: float) -> str:
        """Generate guidance for report wording based on verdict."""
        guidance_map = {
            DebateVerdict.CONFIRMED_HIGH_RISK: (
                "Sử dụng ngôn ngữ khẳng định nhưng khách quan: "
                "'Kết quả phân tích cho thấy dấu hiệu rủi ro cao', "
                "'Có cơ sở để đề xuất thanh tra'. "
                "KHÔNG dùng: 'gian lận', 'trốn thuế' khi chưa có kết luận thanh tra chính thức."
            ),
            DebateVerdict.ESCALATE: (
                "Sử dụng ngôn ngữ thận trọng: "
                "'Cần theo dõi và đánh giá thêm', 'Có một số dấu hiệu cần làm rõ'. "
                "Tránh kết luận dứt khoát."
            ),
            DebateVerdict.DOWNGRADED: (
                "Sử dụng ngôn ngữ trung lập: "
                "'Sau khi xem xét đa chiều, mức rủi ro được điều chỉnh giảm'. "
                "Ghi nhận các yếu tố giảm nhẹ."
            ),
            DebateVerdict.INSUFFICIENT_EVIDENCE: (
                "Sử dụng ngôn ngữ mở: "
                "'Chưa đủ cơ sở để kết luận', 'Cần thu thập thêm dữ liệu'. "
                "Liệt kê rõ những dữ liệu còn thiếu."
            ),
            DebateVerdict.NO_ACTION: (
                "Sử dụng ngôn ngữ tích cực: "
                "'Không phát hiện dấu hiệu bất thường đáng kể', "
                "'Tiếp tục giám sát theo quy trình thường kỳ'."
            ),
        }
        return guidance_map.get(verdict, "")
