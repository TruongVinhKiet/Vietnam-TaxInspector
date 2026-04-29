"""
tax_agent_synthesis.py – Grounded Synthesis Agent (Phase 2)
============================================================
Generates grounded, cited responses from evidence collected by tools.

Architecture:
    Tier 1: Template-based synthesis (deterministic, auditable)
    Tier 2: Custom LLM synthesis (future — user wants own model)

Principles:
    - Every claim must have evidence backing
    - Inline citations [1][2][3] linked to sources
    - Faithfulness check: no hallucination allowed
    - Structured output: summary + analysis + recommendations
    - Vietnamese language output for tax inspectors
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """A piece of evidence from tool execution."""
    source_tool: str
    source_type: str          # "legal", "analytics", "investigation"
    content: str
    title: str = ""
    score: float = 0.0
    citation_key: str = ""     # e.g., "[1]"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """The final synthesized response."""
    summary: str                        # 1-2 sentence conclusion
    detailed_analysis: str              # Full analysis with inline citations
    evidence: list[Evidence]            # All evidence used
    recommendations: list[str]          # Action items
    confidence: float                   # Overall confidence (0-1)
    limitations: str                    # Known limitations
    escalation_needed: bool             # Whether human review is required
    intent: str
    tools_used: list[str]
    reasoning_trace: str                # CoT from planner
    latency_ms: float = 0.0
    synthesis_tier: str = "template"    # "template" or "llm"
    citation_map: dict[str, str] = field(default_factory=dict)  # [1] → source


class TaxAgentSynthesizer:
    """
    Grounded synthesis engine for tax intelligence responses.

    Current: Template-based (Tier 1)
    Future: Custom LLM (Tier 2) — architecture ready for drop-in replacement.

    Usage:
        synthesizer = TaxAgentSynthesizer()
        result = synthesizer.synthesize(
            query="...",
            intent="vat_refund_risk",
            tool_results={...},
            reasoning_trace="...",
        )
    """

    # ─── Intent-specific response templates ────────────────────────────────
    INTENT_TEMPLATES: dict[str, dict[str, str]] = {
        "vat_refund_risk": {
            "summary_prefix": "Đánh giá rủi ro hoàn thuế VAT",
            "section_header": "Phân tích hồ sơ hoàn thuế",
        },
        "invoice_risk": {
            "summary_prefix": "Đánh giá rủi ro hóa đơn",
            "section_header": "Phân tích bất thường hóa đơn",
        },
        "delinquency": {
            "summary_prefix": "Dự báo rủi ro nợ đọng thuế",
            "section_header": "Phân tích lịch sử tuân thủ",
        },
        "osint_ownership": {
            "summary_prefix": "Phân tích cấu trúc sở hữu",
            "section_header": "Kết quả điều tra sở hữu/UBO",
        },
        "transfer_pricing": {
            "summary_prefix": "Đánh giá rủi ro chuyển giá",
            "section_header": "Phân tích giao dịch liên kết",
        },
        "audit_selection": {
            "summary_prefix": "Đánh giá ưu tiên thanh tra",
            "section_header": "Xếp hạng rủi ro thanh tra",
        },
        "general_tax_query": {
            "summary_prefix": "Tra cứu quy định thuế",
            "section_header": "Căn cứ pháp lý",
        },
    }

    def synthesize(
        self,
        query: str,
        intent: str,
        tool_results: dict[str, dict[str, Any]],
        *,
        reasoning_trace: str = "",
        abstained: bool = False,
        escalate: bool = False,
        tax_code: str | None = None,
    ) -> SynthesisResult:
        """
        Synthesize a grounded response from tool results.

        Args:
            query: Original user query
            intent: Classified intent
            tool_results: Results from tool execution {tool_name: result_dict}
            reasoning_trace: CoT from planner
            abstained: Whether the agent should abstain
            escalate: Whether escalation is needed
            tax_code: Active tax code

        Returns:
            SynthesisResult with structured response
        """
        t0 = time.perf_counter()

        if abstained:
            return self._build_abstain_response(query, intent, reasoning_trace, t0)

        # 1. Extract evidence from all tool results
        evidence = self._extract_evidence(tool_results)

        # 2. Build citation map
        citation_map = {}
        for i, ev in enumerate(evidence):
            ev.citation_key = f"[{i + 1}]"
            citation_map[ev.citation_key] = f"{ev.source_tool}: {ev.title}"

        # 3. Generate summary
        summary = self._generate_summary(intent, evidence, tax_code)

        # 4. Generate detailed analysis with citations
        detailed = self._generate_detailed_analysis(
            intent, evidence, tool_results, tax_code,
        )

        # 5. Generate recommendations
        recommendations = self._generate_recommendations(
            intent, evidence, tool_results, tax_code,
        )

        # 6. Assess confidence
        confidence = self._assess_confidence(evidence, tool_results)

        # 7. Identify limitations
        limitations = self._identify_limitations(tool_results, evidence)

        # 8. Check if escalation is needed
        escalation_needed = escalate or confidence < 0.3

        tools_used = list(tool_results.keys())
        latency = (time.perf_counter() - t0) * 1000.0

        return SynthesisResult(
            summary=summary,
            detailed_analysis=detailed,
            evidence=evidence,
            recommendations=recommendations,
            confidence=confidence,
            limitations=limitations,
            escalation_needed=escalation_needed,
            intent=intent,
            tools_used=tools_used,
            reasoning_trace=reasoning_trace,
            latency_ms=latency,
            synthesis_tier="template",
            citation_map=citation_map,
        )

    def _extract_evidence(
        self,
        tool_results: dict[str, dict[str, Any]],
    ) -> list[Evidence]:
        """Extract structured evidence from tool results."""
        evidence: list[Evidence] = []

        # Knowledge search results → legal evidence
        ks = tool_results.get("knowledge_search", {})
        for hit in ks.get("hits", []):
            evidence.append(Evidence(
                source_tool="knowledge_search",
                source_type="legal",
                content=str(hit.get("text", "")),
                title=str(hit.get("title", "")),
                score=float(hit.get("score", 0)),
                metadata={"chunk_key": hit.get("chunk_key"), "doc_type": hit.get("doc_type")},
            ))

        # Company risk → analytics evidence
        cr = tool_results.get("company_risk_lookup", {})
        if cr.get("status") == "found":
            evidence.append(Evidence(
                source_tool="company_risk_lookup",
                source_type="analytics",
                content=(
                    f"Doanh nghiệp {cr.get('company_name', '')} (MST: {cr.get('tax_code', '')}) — "
                    f"Điểm rủi ro: {cr.get('risk_score', 0)}/100, "
                    f"Mức rủi ro: {cr.get('risk_level', 'N/A')}, "
                    f"Ngành: {cr.get('industry', 'N/A')}"
                ),
                title=f"Hồ sơ rủi ro {cr.get('company_name', '')}",
                score=float(cr.get("risk_score", 0)) / 100.0,
            ))

        # Delinquency → analytics evidence
        dq = tool_results.get("delinquency_check", {})
        if dq.get("status") == "analyzed":
            reasons_text = ", ".join(
                r.get("reason", "") for r in dq.get("top_reasons", [])[:3]
            )
            evidence.append(Evidence(
                source_tool="delinquency_check",
                source_type="analytics",
                content=(
                    f"Dự báo nợ đọng — P(30d): {dq.get('prob_30d', 0):.1%}, "
                    f"P(60d): {dq.get('prob_60d', 0):.1%}, "
                    f"P(90d): {dq.get('prob_90d', 0):.1%}. "
                    f"Phân cụm: {dq.get('cluster', 'N/A')}. "
                    f"Lý do chính: {reasons_text}"
                ),
                title="Dự báo nợ đọng thuế",
                score=float(dq.get("prob_90d", 0)),
            ))

        # Invoice risk → analytics evidence
        ir = tool_results.get("invoice_risk_scan", {})
        if ir.get("status") == "analyzed":
            evidence.append(Evidence(
                source_tool="invoice_risk_scan",
                source_type="analytics",
                content=(
                    f"Hóa đơn: tổng {ir.get('total_invoices', 0)}, "
                    f"rủi ro {ir.get('risky_invoices', 0)} "
                    f"({ir.get('risk_ratio', 0):.1%}). "
                    f"Tổng giá trị rủi ro: {ir.get('risky_amount', 0):,.0f} VND"
                ),
                title="Phân tích rủi ro hóa đơn",
                score=float(ir.get("risk_ratio", 0)),
            ))

        # GNN analysis → investigation evidence
        gnn = tool_results.get("gnn_analysis", {})
        if gnn.get("status") == "found":
            outputs = gnn.get("gnn_outputs", {})
            evidence.append(Evidence(
                source_tool="gnn_analysis",
                source_type="investigation",
                content=(
                    f"GNN phát hiện: {json.dumps(outputs, ensure_ascii=False, default=str)[:300]}"
                ),
                title="Phân tích GNN đồ thị giao dịch",
                score=float(outputs.get("risk_probability", 0)),
            ))

        # Motif detection → investigation evidence
        motif = tool_results.get("motif_detection", {})
        if motif.get("status") == "analyzed":
            summary = motif.get("summary", {})
            evidence.append(Evidence(
                source_tool="motif_detection",
                source_type="investigation",
                content=(
                    f"Mẫu phát hiện: {summary.get('total_triangles', 0)} vòng tròn, "
                    f"{summary.get('total_stars', 0)} hình sao, "
                    f"{summary.get('total_chains', 0)} chuỗi, "
                    f"{summary.get('total_fan_out', 0)} fan-out, "
                    f"{summary.get('total_fan_in', 0)} fan-in"
                ),
                title="Phát hiện mẫu giao dịch đáng ngờ",
                score=min(1.0, sum(summary.values()) / 10.0) if summary else 0.0,
            ))

        # Ownership analysis → investigation evidence
        own = tool_results.get("ownership_analysis", {})
        if own.get("status") == "analyzed":
            own_summary = own.get("summary", {})
            evidence.append(Evidence(
                source_tool="ownership_analysis",
                source_type="investigation",
                content=(
                    f"Sở hữu: {own_summary.get('total_clusters', 0)} cụm, "
                    f"{own_summary.get('total_common_controllers', 0)} common controllers, "
                    f"{own_summary.get('total_cross_trades', 0)} giao dịch nội bộ"
                ),
                title="Phân tích cấu trúc sở hữu",
                score=min(1.0, own_summary.get("total_cross_trades", 0) / 5.0),
            ))

        return evidence

    def _generate_summary(
        self,
        intent: str,
        evidence: list[Evidence],
        tax_code: str | None,
    ) -> str:
        """Generate a concise 1-2 sentence summary."""
        template = self.INTENT_TEMPLATES.get(intent, self.INTENT_TEMPLATES["general_tax_query"])
        prefix = template["summary_prefix"]

        entity_str = f" cho MST {tax_code}" if tax_code else ""

        # Determine overall risk level from evidence
        analytics_scores = [e.score for e in evidence if e.source_type == "analytics"]
        avg_risk = sum(analytics_scores) / max(len(analytics_scores), 1)

        if avg_risk > 0.7:
            risk_label = "RỦI RO CAO"
        elif avg_risk > 0.4:
            risk_label = "RỦI RO TRUNG BÌNH"
        elif avg_risk > 0:
            risk_label = "RỦI RO THẤP"
        else:
            risk_label = "CHƯA XÁC ĐỊNH"

        legal_count = sum(1 for e in evidence if e.source_type == "legal")
        analytics_count = sum(1 for e in evidence if e.source_type == "analytics")

        summary = f"{prefix}{entity_str}: {risk_label}."
        if legal_count > 0:
            summary += f" Có {legal_count} căn cứ pháp lý liên quan."
        if analytics_count > 0:
            summary += f" {analytics_count} chỉ số phân tích đã được kiểm tra."

        return summary

    def _generate_detailed_analysis(
        self,
        intent: str,
        evidence: list[Evidence],
        tool_results: dict[str, dict[str, Any]],
        tax_code: str | None,
    ) -> str:
        """Generate detailed analysis with inline citations."""
        template = self.INTENT_TEMPLATES.get(intent, self.INTENT_TEMPLATES["general_tax_query"])
        section_header = template["section_header"]

        parts = [f"## {section_header}\n"]

        # Legal basis section
        legal_evidence = [e for e in evidence if e.source_type == "legal"]
        if legal_evidence:
            parts.append("### Căn cứ pháp lý")
            for ev in legal_evidence[:3]:
                parts.append(
                    f"- **{ev.title}** {ev.citation_key}: {ev.content[:250]}..."
                )

        # Analytics section
        analytics_evidence = [e for e in evidence if e.source_type == "analytics"]
        if analytics_evidence:
            parts.append("\n### Kết quả phân tích")
            for ev in analytics_evidence:
                parts.append(f"- {ev.content} {ev.citation_key}")

        # Investigation section
        investigation_evidence = [e for e in evidence if e.source_type == "investigation"]
        if investigation_evidence:
            parts.append("\n### Kết quả điều tra")
            for ev in investigation_evidence:
                parts.append(f"- {ev.content} {ev.citation_key}")

        if not evidence:
            parts.append(
                "Chưa đủ dữ liệu để phân tích chi tiết. "
                "Vui lòng cung cấp thêm thông tin (MST, kỳ thuế)."
            )

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        intent: str,
        evidence: list[Evidence],
        tool_results: dict[str, dict[str, Any]],
        tax_code: str | None,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Intent-specific recommendations
        analytics_scores = [e.score for e in evidence if e.source_type == "analytics"]
        avg_risk = sum(analytics_scores) / max(len(analytics_scores), 1)

        if avg_risk > 0.7:
            recommendations.append(
                "⚠️ Đề xuất ưu tiên thanh tra/kiểm tra doanh nghiệp này."
            )

        # Delinquency-specific
        dq = tool_results.get("delinquency_check", {})
        if dq.get("prob_90d", 0) > 0.5:
            recommendations.append(
                f"📋 Rủi ro nợ đọng 90 ngày: {dq['prob_90d']:.0%} — "
                f"cần theo dõi sát tình hình nộp thuế."
            )

        # Invoice-specific
        ir = tool_results.get("invoice_risk_scan", {})
        if ir.get("risk_ratio", 0) > 0.2:
            recommendations.append(
                f"🔍 Tỷ lệ hóa đơn rủi ro: {ir['risk_ratio']:.0%} — "
                f"cần rà soát hóa đơn đầu vào."
            )

        # Ownership-specific
        own = tool_results.get("ownership_analysis", {})
        if own.get("summary", {}).get("total_cross_trades", 0) > 0:
            recommendations.append(
                "🔗 Phát hiện giao dịch nội bộ giữa các đơn vị liên kết — "
                "cần kiểm tra giá chuyển giao."
            )

        # Motif-specific
        motif = tool_results.get("motif_detection", {})
        if motif.get("summary", {}).get("total_triangles", 0) > 0:
            recommendations.append(
                "⭕ Phát hiện mẫu giao dịch vòng tròn (carousel) — "
                "cần điều tra chi tiết gian lận VAT."
            )

        # Always add a general recommendation
        recommendations.append(
            "📌 Đối chiếu thêm số liệu nghiệp vụ trước khi ra quyết định cuối cùng."
        )

        return recommendations

    def _assess_confidence(
        self,
        evidence: list[Evidence],
        tool_results: dict[str, dict[str, Any]],
    ) -> float:
        """Assess overall response confidence."""
        if not evidence:
            return 0.1

        factors = []

        # Evidence quantity factor
        quantity_score = min(1.0, len(evidence) / 5.0)
        factors.append(quantity_score * 0.3)

        # Evidence quality factor (average score)
        scores = [e.score for e in evidence if e.score > 0]
        if scores:
            quality_score = sum(scores) / len(scores)
            factors.append(quality_score * 0.3)

        # Source diversity factor
        source_types = set(e.source_type for e in evidence)
        diversity_score = len(source_types) / 3.0
        factors.append(min(1.0, diversity_score) * 0.2)

        # Tool success rate factor
        total_tools = len(tool_results)
        successful = sum(
            1 for r in tool_results.values()
            if r.get("status") not in ("error", "no_data", "not_found", None)
        )
        success_rate = successful / max(total_tools, 1)
        factors.append(success_rate * 0.2)

        return round(min(1.0, sum(factors)), 4)

    def _identify_limitations(
        self,
        tool_results: dict[str, dict[str, Any]],
        evidence: list[Evidence],
    ) -> str:
        """Identify limitations of the analysis."""
        limitations = []

        # Check for missing data
        for tool_name, result in tool_results.items():
            if result.get("status") in ("no_data", "not_found"):
                limitations.append(f"Thiếu dữ liệu từ {tool_name}")
            if result.get("status") == "error":
                limitations.append(f"Lỗi khi truy vấn {tool_name}")

        # Check evidence quality
        low_quality = [e for e in evidence if e.score < 0.3 and e.source_type == "legal"]
        if low_quality:
            limitations.append("Một số căn cứ pháp lý có độ liên quan thấp")

        if not evidence:
            limitations.append("Không tìm được evidence hỗ trợ phân tích")

        return "; ".join(limitations) if limitations else "Không có giới hạn đáng kể."

    def _build_abstain_response(
        self,
        query: str,
        intent: str,
        reasoning_trace: str,
        t0: float,
    ) -> SynthesisResult:
        """Build a response when the agent abstains."""
        return SynthesisResult(
            summary=(
                "Tôi chưa đủ độ tin cậy để kết luận ngay từ dữ liệu hiện có."
            ),
            detailed_analysis=(
                "Để đảm bảo chất lượng tư vấn, hệ thống cần thêm thông tin:\n"
                "- Mã số thuế (MST) cụ thể\n"
                "- Kỳ thuế cần tra cứu\n"
                "- Loại hồ sơ/sắc thuế\n\n"
                "Vui lòng cung cấp thêm bối cảnh hoặc chuyển hồ sơ cho chuyên viên."
            ),
            evidence=[],
            recommendations=[
                "Cung cấp thêm thông tin cụ thể (MST, kỳ thuế).",
                "Hoặc chuyển cho chuyên viên để xác minh.",
            ],
            confidence=0.0,
            limitations="Thiếu thông tin đầu vào để phân tích.",
            escalation_needed=True,
            intent=intent,
            tools_used=[],
            reasoning_trace=reasoning_trace,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            synthesis_tier="template",
        )

    def format_response_text(self, result: SynthesisResult) -> str:
        """Format SynthesisResult into a single text response for the chat API."""
        parts = []

        # Summary (bold)
        parts.append(f"**{result.summary}**\n")

        # Detailed analysis
        parts.append(result.detailed_analysis)

        # Recommendations
        if result.recommendations:
            parts.append("\n### Khuyến nghị")
            for rec in result.recommendations:
                parts.append(f"- {rec}")

        # Confidence indicator
        conf_bar = "█" * int(result.confidence * 10)
        conf_empty = "░" * (10 - int(result.confidence * 10))
        parts.append(
            f"\n---\n_Độ tin cậy: {conf_bar}{conf_empty} {result.confidence:.0%} "
            f"| Công cụ: {', '.join(result.tools_used)} "
            f"| Tier: {result.synthesis_tier}_"
        )

        # Limitations
        if result.limitations and "Không có" not in result.limitations:
            parts.append(f"\n⚠️ _Giới hạn: {result.limitations}_")

        return "\n".join(parts)
