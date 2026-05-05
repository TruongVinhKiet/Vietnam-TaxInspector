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
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ml_engine.tax_agent_legal_intelligence import (
    LegalFaithfulnessVerifier,
    LegalSlotAnalyzer,
)

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
    verification: dict[str, Any] = field(default_factory=dict)
    clarification_needed: bool = False
    clarification_questions: list[str] = field(default_factory=list)
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
        "top_n_query": {
            "summary_prefix": "Danh sách doanh nghiệp rủi ro",
            "section_header": "Kết quả xếp hạng rủi ro",
        },
        "company_name_lookup": {
            "summary_prefix": "Kết quả tra cứu doanh nghiệp",
            "section_header": "Thông tin doanh nghiệp tìm thấy",
        },
        "batch_analysis": {
            "summary_prefix": "Kết quả phân tích lô",
            "section_header": "Tổng hợp phân tích batch",
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
        answer_contract: str | None = None,
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

        if answer_contract == "data_table" and intent == "top_n_query":
            return self._build_top_n_table_response(
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                t0=t0,
            )

        # 1. Extract evidence from all tool results
        evidence = self._extract_evidence(tool_results)

        # 2. Build citation map
        citation_map = {}
        for i, ev in enumerate(evidence):
            ev.citation_key = f"[{i + 1}]"
            citation_map[ev.citation_key] = f"{ev.source_tool}: {ev.title}"

        missing_slots: list[str] = []
        slot_analyzer = LegalSlotAnalyzer()
        is_legal_consultation = self._is_legal_consultation_intent(
            intent, evidence, answer_contract=answer_contract,
        )
        if is_legal_consultation:
            missing_slots = slot_analyzer.missing_slots(query, intent=intent)
            legal_hits = [ev for ev in evidence if ev.source_type == "legal"]
            if len(missing_slots) >= 3 and not legal_hits:
                return self._build_clarification_response(
                    query=query,
                    intent=intent,
                    reasoning_trace=reasoning_trace,
                    t0=t0,
                    missing_slots=missing_slots,
                    prompt=slot_analyzer.clarification_prompt(missing_slots),
                )

        # 3. Generate summary
        summary = self._generate_summary(intent, evidence, tax_code)

        synthesis_tier = "template"

        # 4. Generate detailed analysis with citations
        if is_legal_consultation:
            llm_text, llm_tier = self._try_llm_legal_synthesis(
                query=query,
                intent=intent,
                evidence=evidence,
                tool_results=tool_results,
            )
            if llm_text:
                detailed = llm_text
                synthesis_tier = llm_tier
            else:
                detailed = self._generate_grounded_legal_consultation(
                    intent, evidence, tool_results, tax_code,
                )
        else:
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

        verification = self._verify_synthesis(detailed, evidence)
        if verification.get("requires_abstain"):
            detailed = self._generate_grounded_legal_consultation(
                intent, evidence, tool_results, tax_code,
                verification=verification,
            )
            synthesis_tier = "template_verified_fallback"
            verification = self._verify_synthesis(detailed, evidence)
        if verification.get("status") == "review":
            confidence = round(max(0.05, confidence * 0.72), 4)
            limitations = (
                f"{limitations}; một số kết luận cần kiểm chứng trích dẫn"
                if limitations else
                "Một số kết luận cần kiểm chứng trích dẫn"
            )

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
            synthesis_tier=synthesis_tier,
            citation_map=citation_map,
            verification=verification,
            clarification_needed=bool(missing_slots),
            clarification_questions=missing_slots,
        )

    def _build_top_n_table_response(
        self,
        *,
        tool_results: dict[str, dict[str, Any]],
        reasoning_trace: str,
        t0: float,
    ) -> SynthesisResult:
        top_n = tool_results.get("top_n_risky_companies", {}) or {}
        companies = list(top_n.get("companies") or [])
        requested = int(top_n.get("query_n") or len(companies) or 10)
        total = int(top_n.get("total") or len(companies) or 0)

        if not companies:
            summary = "Chưa có dữ liệu chấm điểm rủi ro để lập danh sách top doanh nghiệp."
            detailed = (
                "Hệ thống chưa tìm thấy bản ghi có `risk_score` trong CSDL. "
                "Hãy chạy batch scoring hoặc upload CSV trước, sau đó truy vấn lại top doanh nghiệp rủi ro."
            )
            confidence = 0.45
        else:
            found_note = (
                f" Tìm thấy {len(companies)}/{requested} doanh nghiệp theo yêu cầu."
                if len(companies) < requested else
                f" Tìm thấy {len(companies)} doanh nghiệp rủi ro cao nhất."
            )
            summary = f"Danh sách top {min(requested, len(companies))} doanh nghiệp rủi ro cao nhất.{found_note}"
            lines = [
                "| STT | MST | Tên DN | Ngành | Điểm rủi ro | Mức | Năm |",
                "|---:|---|---|---|---:|---|---:|",
            ]
            for item in companies:
                lines.append(
                    "| {stt} | {tax_code} | {company_name} | {industry} | {risk_score} | {risk_level} | {year} |".format(
                        stt=item.get("stt", ""),
                        tax_code=item.get("tax_code", ""),
                        company_name=str(item.get("company_name", "")).replace("|", "/"),
                        industry=str(item.get("industry", "")).replace("|", "/"),
                        risk_score=item.get("risk_score", 0),
                        risk_level=item.get("risk_level", ""),
                        year=item.get("year", ""),
                    )
                )
            detailed = "\n".join(lines)
            if total and total > len(companies):
                detailed += f"\n\nTổng số doanh nghiệp có điểm trong CSDL: {total}."
            confidence = 0.9

        return SynthesisResult(
            summary=summary,
            detailed_analysis=detailed,
            evidence=[],
            recommendations=[],
            confidence=confidence,
            limitations="",
            escalation_needed=False,
            intent="top_n_query",
            tools_used=["top_n_risky_companies"] if top_n else [],
            reasoning_trace=reasoning_trace,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            synthesis_tier="data_table",
            verification={"status": "not_required", "reason": "direct_data_table"},
            citation_map={},
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
                metadata={
                    "chunk_key": hit.get("chunk_key"),
                    "doc_type": hit.get("doc_type"),
                    "citation_spans": hit.get("citation_spans", []),
                    "authority_path": hit.get("authority_path", []),
                    "effective_status": hit.get("effective_status", {}),
                    "official_letter_scope": hit.get("official_letter_scope", {}),
                    "relation_path": hit.get("relation_path", []),
                    "legal_metadata": hit.get("legal_metadata", {}),
                    "full_text": hit.get("full_text", ""),
                },
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

    def _is_legal_consultation_intent(
        self,
        intent: str,
        evidence: list[Evidence],
        *,
        answer_contract: str | None = None,
    ) -> bool:
        if answer_contract == "legal_consultation":
            return True
        if answer_contract in {"data_table", "smalltalk", "file_analysis", "vat_graph"}:
            return False
        if intent in {"general_tax_query", "vat_refund_risk", "invoice_risk", "transfer_pricing"}:
            return True
        return any(ev.source_type == "legal" for ev in evidence)

    def _evidence_dicts(self, evidence: list[Evidence]) -> list[dict[str, Any]]:
        return [
            {
                "source_tool": ev.source_tool,
                "source_type": ev.source_type,
                "content": ev.content,
                "title": ev.title,
                "score": ev.score,
                "citation_key": ev.citation_key,
                **(ev.metadata or {}),
            }
            for ev in evidence
        ]

    def _try_llm_legal_synthesis(
        self,
        *,
        query: str,
        intent: str,
        evidence: list[Evidence],
        tool_results: dict[str, dict[str, Any]],
    ) -> tuple[str | None, str]:
        """Use local TaxAgentLLM when an adapter/base is explicitly available."""
        enable = os.getenv("TAX_AGENT_ENABLE_LLM", "").strip().lower() in {"1", "true", "yes"}
        adapter = os.getenv("TAX_AGENT_LLM_ADAPTER")
        default_adapter = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "models",
            "tax_llm_lora",
        )
        has_adapter = os.path.exists(os.path.join(adapter or default_adapter, "adapter_model.safetensors"))
        if not enable and not has_adapter:
            return None, "template"

        legal_evidence = [ev for ev in evidence if ev.source_type == "legal"]
        if not legal_evidence:
            return None, "template"

        context_parts = [
            f"{ev.citation_key} {ev.title}: {ev.content[:700]}"
            for ev in legal_evidence[:5]
        ]
        try:
            from ml_engine.tax_agent_llm_model import get_tax_llm

            llm = get_tax_llm()
            response = llm.generate(
                query=query,
                context="\n".join(context_parts),
                intent=intent,
                evidence=self._evidence_dicts(legal_evidence),
                max_new_tokens=420,
            )
            if response.tier.value == "template" or not response.text.strip():
                return None, "template"
            return response.text.strip(), f"llm_{response.tier.value}"
        except Exception as exc:
            logger.debug("[Synthesis] local LLM unavailable, using template: %s", exc)
            return None, "template"

    def _generate_grounded_legal_consultation(
        self,
        intent: str,
        evidence: list[Evidence],
        tool_results: dict[str, dict[str, Any]],
        tax_code: str | None,
        *,
        verification: dict[str, Any] | None = None,
    ) -> str:
        legal_evidence = [e for e in evidence if e.source_type == "legal"]
        parts = ["## Tư vấn pháp lý có căn cứ\n"]
        if legal_evidence:
            strongest = max(legal_evidence, key=lambda ev: ev.score)
            parts.append(
                f"**Kết luận ngắn:** Có căn cứ liên quan trong {strongest.title} {strongest.citation_key}. "
                "Cần đối chiếu hồ sơ thực tế trước khi áp dụng."
            )
            parts.append("\n### Điều kiện áp dụng")
            for ev in legal_evidence[:4]:
                metadata = ev.metadata or {}
                effective = metadata.get("effective_status") or {}
                official_scope = metadata.get("official_letter_scope") or {}
                state = effective.get("dominant_state") or effective.get("state") or "chưa xác định"
                state_vi = {
                    "active": "còn hiệu lực",
                    "expired": "hết hiệu lực",
                    "pending": "chờ hiệu lực",
                    "unknown": "chưa xác định",
                }.get(state, state)
                scope_note = ""
                if official_scope.get("has_official_letter") or official_scope.get("is_official_letter"):
                    scope_note = " **Lưu ý:** Công văn chỉ có giá trị hướng dẫn theo phạm vi cụ thể."
                parts.append(
                    f"- {ev.title} {ev.citation_key}: {ev.content[:260]}... "
                    f"Hiệu lực: **{state_vi}**.{scope_note}"
                )

            parts.append("\n### Căn cứ và chuỗi quan hệ pháp lý")
            graph_context = tool_results.get("knowledge_search", {}).get("graph_context") or {}
            authority_path = graph_context.get("authority_path") or []
            if authority_path:
                for item in authority_path[:5]:
                    entity_type_vi = {
                        "law": "Luật",
                        "decree": "Nghị định",
                        "circular": "Thông tư",
                        "decision": "Quyết định",
                        "official_letter": "Công văn",
                        "article": "Điều",
                        "clause": "Khoản",
                    }.get(item.get("entity_type", ""), item.get("entity_type", ""))
                    parts.append(
                        f"- {item.get('display_name')} "
                        f"({entity_type_vi}, thẩm quyền={item.get('authority_rank')})"
                    )
            else:
                for ev in legal_evidence[:3]:
                    parts.append(f"- {ev.title} {ev.citation_key}")

            parts.append("\n### Ngoại lệ / Rủi ro pháp lý")
            official_rollup = graph_context.get("official_letter_scope") or {}
            if official_rollup.get("warnings"):
                for warning in official_rollup["warnings"][:3]:
                    warning_vi = warning
                    if "Official letters are guidance" in warning:
                        warning_vi = (
                            "Công văn chỉ mang tính chất hướng dẫn cho trường hợp cụ thể hoặc giải thích hành chính; "
                            "không được coi là văn bản quy phạm pháp luật có hiệu lực cao hơn."
                        )
                    parts.append(f"- {warning_vi}")
            effective_rollup = graph_context.get("effective_status") or {}
            if effective_rollup.get("has_non_usable"):
                parts.append(
                    "- Có văn bản hết hiệu lực hoặc chờ hiệu lực trong chuỗi trích dẫn; "
                    "cần ưu tiên áp dụng văn bản còn hiệu lực."
                )
            if verification and verification.get("unsupported_claims"):
                parts.append(
                    "- Đã hạ mức câu trả lời về template có căn cứ vì bộ xác minh phát hiện "
                    "một số lập luận chưa được trích dẫn hỗ trợ."
                )
        else:
            parts.append("Chưa có trích dẫn pháp lý đủ mạnh để đưa ra kết luận.")

        parts.append("\n### Bước xử lý tiếp theo")
        parts.append("- Xác định kỳ thuế, ngày chứng từ và loại giao dịch cụ thể.")
        parts.append(
            "- Ưu tiên văn bản theo thứ tự hiệu lực: "
            "**Luật → Nghị định → Thông tư**; Công văn chỉ là hướng dẫn theo phạm vi."
        )
        return "\n".join(parts)

    def _verify_synthesis(self, detailed: str, evidence: list[Evidence]) -> dict[str, Any]:
        verifier = LegalFaithfulnessVerifier()
        return verifier.verify(answer_text=detailed, evidence=self._evidence_dicts(evidence))

    def _build_clarification_response(
        self,
        *,
        query: str,
        intent: str,
        reasoning_trace: str,
        t0: float,
        missing_slots: list[str],
        prompt: str,
    ) -> SynthesisResult:
        return SynthesisResult(
            summary="Cần bổ sung thông tin trước khi tư vấn pháp lý.",
            detailed_analysis=prompt,
            evidence=[],
            recommendations=["Bổ sung kỳ thuế/ngày chứng từ, loại người nộp thuế và loại giao dịch."],
            confidence=0.0,
            limitations="Thiếu thông tin bắt buộc để xác định văn bản và phạm vi áp dụng.",
            escalation_needed=False,
            intent=intent,
            tools_used=[],
            reasoning_trace=reasoning_trace,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            synthesis_tier="clarification",
            verification={"status": "clarification", "missing_slots": missing_slots},
            clarification_needed=True,
            clarification_questions=missing_slots,
        )

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

        verification = result.verification or {}
        if verification and verification.get("status") in {"review", "clarification"}:
            parts.append(
                "\n_Verifier: "
                f"{verification.get('status')} "
                f"(faithfulness={verification.get('faithfulness_score', 'n/a')})._"
            )

        # Limitations
        if result.limitations and "Không có" not in result.limitations:
            parts.append(f"\n⚠️ _Giới hạn: {result.limitations}_")

        return "\n".join(parts)
