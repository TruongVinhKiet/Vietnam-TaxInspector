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

        # Batch analysis from uploaded file — generate rich summary
        if answer_contract == "fraud_analysis" and ("_batch_results" in tool_results or "_attachment_analysis" in tool_results):
            return self._build_fraud_analysis_response(
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                t0=t0,
            )

        if answer_contract == "vat_graph" and (
            "_vat_graph_batch_results" in tool_results
            or "_ocr_document_results" in tool_results
            or "_attachment_analysis" in tool_results
        ):
            return self._build_vat_graph_response(
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                t0=t0,
            )

        if intent == "batch_analysis" and ("_batch_results" in tool_results or "_attachment_analysis" in tool_results):
            return self._build_batch_analysis_response(
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
            detailed = "*(Danh sách chi tiết đã được hiển thị trong bảng giao diện bên dưới. Bạn có thể nhấn vào từng dòng để xem báo cáo rủi ro cho riêng doanh nghiệp đó.)*"
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

    def _build_fraud_analysis_response(
        self,
        *,
        tool_results: dict[str, dict[str, Any]],
        reasoning_trace: str,
        t0: float,
    ) -> SynthesisResult:
        batch = tool_results.get("_batch_results") or tool_results.get("_attachment_analysis") or {}
        companies = list(batch.get("assessments", []) or batch.get("companies", []) or [])
        top_risky = list(batch.get("top_5") or batch.get("top_risky") or companies[:10])
        total = int(batch.get("total") or len(companies) or 0)
        by_level = batch.get("by_level") or {}
        filename = batch.get("filename") or "CSV"
        if not total and batch.get("status") == "error":
            return SynthesisResult(
                summary=f"File {filename} chưa thể phân tích rủi ro.",
                detailed_analysis=f"Schema hoặc dữ liệu đầu vào chưa phù hợp. Lỗi: {batch.get('error', 'unknown')}",
                evidence=[],
                recommendations=["Kiểm tra lại contract CSV risk scoring và upload lại."],
                confidence=0.35,
                limitations="Không đủ dữ liệu để chạy TaxFraudPipeline.",
                escalation_needed=False,
                intent="batch_analysis",
                tools_used=["_batch_results"],
                reasoning_trace=reasoning_trace,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                synthesis_tier="fraud_analysis_v2",
                verification={"status": "not_required", "reason": "model_output"},
                citation_map={},
            )

        highest = 0.0
        for row in top_risky:
            try:
                highest = max(highest, float(row.get("risk_score") or 0.0))
            except Exception:
                pass
        high_count = int(by_level.get("critical", 0) or by_level.get("very_high", 0) or 0) + int(by_level.get("high", 0) or 0)
        critical_count = int(by_level.get("critical", by_level.get("very_high", 0)) or 0)
        medium_count = int(by_level.get("medium", 0) or 0)
        low_count = int(by_level.get("low", 0) or 0)
        summary = (
            f"Đã phân tích {total} doanh nghiệp từ {filename}. "
            f"Xác định {high_count} hồ sơ thuộc nhóm ưu tiên thanh tra, điểm rủi ro cao nhất là {highest:.1f}."
        )
        rows = []
        for idx, row in enumerate(top_risky[:10], 1):
            rows.append(
                f"| {idx} | {row.get('tax_code') or row.get('mst') or ''} | "
                f"{row.get('company_name') or row.get('ten_dn') or ''} | "
                f"{row.get('industry') or ''} | {row.get('risk_score') or 0} | "
                f"{row.get('risk_level') or ''} |"
            )
        table = "\n".join(rows) if rows else "| - | - | - | - | - | - |"
        detailed = (
            "### Kết luận điều hành\n"
            f"{summary}\n\n"
            "### Điểm rủi ro và mức cảnh báo\n"
            f"- Nhóm rất cao: {critical_count}\n"
            f"- Nhóm cao: {int(by_level.get('high', 0) or 0)}\n"
            f"- Nhóm trung bình: {medium_count}\n"
            f"- Nhóm thấp/an toàn: {low_count}\n"
            f"- Mức ưu tiên hiện tại: {'cao' if high_count > 0 else 'trung bình'}\n\n"
            "### Top hồ sơ cần ưu tiên\n"
            "| STT | MST | Tên DN | Ngành | Điểm rủi ro | Mức |\n"
            "|---:|---|---|---|---:|---|\n"
            f"{table}\n\n"
            "### Yếu tố rủi ro trọng tâm (F1-F4)\n"
            "- F1 (hồ sơ tuân thủ): biến động nghĩa vụ khai nộp theo kỳ và tần suất điều chỉnh.\n"
            "- F2 (hành vi tài chính): độ lệch doanh thu-biên lợi nhuận so với baseline ngành.\n"
            "- F3 (VAT-hóa đơn): dấu hiệu lệch pha đầu vào/đầu ra và bất thường chuỗi hóa đơn.\n"
            "- F4 (liên kết mạng): mức tập trung giao dịch với đối tác rủi ro và motif vòng.\n\n"
            "### So sánh theo ngành và theo kỳ\n"
            "Phân tích multi-agent ưu tiên đối chiếu cùng ngành/cùng giai đoạn để giảm nhiễu do mùa vụ. "
            "Hồ sơ có điểm cao nhưng bối cảnh ngành đặc thù sẽ được gắn cờ để xác minh thủ công trước khi kết luận.\n\n"
            "### Kết luận nghiệp vụ và độ tin cậy\n"
            "Kết quả được hợp nhất từ pipeline gian lận, lớp phân tích bất thường và lớp điều phối miền nghiệp vụ. "
            "Hệ thống chỉ tập trung miền gian lận khi người dùng không yêu cầu pháp lý, nhằm giảm trả lời lệch trọng tâm. "
            "Độ tin cậy phụ thuộc đầy đủ cột dữ liệu, tính nhất quán mã số thuế và độ bao phủ kỳ phân tích."
        )
        recommendations = [
            "Thiết lập danh sách kiểm tra theo 3 tầng: rất cao, cao, trung bình để phân bổ nguồn lực thanh tra.",
            "Đối chiếu đồng thời doanh thu, VAT, biên lợi nhuận và giao dịch liên kết theo từng kỳ khai thuế.",
            "Kích hoạt phân tích VAT graph và OCR chứng từ cho các hồ sơ có nghi vấn vòng hóa đơn hoặc chuỗi trung gian bất thường.",
            "Yêu cầu giải trình mục tiêu cho các hồ sơ tăng điểm đột ngột so với kỳ trước hoặc lệch chuẩn ngành.",
        ]
        return SynthesisResult(
            summary=summary,
            detailed_analysis=detailed,
            evidence=[],
            recommendations=recommendations,
            confidence=0.88 if total else 0.45,
            limitations="Kết quả phụ thuộc chất lượng CSV và dữ liệu nền hiện có.",
            escalation_needed=highest >= 90,
            intent="batch_analysis",
            tools_used=["_batch_results"],
            reasoning_trace=reasoning_trace,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            synthesis_tier="fraud_analysis_v2",
            verification={"status": "not_required", "reason": "model_output"},
            citation_map={},
        )

    def _build_vat_graph_response(
        self,
        *,
        tool_results: dict[str, dict[str, Any]],
        reasoning_trace: str,
        t0: float,
    ) -> SynthesisResult:
        vat = tool_results.get("_vat_graph_batch_results") or tool_results.get("_attachment_analysis") or {}
        ocr = tool_results.get("_ocr_document_results") or {}
        graph = vat.get("graph") or {}
        summary_data = vat.get("summary") or {}
        processed = vat.get("processed_rows") or vat.get("row_count") or 0
        top_edges = graph.get("top_invoice_risks") or []
        rings = graph.get("rings") or graph.get("ring_findings") or graph.get("motifs") or []
        suspect_value = summary_data.get("suspect_value") or summary_data.get("total_suspicious_amount") or 0
        if ocr and not vat:
            fields = ocr.get("extracted_fields") or {}
            risk = ocr.get("invoice_risk") or {}
            summary = f"Đã OCR chứng từ {ocr.get('filename', '')} và chấm rủi ro hóa đơn."
            detailed = (
                "### Kết luận hóa đơn\n"
                f"{summary}\n\n"
                "### Trường đã trích xuất\n"
                f"- Số hóa đơn: {fields.get('invoice_number') or 'chưa xác định'}\n"
                f"- Người bán: {fields.get('seller_tax_code') or 'chưa xác định'}\n"
                f"- Người mua: {fields.get('buyer_tax_code') or 'chưa xác định'}\n"
                f"- Tổng tiền: {fields.get('total_amount') or fields.get('amount') or 'chưa xác định'}\n\n"
                "### Rủi ro chứng từ\n"
                f"Điểm rủi ro: {risk.get('risk_score', risk.get('score', 'N/A'))}. "
                "Nên liên kết chứng từ này vào VAT graph nếu có dữ liệu giao dịch đối ứng."
            )
            return SynthesisResult(
                summary=summary,
                detailed_analysis=detailed,
                evidence=[],
                recommendations=["Đối chiếu MST hai bên và đưa hóa đơn vào graph để kiểm tra vòng giao dịch."],
                confidence=float(ocr.get("confidence") or 0.75),
                limitations="OCR có thể sai nếu ảnh/PDF chất lượng thấp.",
                escalation_needed=False,
                intent="invoice_risk",
                tools_used=["_ocr_document_results"],
                reasoning_trace=reasoning_trace,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                synthesis_tier="vat_graph_v2",
                verification={"status": "not_required", "reason": "model_output"},
                citation_map={},
            )

        summary = (
            f"Đã xử lý {processed} giao dịch VAT. "
            f"Phát hiện {len(rings)} vòng/motif nghi vấn và {len(top_edges)} cạnh hóa đơn có mức rủi ro cao."
        )
        key_nodes = list((graph.get("top_nodes") or graph.get("key_nodes") or [])[:5])
        edge_lines = []
        for idx, edge in enumerate(top_edges[:10], 1):
            edge_lines.append(
                f"| {idx} | {edge.get('seller_tax_code') or edge.get('source') or ''} | "
                f"{edge.get('buyer_tax_code') or edge.get('target') or ''} | "
                f"{edge.get('amount') or edge.get('value') or ''} | "
                f"{edge.get('edge_risk_score') or edge.get('risk_score') or ''} |"
            )
        edge_table = "\n".join(edge_lines) if edge_lines else "| - | - | - | - | - |"
        detailed = (
            "### Kết luận mạng VAT\n"
            f"{summary}\n\n"
            "### Tín hiệu trọng yếu\n"
            f"- Tổng giá trị nghi vấn: {suspect_value}\n"
            f"- Số vòng/motif nghi vấn: {len(rings)}\n"
            f"- Số cạnh rủi ro cao: {len(top_edges)}\n"
            f"- Số cảnh báo dữ liệu: {len(vat.get('warnings') or [])}\n\n"
            "### Pháp nhân/nút trọng yếu\n"
            f"- Số nút tham gia phân tích: {len(graph.get('nodes') or [])}\n"
            f"- Số cạnh giao dịch: {len(graph.get('edges') or [])}\n"
            f"- Top nút cần theo dõi: {', '.join(str(n.get('tax_code') or n.get('id') or '') for n in key_nodes) if key_nodes else 'Chưa đủ dữ liệu xếp hạng nút'}\n\n"
            "### Hóa đơn/cạnh rủi ro cao\n"
            "| STT | Bên bán | Bên mua | Giá trị | Điểm cạnh |\n"
            "|---:|---|---|---:|---:|\n"
            f"{edge_table}\n\n"
            "### Chuỗi bằng chứng và giả thuyết điều tra\n"
            "Multi-agent hợp nhất tín hiệu từ VAT graph, scoring cạnh hóa đơn và dữ liệu chứng từ để tạo thứ tự điều tra. "
            "Ưu tiên xác minh theo chuỗi: tồn tại giao dịch thực, dòng tiền tương ứng, quan hệ sở hữu/liên kết, tính hợp lệ chứng từ và đối chiếu khai thuế.\n\n"
            "### Bước tiếp theo đề xuất\n"
            "Tập trung trước vào các cụm có vòng giao dịch ngắn, cạnh giá trị lớn và nút trung gian lặp lại nhiều kỳ. "
            "Khi có dữ liệu OCR hoặc thông tin xuyên biên giới, cần ghép vào đường dẫn bằng chứng để nâng độ chắc chắn trước khi kiến nghị thanh tra."
        )
        return SynthesisResult(
            summary=summary,
            detailed_analysis=detailed,
            evidence=[],
            recommendations=[
                "Mở rộng điều tra các nút nằm trong vòng/motif và các nút trung gian lặp lại nhiều kỳ.",
                "Đối chiếu hóa đơn giá trị lớn với dòng tiền, chứng từ vận chuyển và năng lực giao nhận thực tế.",
                "Chạy OCR/invoice risk cho các chứng từ thuộc cạnh rủi ro cao và hợp nhất vào evidence path.",
                "Lập danh sách kiểm tra theo mức ưu tiên để chuyển đội thanh tra xử lý theo từng đợt.",
            ],
            confidence=0.86 if processed else 0.45,
            limitations="Kết quả phụ thuộc độ đầy đủ của CSV hóa đơn và dữ liệu liên kết sở hữu.",
            escalation_needed=bool(rings or top_edges),
            intent="vat_network_analysis",
            tools_used=["_vat_graph_batch_results"],
            reasoning_trace=reasoning_trace,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            synthesis_tier="vat_graph_v2",
            verification={"status": "not_required", "reason": "model_output"},
            citation_map={},
        )

    def _build_batch_analysis_response(
        self,
        *,
        tool_results: dict[str, dict[str, Any]],
        reasoning_trace: str,
        t0: float,
    ) -> SynthesisResult:
        """Build a rich summary response for batch file analysis."""
        batch = tool_results.get("_batch_results", {}) or {}
        attachment = tool_results.get("_attachment_analysis", {}) or {}

        total = int(batch.get("total", 0) or attachment.get("total", 0) or 0)
        companies = list(
            batch.get("assessments", [])
            or batch.get("companies", [])
            or attachment.get("assessments", [])
            or attachment.get("companies", [])
            or []
        )
        by_level = batch.get("by_level", {}) or attachment.get("by_level", {}) or {}
        top_risky = (
            batch.get("top_5", [])
            or batch.get("top_risky", [])
            or attachment.get("top_5", [])
            or attachment.get("top_risky", [])
            or []
        )
        filename = batch.get("filename", "") or attachment.get("filename", "CSV")

        if not total and not companies:
            return SynthesisResult(
                summary=f"Đã nhận file {filename} nhưng chưa thể phân tích batch.",
                detailed_analysis="File có thể không đúng định dạng hoặc không chứa cột tax_code/MST. "
                                  "Vui lòng kiểm tra lại file và thử upload lại.",
                evidence=[], recommendations=["Kiểm tra lại format file CSV với cột tax_code"],
                confidence=0.3, limitations="Không đọc được dữ liệu từ file",
                escalation_needed=False, intent="batch_analysis",
                tools_used=["_batch_results"], reasoning_trace=reasoning_trace,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                synthesis_tier="batch_analysis", verification={"status": "not_required"},
                citation_map={},
            )

        # Build risk distribution summary
        very_high = int(by_level.get("very_high", 0) or by_level.get("Rất cao", 0) or 0)
        high = int(by_level.get("high", 0) or by_level.get("Cao", 0) or 0)
        medium = int(by_level.get("medium", 0) or by_level.get("Trung bình", 0) or 0)
        low = int(by_level.get("low", 0) or by_level.get("An toàn", 0) or by_level.get("Thấp", 0) or 0)

        summary = (
            f"**Kết quả phân tích lô: {filename}** — "
            f"{total} doanh nghiệp đã được chấm điểm rủi ro."
        )

        risk_lines = []
        if very_high:
            risk_lines.append(f"🔴 **Rất cao**: {very_high} DN")
        if high:
            risk_lines.append(f"🟠 **Cao**: {high} DN")
        if medium:
            risk_lines.append(f"🟡 **Trung bình**: {medium} DN")
        if low:
            risk_lines.append(f"🟢 **An toàn**: {low} DN")

        risk_summary = "\n".join(risk_lines) if risk_lines else f"Tổng cộng {total} DN đã phân tích."

        # Top risky companies detail
        top_detail = ""
        display_top = top_risky[:5] if top_risky else companies[:5]
        if display_top:
            top_lines = []
            for i, c in enumerate(display_top, 1):
                tc = c.get("tax_code", c.get("mst", ""))
                name = c.get("company_name", c.get("ten_dn", "N/A"))
                score = c.get("risk_score", c.get("diem_rui_ro", 0))
                level = c.get("risk_level", c.get("muc_do", ""))
                top_lines.append(f"{i}. **MST {tc}** — {name} — Điểm: **{score}** ({level})")
            top_detail = "\n".join(top_lines)

        detailed = (
            f"### Phân bố rủi ro\n{risk_summary}\n\n"
            f"### Top doanh nghiệp rủi ro cao nhất\n{top_detail}\n\n"
            f"*(Bảng chi tiết đã hiển thị bên dưới. Bạn có thể nhấn vào từng dòng để xem "
            f"phân tích chuyên sâu cho doanh nghiệp đó.)*"
        )

        recommendations = [
            "Tập trung kiểm tra các DN có điểm rủi ro trên 80.",
            "Đối chiếu thêm số liệu nghiệp vụ trước khi ra quyết định.",
        ]
        if very_high > 0:
            recommendations.insert(0, f"⚠️ Có {very_high} DN ở mức RỦI RO RẤT CAO — cần ưu tiên xử lý ngay.")

        confidence = 0.85 if total > 0 else 0.3

        return SynthesisResult(
            summary=summary,
            detailed_analysis=detailed,
            evidence=[],
            recommendations=recommendations,
            confidence=confidence,
            limitations="",
            escalation_needed=(very_high > 5),
            intent="batch_analysis",
            tools_used=[k for k in tool_results if k.startswith("_")],
            reasoning_trace=reasoning_trace,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            synthesis_tier="batch_analysis",
            verification={"status": "not_required", "reason": "direct_batch_data"},
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
        sur = tool_results.get("_session_upload_row", {})
        row = sur.get("row") if isinstance(sur, dict) else None
        if isinstance(row, dict) and sur.get("status") == "matched":
            src = sur.get("source_filename") or "file đã upload"
            tc = row.get("tax_code") or sur.get("tax_code")
            evidence.append(Evidence(
                source_tool="_session_upload_row",
                source_type="analytics",
                content=(
                    f"(Theo snapshot phiên làm việc từ `{src}`, không phải CSDL cố định) MST {tc}: "
                    f"{row.get('company_name') or ''} — Điểm rủi ro: {row.get('risk_score')} "
                    f"({row.get('risk_level', '')}); F1–F4: "
                    f"{row.get('f1_divergence')}/{row.get('f2_ratio_limit')}/"
                    f"{row.get('f3_vat_structure')}/{row.get('f4_peer_comparison')}"
                ),
                title="Chi tiết từ file vừa phân tích",
                score=min(1.0, float(row.get("risk_score") or 0.0) / 100.0),
            ))

        if cr.get("status") == "found":
            if isinstance(row, dict) and sur.get("status") == "matched":
                evidence.append(Evidence(
                    source_tool="company_risk_lookup",
                    source_type="analytics",
                    content=(
                        f"(Tham chiếu CSDL) MST {cr.get('tax_code', '')}: điểm {cr.get('risk_score', 0)}. "
                        f"Kết luận theo file vừa upload ưu tiên dòng snapshot phiên làm việc phía trên."
                    ),
                    title="Đối chiếu CSDL",
                    score=float(cr.get("risk_score", 0)) / 100.0,
                ))
            else:
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

        vf = tool_results.get("_vat_session_focus", {})
        if isinstance(vf, dict) and vf.get("status") == "matched" and vf.get("invoices"):
            evidence.append(Evidence(
                source_tool="_vat_session_focus",
                source_type="investigation",
                content=(
                    f"(Theo file VAT `{vf.get('source_filename') or 'upload'}`, batch {vf.get('batch_id')}) "
                    f"MST {vf.get('tax_code')} — {len(vf.get('invoices', []))} dòng giao dịch/hóa đơn liên quan "
                    "trong snapshot phiên làm việc."
                ),
                title="VAT snapshot theo MST",
                score=0.75,
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
        legal_count = sum(1 for e in evidence if e.source_type == "legal")
        analytics_count = sum(1 for e in evidence if e.source_type == "analytics")

        summary_parts = []
        if analytics_count > 0:
            analytics_scores = [e.score for e in evidence if e.source_type == "analytics"]
            avg_risk = sum(analytics_scores) / max(len(analytics_scores), 1)
            
            if avg_risk > 0.7:
                risk_label = "RỦI RO CAO"
            elif avg_risk > 0.4:
                risk_label = "RỦI RO TRUNG BÌNH"
            else:
                risk_label = "RỦI RO THẤP"
            summary_parts.append(f"{prefix}{entity_str}: {risk_label}.")
            summary_parts.append(f"Đã kiểm tra {analytics_count} chỉ số phân tích.")
        else:
            summary_parts.append(f"{prefix}{entity_str} hoàn tất.")

        if legal_count > 0:
            summary_parts.append(f"Có {legal_count} căn cứ pháp lý liên quan.")

        summary = " ".join(summary_parts)
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
        if answer_contract in {"data_table", "smalltalk", "file_analysis", "vat_graph", "fraud_analysis", "mode_mismatch"}:
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
        """Use OpenRouter or Gemini API if available, else fallback to template."""
        enable = os.getenv("TAX_AGENT_ENABLE_LLM", "").strip().lower() in {"1", "true", "yes"}
        or_api_key = os.getenv("OPENROUTER_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        api_model = os.getenv("TAX_AGENT_API_MODEL", "gemini-1.5-flash")
        
        legal_evidence = [ev for ev in evidence if ev.source_type == "legal"]
        if not legal_evidence:
            return None, "template"

        if enable and (or_api_key or gemini_api_key):
            context_parts = [
                f"{ev.citation_key} {ev.title}: {ev.content[:700]}"
                for ev in legal_evidence[:5]
            ]
            try:
                import requests
                prompt = (
                    "Bạn là một chuyên gia tư vấn thuế (TaxInspector AI) dạn dày kinh nghiệm của Cục Thuế. "
                    "Hãy trả lời câu hỏi sau của người nộp thuế một cách DÀI, CHI TIẾT và THÔNG MINH dựa trên CÁC CĂN CỨ PHÁP LÝ được cung cấp.\n"
                    "YÊU CẦU QUAN TRỌNG: \n"
                    "- Mở đầu bằng lời khẳng định đi thẳng vào vấn đề (Ví dụ: Bạn hoàn toàn có thể..., Trong trường hợp này bạn sẽ bị phạt..., v.v.).\n"
                    "- Giải thích cặn kẽ tại sao lại như vậy dựa trên các trích dẫn luật [1], [2].\n"
                    "- Viết bằng văn phong lịch sự, thấu cảm, dễ hiểu nhưng vẫn giữ tính pháp lý.\n"
                    "- Trình bày Markdown rõ ràng (Dùng in đậm, gạch đầu dòng).\n\n"
                    f"CÂU HỎI CỦA NGƯỜI NỘP THUẾ: {query}\n\n"
                    f"CĂN CỨ PHÁP LÝ:\n{chr(10).join(context_parts)}"
                )
                
                # Dùng Gemini API trực tiếp nếu model chứa chữ gemini
                if "gemini" in api_model.lower() and gemini_api_key:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={gemini_api_key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}]
                    }
                    response = requests.post(url, json=payload, timeout=20)
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["candidates"][0]["content"]["parts"][0]["text"]
                        return answer.strip(), "llm_api"
                    else:
                        logger.debug(f"[Synthesis] Gemini API returned status {response.status_code}: {response.text}")
                
                # Fallback sang OpenRouter
                elif or_api_key:
                    or_model = api_model if "openrouter" in api_model.lower() else "openrouter/free"
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {or_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": or_model,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=20,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["choices"][0]["message"]["content"]
                        return answer.strip(), "llm_api"
                    else:
                        logger.debug(f"[Synthesis] OpenRouter API returned status {response.status_code}: {response.text}")
            except Exception as exc:
                logger.debug("[Synthesis] API call failed, fallback to template: %s", exc)

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
        
        parts = []
        parts.append("Dựa trên các quy định pháp luật hiện hành và thông tin bạn cung cấp, hệ thống xin đưa ra tư vấn chi tiết như sau:\n")
        
        if legal_evidence:
            strongest = max(legal_evidence, key=lambda ev: ev.score)
            parts.append("### 1. Trả lời trọng tâm")
            parts.append(
                f"**Về cơ bản:** Trường hợp của bạn hoàn toàn có thể được giải quyết dựa trên quy định tại **{strongest.title}** {strongest.citation_key}. "
                "Tuy nhiên, kết quả chính xác sẽ phụ thuộc vào việc đối chiếu hồ sơ và số liệu thực tế của bạn với cơ quan thuế."
            )
            
            parts.append("\n### 2. Phân tích chi tiết và Căn cứ pháp lý")
            parts.append("Dưới đây là các cơ sở pháp lý trực tiếp điều chỉnh trường hợp của bạn:")
            
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
                    scope_note = " *(Lưu ý: Đây là công văn hướng dẫn nghiệp vụ, chỉ có giá trị tham khảo cho các trường hợp tương tự)*"
                
                parts.append(
                    f"\n**{ev.title}** {ev.citation_key} (Tình trạng: {state_vi}){scope_note}"
                )
                parts.append(f"> {ev.content[:400]}...")

            parts.append("\n### 3. Chuỗi quan hệ pháp lý liên quan")
            graph_context = tool_results.get("knowledge_search", {}).get("graph_context") or {}
            authority_path = graph_context.get("authority_path") or []
            if authority_path:
                parts.append("Hệ thống GraphRAG đã tự động đối chiếu các văn bản gốc để đảm bảo tính chính xác:")
                for item in authority_path[:5]:
                    entity_type_vi = {
                        "law": "Luật", "decree": "Nghị định", "circular": "Thông tư",
                        "decision": "Quyết định", "official_letter": "Công văn",
                        "article": "Điều", "clause": "Khoản",
                    }.get(item.get("entity_type", ""), item.get("entity_type", ""))
                    parts.append(f"- Tham chiếu từ: {item.get('display_name')} ({entity_type_vi})")
            else:
                for ev in legal_evidence[:3]:
                    parts.append(f"- Tham chiếu từ: {ev.title} {ev.citation_key}")

            parts.append("\n### 4. Rủi ro pháp lý và Khuyến nghị")
            official_rollup = graph_context.get("official_letter_scope") or {}
            if official_rollup.get("warnings"):
                for warning in official_rollup["warnings"][:2]:
                    if "Official letters are guidance" in warning:
                        parts.append("- ⚠️ **Lưu ý pháp lý:** Công văn chỉ mang tính chất hướng dẫn hành chính, không phải là văn bản quy phạm pháp luật cao nhất.")
                    else:
                        parts.append(f"- ⚠️ {warning}")
            
            effective_rollup = graph_context.get("effective_status") or {}
            if effective_rollup.get("has_non_usable"):
                parts.append("- ⚠️ **Rủi ro hiệu lực:** Có văn bản hết hiệu lực trong chuỗi tra cứu, hệ thống đã ưu tiên trích xuất văn bản mới nhất.")
        else:
            parts.append("Hiện tại hệ thống chưa tìm thấy trích dẫn pháp lý đủ mạnh để đưa ra kết luận chắc chắn. Vui lòng cung cấp thêm chi tiết cụ thể hơn.")

        parts.append("\n### 5. Hướng dẫn các bước tiếp theo")
        parts.append("- Bạn cần tập hợp đầy đủ chứng từ gốc, hóa đơn, và các hợp đồng liên quan.")
        parts.append("- Xác định chính xác kỳ thuế và nộp hồ sơ kê khai/hoàn thuế lên cơ quan thuế quản lý trực tiếp.")
        parts.append("- *Lưu ý: Phản hồi này được tạo tự động bởi AI dựa trên cơ sở dữ liệu pháp luật hiện hành và chỉ mang tính chất tham khảo.*")
        
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
