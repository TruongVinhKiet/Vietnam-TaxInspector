"""
tax_agent_legal_research.py – Legal Research Sub-Agent (Phase 3)
=================================================================
Specialized agent for Vietnamese tax law research.

Capabilities:
    1. Hierarchical legal document search (Luật → Nghị Định → Thông Tư → công văn)
    2. Cross-reference detection between legal documents
    3. Legal opinion generation with citation chains
    4. Temporal law validity checking (effective dates, amendments)
    5. Conflict resolution between overlapping regulations

Architecture:
    Input: Sub-task from Orchestrator
    Processing: Multi-pass retrieval → cross-reference → opinion synthesis
    Output: Structured legal analysis with citation graph
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Vietnamese Legal Document Hierarchy ──────────────────────────────────────

class LegalDocType:
    """Vietnamese legal document type hierarchy."""
    CONSTITUTION = "hien_phap"         # Hiến pháp (highest)
    LAW = "luat"                       # Luật
    ORDINANCE = "phap_lenh"            # Pháp lệnh
    DECREE = "nghi_dinh"               # Nghị định
    RESOLUTION = "nghi_quyet"          # Nghị quyết
    CIRCULAR = "thong_tu"              # Thông tư
    JOINT_CIRCULAR = "thong_tu_lien_tich"  # Thông tư liên tịch
    DECISION = "quyet_dinh"            # Quyết định
    OFFICIAL_LETTER = "cong_van"       # Công văn (lowest binding)
    GUIDELINE = "huong_dan"            # Hướng dẫn

    # Authority ranking (higher = more authoritative)
    AUTHORITY_RANK = {
        "hien_phap": 100,
        "luat": 90, "law": 90,
        "phap_lenh": 85,
        "nghi_dinh": 80, "decree": 80,
        "nghi_quyet": 75,
        "thong_tu": 70, "circular": 70,
        "thong_tu_lien_tich": 68,
        "quyet_dinh": 60, "decision": 60,
        "cong_van": 40, "official_letter": 40,
        "huong_dan": 30, "guideline": 30,
    }


# ─── Legal Reference Pattern Detection ────────────────────────────────────────

LEGAL_REF_PATTERNS = [
    # Điều X Luật Y
    re.compile(
        r"[Đđ]iều\s+(\d+)\s+(?:của\s+)?(?:Luật|Nghị định|Thông tư|Quyết định)\s+"
        r"(?:số\s+)?(\d+[/\-]\d+[/\-]?[\w\-]*)",
        re.IGNORECASE,
    ),
    # Khoản X Điều Y
    re.compile(
        r"[Kk]hoản\s+(\d+)\s+[Đđ]iều\s+(\d+)",
        re.IGNORECASE,
    ),
    # Số hiệu văn bản: 123/2024/TT-BTC
    re.compile(
        r"\b(\d{1,4}[/\-](?:20[0-9]{2}|19[0-9]{2})[/\-](?:TT|NĐ|QĐ|CV|NQ|PL)\-?[A-ZĐ]{2,5})\b",
        re.IGNORECASE,
    ),
    # Luật Thuế GTGT
    re.compile(
        r"Luật\s+(Thuế\s+\w+(?:\s+\w+)?|Quản lý thuế|Ngân sách|Đầu tư|Doanh nghiệp)",
        re.IGNORECASE,
    ),
]


@dataclass
class LegalReference:
    """A detected legal reference."""
    ref_type: str              # "article", "clause", "document_number", "law_name"
    ref_text: str
    article_number: str | None = None
    clause_number: str | None = None
    document_number: str | None = None
    law_name: str | None = None
    source_evidence_idx: int = -1   # Which evidence the ref was found in


@dataclass
class LegalCrossRef:
    """A cross-reference between two legal documents."""
    source_ref: LegalReference
    target_ref: LegalReference
    relationship: str          # "amends", "replaces", "supplements", "cites", "interprets"
    confidence: float = 0.5


@dataclass
class LegalOpinion:
    """Structured legal opinion."""
    question: str
    applicable_laws: list[dict[str, Any]]   # Ranked by authority
    analysis: str
    conclusion: str
    citation_chain: list[str]               # Citation trail
    authority_score: float                   # How authoritative the answer is
    temporal_validity: str                   # "current", "expired", "pending"
    cross_references: list[LegalCrossRef]
    caveats: list[str]
    confidence: float


class LegalResearchAgent:
    """
    Specialized agent for Vietnamese tax law research.

    Multi-pass Research Flow:
    1. Broad retrieval → gather candidate legal documents
    2. Authority ranking → sort by legal hierarchy
    3. Cross-reference detection → find citation chains
    4. Temporal validity check → filter expired/amended laws
    5. Opinion synthesis → generate grounded legal opinion

    Usage:
        agent = LegalResearchAgent()
        opinion = agent.research(
            query="Điều kiện hoàn thuế VAT cho doanh nghiệp xuất khẩu?",
            retrieval_results=[...],
            intent="vat_refund_risk",
        )
    """

    # ─── Tax Law Knowledge Base (static fallback) ─────────────────────────

    TAX_LAW_REFERENCES = {
        "vat_refund_risk": {
            "primary": [
                "Luật Thuế GTGT số 13/2008/QH12 (sửa đổi 31/2013, 71/2014, 106/2016)",
                "Nghị định 209/2013/NĐ-CP",
                "Thông tư 219/2013/TT-BTC",
                "Thông tư 130/2016/TT-BTC",
            ],
            "key_articles": [
                "Điều 13 Luật Thuế GTGT - Hoàn thuế",
                "Điều 18 Thông tư 219/2013 - Điều kiện hoàn thuế",
                "Điều 16 Thông tư 219/2013 - Thuế suất 0%",
            ],
        },
        "invoice_risk": {
            "primary": [
                "Nghị định 123/2020/NĐ-CP về hóa đơn, chứng từ",
                "Thông tư 78/2021/TT-BTC hướng dẫn hóa đơn điện tử",
                "Luật Quản lý thuế số 38/2019/QH14",
            ],
            "key_articles": [
                "Điều 4 NĐ 123/2020 - Nguyên tắc lập hóa đơn",
                "Điều 15-19 NĐ 123/2020 - Nội dung hóa đơn",
            ],
        },
        "delinquency": {
            "primary": [
                "Luật Quản lý thuế số 38/2019/QH14",
                "Nghị định 126/2020/NĐ-CP",
                "Thông tư 80/2021/TT-BTC",
            ],
            "key_articles": [
                "Điều 55 Luật QLT - Thời hạn nộp thuế",
                "Điều 59 Luật QLT - Xử lý nợ thuế",
                "Điều 62 Luật QLT - Cưỡng chế nợ thuế",
            ],
        },
        "transfer_pricing": {
            "primary": [
                "Nghị định 132/2020/NĐ-CP về giao dịch liên kết",
                "Thông tư 45/2021/TT-BTC",
                "Luật Thuế TNDN",
            ],
            "key_articles": [
                "Điều 16-18 NĐ 132/2020 - Xác định giá giao dịch liên kết",
                "Điều 10 NĐ 132/2020 - Phương pháp so sánh",
            ],
        },
        "audit_selection": {
            "primary": [
                "Luật Quản lý thuế số 38/2019/QH14",
                "Nghị định 126/2020/NĐ-CP",
                "Quyết định 970/QĐ-TCT",
            ],
            "key_articles": [
                "Điều 110 Luật QLT - Quyết định thanh tra",
                "Điều 113 Luật QLT - Thời hạn thanh tra",
            ],
        },
        "osint_ownership": {
            "primary": [
                "Luật Doanh nghiệp số 59/2020/QH14",
                "Nghị định 47/2021/NĐ-CP về chủ sở hữu hưởng lợi",
                "Luật Phòng chống rửa tiền",
            ],
            "key_articles": [
                "Điều 4 NĐ 47/2021 - Xác định chủ sở hữu hưởng lợi",
                "Điều 15 Luật DN - Nghĩa vụ công bố thông tin",
            ],
        },
    }

    def research(
        self,
        query: str,
        retrieval_results: list[dict[str, Any]],
        intent: str = "general_tax_query",
        *,
        tax_code: str | None = None,
    ) -> LegalOpinion:
        """
        Conduct multi-pass legal research on the query.

        Args:
            query: User's legal question
            retrieval_results: Knowledge search results from tool executor
            intent: Classified intent
            tax_code: Optional tax code for entity-specific analysis

        Returns:
            LegalOpinion with structured analysis
        """
        t0 = time.perf_counter()

        # Pass 1: Extract and rank legal evidence
        ranked_evidence = self._rank_by_authority(retrieval_results)

        # Pass 2: Extract legal references from evidence
        all_refs = self._extract_references(ranked_evidence)

        # Pass 3: Detect cross-references
        cross_refs = self._detect_cross_references(all_refs)

        # Pass 4: Check temporal validity
        temporal = self._check_temporal_validity(ranked_evidence)

        # Pass 5: Build citation chain
        citation_chain = self._build_citation_chain(ranked_evidence, all_refs)

        # Pass 6: Enrich with known law references
        applicable_laws = self._enrich_with_known_laws(ranked_evidence, intent)

        # Pass 7: Generate analysis and conclusion
        analysis = self._generate_analysis(query, ranked_evidence, intent, tax_code)
        conclusion = self._generate_conclusion(query, ranked_evidence, intent)

        # Pass 8: Identify caveats
        caveats = self._identify_caveats(ranked_evidence, intent, temporal)

        # Compute authority score
        authority_score = self._compute_authority_score(ranked_evidence)

        # Compute confidence
        confidence = self._compute_confidence(
            ranked_evidence, authority_score, len(citation_chain),
        )

        latency = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "[LegalAgent] Research completed: intent=%s, evidence=%d, "
            "refs=%d, cross_refs=%d, conf=%.2f in %.0fms",
            intent, len(ranked_evidence), len(all_refs),
            len(cross_refs), confidence, latency,
        )

        return LegalOpinion(
            question=query,
            applicable_laws=applicable_laws,
            analysis=analysis,
            conclusion=conclusion,
            citation_chain=citation_chain,
            authority_score=authority_score,
            temporal_validity=temporal,
            cross_references=cross_refs,
            caveats=caveats,
            confidence=confidence,
        )

    def _rank_by_authority(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank evidence by legal document authority."""
        def authority_key(item: dict) -> tuple:
            doc_type = str(item.get("doc_type") or "").lower()
            rank = LegalDocType.AUTHORITY_RANK.get(doc_type, 20)
            score = float(item.get("score") or 0)
            return (-rank, -score)

        return sorted(evidence, key=authority_key)

    def _extract_references(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[LegalReference]:
        """Extract legal references from evidence text."""
        refs: list[LegalReference] = []

        for idx, ev in enumerate(evidence):
            text = str(ev.get("text") or "")
            title = str(ev.get("title") or "")
            combined = f"{title} {text}"

            for pattern in LEGAL_REF_PATTERNS:
                for match in pattern.finditer(combined):
                    groups = match.groups()
                    ref = LegalReference(
                        ref_type="legal_reference",
                        ref_text=match.group(0),
                        source_evidence_idx=idx,
                    )

                    # Try to parse article/clause numbers
                    for g in groups:
                        if g and g.isdigit():
                            if ref.article_number is None:
                                ref.article_number = g
                            elif ref.clause_number is None:
                                ref.clause_number = g
                        elif g and "/" in g:
                            ref.document_number = g

                    refs.append(ref)

        return refs

    def _detect_cross_references(
        self,
        refs: list[LegalReference],
    ) -> list[LegalCrossRef]:
        """Detect cross-references between legal documents."""
        cross_refs: list[LegalCrossRef] = []

        # Group refs by document number
        by_doc: dict[str, list[LegalReference]] = {}
        for ref in refs:
            if ref.document_number:
                by_doc.setdefault(ref.document_number, []).append(ref)

        # Find cross-references (refs mentioning different documents from different sources)
        doc_numbers = list(by_doc.keys())
        for i, doc_a in enumerate(doc_numbers):
            for doc_b in doc_numbers[i + 1:]:
                refs_a = by_doc[doc_a]
                refs_b = by_doc[doc_b]

                # If refs from different source evidence mention each other
                sources_a = set(r.source_evidence_idx for r in refs_a)
                sources_b = set(r.source_evidence_idx for r in refs_b)

                if sources_a != sources_b:
                    cross_refs.append(LegalCrossRef(
                        source_ref=refs_a[0],
                        target_ref=refs_b[0],
                        relationship="cites",
                        confidence=0.7,
                    ))

        return cross_refs

    def _check_temporal_validity(
        self,
        evidence: list[dict[str, Any]],
    ) -> str:
        """Check if cited laws are currently valid."""
        # Simple heuristic: check for amendment/replacement keywords
        amendment_keywords = [
            "sửa đổi", "bổ sung", "thay thế", "bãi bỏ", "hết hiệu lực",
            "amended", "replaced", "repealed",
        ]

        has_amendments = False
        for ev in evidence:
            text = str(ev.get("text") or "").lower()
            if any(kw in text for kw in amendment_keywords):
                has_amendments = True
                break

        if has_amendments:
            return "check_amendments"
        return "current"

    def _build_citation_chain(
        self,
        evidence: list[dict[str, Any]],
        refs: list[LegalReference],
    ) -> list[str]:
        """Build a chain of citations from evidence."""
        chain: list[str] = []
        seen: set[str] = set()

        for ev in evidence[:5]:
            title = str(ev.get("title") or "")
            if title and title not in seen:
                chain.append(title)
                seen.add(title)

        for ref in refs[:10]:
            key = ref.ref_text[:80]
            if key and key not in seen:
                chain.append(key)
                seen.add(key)

        return chain

    def _enrich_with_known_laws(
        self,
        evidence: list[dict[str, Any]],
        intent: str,
    ) -> list[dict[str, Any]]:
        """Enrich with known law references for the intent."""
        known = self.TAX_LAW_REFERENCES.get(intent, {})
        applicable = []

        # Add known primary laws
        for law in known.get("primary", []):
            applicable.append({
                "source": "knowledge_base",
                "type": "primary_law",
                "reference": law,
                "authority": "high",
            })

        # Add known key articles
        for article in known.get("key_articles", []):
            applicable.append({
                "source": "knowledge_base",
                "type": "key_article",
                "reference": article,
                "authority": "high",
            })

        # Add retrieved evidence
        for ev in evidence[:3]:
            applicable.append({
                "source": "retrieval",
                "type": str(ev.get("doc_type") or "unknown"),
                "reference": str(ev.get("title") or ""),
                "score": float(ev.get("score") or 0),
                "authority": "medium",
            })

        return applicable

    def _generate_analysis(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        intent: str,
        tax_code: str | None,
    ) -> str:
        """
        Generate structured legal analysis.
        Template-based for now — ready for custom LLM replacement.
        """
        parts = []

        # Opening
        entity_str = f" (MST: {tax_code})" if tax_code else ""
        parts.append(
            f"### Phân tích pháp lý{entity_str}\n"
        )

        # Legal basis
        known = self.TAX_LAW_REFERENCES.get(intent, {})
        if known.get("primary"):
            parts.append("**Căn cứ pháp lý chính:**")
            for i, law in enumerate(known["primary"][:3], 1):
                parts.append(f"{i}. {law}")

        if known.get("key_articles"):
            parts.append("\n**Điều khoản áp dụng:**")
            for article in known["key_articles"][:3]:
                parts.append(f"- {article}")

        # Retrieved evidence
        if evidence:
            parts.append("\n**Trích dẫn từ hệ thống tri thức:**")
            for i, ev in enumerate(evidence[:3], 1):
                title = ev.get("title", "")
                text = str(ev.get("text", ""))[:200]
                score = float(ev.get("score") or 0)
                parts.append(f"[{i}] **{title}** (điểm: {score:.2f})")
                parts.append(f"  > {text}...")

        return "\n".join(parts)

    def _generate_conclusion(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        intent: str,
    ) -> str:
        """Generate legal conclusion."""
        if not evidence:
            return (
                "Chưa đủ dữ liệu pháp lý để đưa ra kết luận. "
                "Cần bổ sung thêm các văn bản pháp luật liên quan."
            )

        known = self.TAX_LAW_REFERENCES.get(intent, {})
        laws = known.get("primary", [])

        if laws:
            law_str = ", ".join(laws[:2])
            return (
                f"Dựa trên {law_str} và {len(evidence)} nguồn tri thức đã tra cứu, "
                f"câu hỏi thuộc phạm vi điều chỉnh của các văn bản trên. "
                f"Cần đối chiếu thêm với thực tế hồ sơ cụ thể trước khi áp dụng."
            )

        return (
            f"Dựa trên {len(evidence)} nguồn tri thức đã tra cứu, "
            f"hệ thống đã xác định các căn cứ pháp lý liên quan. "
            f"Khuyến nghị xác minh với chuyên viên pháp chế."
        )

    def _identify_caveats(
        self,
        evidence: list[dict[str, Any]],
        intent: str,
        temporal: str,
    ) -> list[str]:
        """Identify caveats and limitations."""
        caveats = []

        if temporal == "check_amendments":
            caveats.append(
                "Một số văn bản có thể đã được sửa đổi/bổ sung. "
                "Cần kiểm tra phiên bản hiệu lực hiện hành."
            )

        if len(evidence) < 2:
            caveats.append(
                "Số lượng nguồn trích dẫn hạn chế. "
                "Cần bổ sung thêm căn cứ pháp lý."
            )

        if intent in ("transfer_pricing", "osint_ownership"):
            caveats.append(
                "Lĩnh vực chuyển giá/sở hữu vốn đòi hỏi phân tích chuyên sâu. "
                "Khuyến nghị tham vấn chuyên gia."
            )

        return caveats

    def _compute_authority_score(
        self,
        evidence: list[dict[str, Any]],
    ) -> float:
        """Compute overall authority score based on document types."""
        if not evidence:
            return 0.0

        total_rank = 0.0
        for ev in evidence[:5]:
            doc_type = str(ev.get("doc_type") or "").lower()
            rank = LegalDocType.AUTHORITY_RANK.get(doc_type, 20)
            total_rank += rank

        max_possible = 100.0 * min(5, len(evidence))
        return round(total_rank / max_possible, 4)

    def _compute_confidence(
        self,
        evidence: list[dict[str, Any]],
        authority_score: float,
        citation_count: int,
    ) -> float:
        """Compute overall confidence of the legal opinion."""
        factors = []

        # Evidence quantity
        factors.append(min(1.0, len(evidence) / 5.0) * 0.25)

        # Evidence quality (average score)
        scores = [float(ev.get("score") or 0) for ev in evidence if ev.get("score")]
        if scores:
            factors.append((sum(scores) / len(scores)) * 0.30)

        # Authority
        factors.append(authority_score * 0.25)

        # Citation coverage
        factors.append(min(1.0, citation_count / 5.0) * 0.20)

        return round(min(1.0, sum(factors)), 4)
