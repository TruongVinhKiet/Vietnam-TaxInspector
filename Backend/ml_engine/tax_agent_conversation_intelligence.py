"""
tax_agent_conversation_intelligence.py – Advanced Conversation Management
==========================================================================
Provides high-level conversation intelligence capabilities that sit on top
of the basic ConversationMemory module.

Capabilities:
    1. Coreference Resolution: "công ty đó" → resolve to last MST
    2. Ellipsis Expansion: "VAT thì sao?" → "phân tích rủi ro VAT cho [MST]"
    3. Ambiguity Detection: detect when a query is too vague to process
    4. Comparison Handling: "so sánh A với B" → parallel analysis
    5. Follow-up Detection: identify follow-up questions vs new topics

Design Decisions:
    - Rule-based + regex: No LLM dependency for deterministic behavior
    - Vietnamese-first: All patterns optimized for Vietnamese tax domain
    - Graceful fallback: If no resolution found, returns original query unchanged
    - Non-destructive: Original message is preserved alongside resolved version
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class ConversationIntelligenceResult:
    """Result of conversation intelligence processing."""
    original_message: str
    resolved_message: str              # After coreference + ellipsis resolution
    is_followup: bool                  # True if this is a follow-up to previous turn
    is_comparison: bool                # True if user wants to compare entities
    is_ambiguous: bool                 # True if query is too vague
    ambiguity_reason: str | None = None   # Why it's ambiguous
    clarification_prompt: str | None = None  # Suggested clarification question
    resolved_entities: dict[str, str] = field(default_factory=dict)  # What was resolved
    comparison_targets: list[str] = field(default_factory=list)  # For comparison queries

    def to_dict(self) -> dict:
        return {
            "original": self.original_message,
            "resolved": self.resolved_message,
            "is_followup": self.is_followup,
            "is_comparison": self.is_comparison,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "clarification_prompt": self.clarification_prompt,
            "resolved_entities": self.resolved_entities,
            "comparison_targets": self.comparison_targets,
        }


# ════════════════════════════════════════════════════════════════
#  Patterns (Vietnamese tax domain)
# ════════════════════════════════════════════════════════════════

# Coreference patterns: pronouns/references that need resolution
COREFERENCE_PATTERNS = [
    # "công ty đó", "doanh nghiệp đó", "DN đó"
    re.compile(r"\b(công ty|doanh nghiệp|DN|cty)\s+(đó|này|trên|kia|ấy)\b", re.IGNORECASE),
    # "nó", "họ" (referring to a company)
    re.compile(r"\b(của\s+)?(nó|họ|đơn vị đó)\b", re.IGNORECASE),
    # "mã số thuế đó", "MST đó"
    re.compile(r"\b(mã số thuế|MST|mst)\s+(đó|này|trên|kia)\b", re.IGNORECASE),
    # "công ty vừa rồi", "DN vừa phân tích"
    re.compile(r"\b(công ty|DN|doanh nghiệp)\s+(vừa rồi|vừa nãy|vừa phân tích|trước đó)\b", re.IGNORECASE),
]

# Ellipsis patterns: incomplete questions that need expansion
ELLIPSIS_PATTERNS = [
    # "còn VAT thì sao?", "thế VAT?"
    re.compile(r"^(còn|thế|vậy|rồi)\s+(.+?)\s*(thì sao|thế nào|ra sao|như thế nào)?\s*\??$", re.IGNORECASE),
    # "hóa đơn thì sao?"
    re.compile(r"^(.+?)\s+(thì sao|thế nào|ra sao|như thế nào)\s*\??$", re.IGNORECASE),
    # "entso với cái trước?"
    re.compile(r"^(so sánh|so với|compare)\s*(với\s*)?(cái\s*)?(trước|trên|kia)\s*\??$", re.IGNORECASE),
]

# Comparison patterns
COMPARISON_PATTERNS = [
    # "so sánh A với B"
    re.compile(r"(so sánh|compare|đối chiếu)\s+(.+?)\s+(với|và|vs\.?|so với)\s+(.+)", re.IGNORECASE),
    # "A và B khác gì nhau?"
    re.compile(r"(.+?)\s+(và|với)\s+(.+?)\s+(khác|giống|khác nhau|giống nhau|chênh lệch)", re.IGNORECASE),
]

# Follow-up patterns: questions building on previous context
FOLLOWUP_PATTERNS = [
    re.compile(r"^(vậy|thế|rồi|còn|ngoài ra|bên cạnh đó|thêm nữa)", re.IGNORECASE),
    re.compile(r"^(chi tiết hơn|giải thích thêm|nói rõ hơn|phân tích sâu hơn)", re.IGNORECASE),
    re.compile(r"^(tại sao|vì sao|lý do|nguyên nhân)", re.IGNORECASE),
    re.compile(r"\b(ở trên|vừa nãy|bạn nói|bạn vừa)\b", re.IGNORECASE),
]

# Ambiguity patterns: queries too vague to process
AMBIGUOUS_PATTERNS = [
    # Very short queries
    re.compile(r"^.{1,5}$"),
    # Just a greeting
    re.compile(r"^(hi|hello|xin chào|chào|hey)\s*[!.]?\s*$", re.IGNORECASE),
]

# Topic keywords for ellipsis expansion
TOPIC_EXPANSION: dict[str, str] = {
    "vat": "phân tích rủi ro VAT hoàn thuế",
    "hoàn thuế": "phân tích rủi ro hoàn thuế VAT",
    "hóa đơn": "phân tích rủi ro hóa đơn",
    "nợ đọng": "dự báo rủi ro nợ đọng thuế",
    "nợ": "dự báo rủi ro nợ đọng thuế",
    "giao dịch": "phân tích mẫu giao dịch đáng ngờ",
    "sở hữu": "phân tích cấu trúc sở hữu",
    "ownership": "phân tích cấu trúc sở hữu",
    "gnn": "phân tích rủi ro bằng Graph Neural Network",
    "rủi ro": "đánh giá rủi ro tổng thể",
    "risk": "đánh giá rủi ro tổng thể",
    "thuế": "tra cứu thông tin thuế",
}

# MST pattern for extraction
MST_PATTERN = re.compile(r"\b(\d{10}(?:-\d{3})?)\b")


# ════════════════════════════════════════════════════════════════
#  Main Intelligence Engine
# ════════════════════════════════════════════════════════════════

class ConversationIntelligence:
    """
    Advanced conversation management layer.

    Sits between user input and the orchestrator to resolve ambiguities,
    expand ellipses, and handle coreferences before intent classification.

    Usage:
        ci = ConversationIntelligence()
        result = ci.process(
            message="còn VAT thì sao?",
            active_tax_code="0312345678",
            recent_turns=[...],
            active_entities=[...],
        )
        if result.is_ambiguous:
            return clarification_response(result.clarification_prompt)
        else:
            proceed_with(result.resolved_message)
    """

    def process(
        self,
        message: str,
        active_tax_code: str | None = None,
        recent_turns: list[dict[str, Any]] | None = None,
        active_entities: list[Any] | None = None,
        intent_history: list[str] | None = None,
    ) -> ConversationIntelligenceResult:
        """
        Process a user message through the intelligence pipeline.

        Args:
            message: Raw user input.
            active_tax_code: Currently active MST from memory.
            recent_turns: Recent conversation turns for context.
            active_entities: Tracked entities from memory.
            intent_history: Previous intents in this session.

        Returns:
            ConversationIntelligenceResult with resolved message.
        """
        recent_turns = recent_turns or []
        active_entities = active_entities or []
        intent_history = intent_history or []

        resolved = message.strip()
        resolved_entities: dict[str, str] = {}
        is_followup = False
        is_comparison = False
        is_ambiguous = False
        ambiguity_reason = None
        clarification_prompt = None
        comparison_targets: list[str] = []

        # Step 1: Check if it's a comparison query
        is_comparison, comparison_targets, resolved = self._check_comparison(resolved)

        # Step 2: Resolve coreferences ("công ty đó" → MST)
        if not is_comparison:
            resolved, coref_resolved = self._resolve_coreferences(
                resolved, active_tax_code, recent_turns,
            )
            resolved_entities.update(coref_resolved)

        # Step 3: Expand ellipsis ("VAT thì sao?" → full question)
        resolved, was_expanded = self._expand_ellipsis(
            resolved, active_tax_code, intent_history,
        )
        if was_expanded:
            resolved_entities["ellipsis_expanded"] = "true"

        # Step 4: Detect follow-up
        is_followup = self._is_followup(message, recent_turns)

        # Step 5: Detect ambiguity
        is_ambiguous, ambiguity_reason, clarification_prompt = (
            self._detect_ambiguity(message, active_tax_code, recent_turns)
        )

        # Step 6: If we have an active tax code but message doesn't mention it,
        # and message seems to be about a company — inject context
        if active_tax_code and not MST_PATTERN.search(resolved):
            # Check if message is asking about a company-related topic
            company_keywords = [
                "rủi ro", "risk", "phân tích", "hóa đơn", "nợ", "giao dịch",
                "thuế", "vat", "hoàn", "sở hữu", "đánh giá", "kiểm tra",
            ]
            if any(kw in resolved.lower() for kw in company_keywords) and is_followup:
                resolved = f"{resolved} (MST: {active_tax_code})"
                resolved_entities["injected_tax_code"] = active_tax_code

        result = ConversationIntelligenceResult(
            original_message=message,
            resolved_message=resolved,
            is_followup=is_followup,
            is_comparison=is_comparison,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            clarification_prompt=clarification_prompt,
            resolved_entities=resolved_entities,
            comparison_targets=comparison_targets,
        )

        if resolved != message:
            logger.info(
                "[ConvIntel] Resolved: '%s' → '%s' (followup=%s, comparison=%s, ambiguous=%s)",
                message[:50], resolved[:80], is_followup, is_comparison, is_ambiguous,
            )

        return result

    # ─── Coreference Resolution ──────────────────────────────────

    def _resolve_coreferences(
        self,
        message: str,
        active_tax_code: str | None,
        recent_turns: list[dict],
    ) -> tuple[str, dict[str, str]]:
        """Resolve pronouns and references to concrete entities."""
        resolved = message
        resolved_entities: dict[str, str] = {}

        if not active_tax_code:
            return resolved, resolved_entities

        for pattern in COREFERENCE_PATTERNS:
            match = pattern.search(resolved)
            if match:
                original_ref = match.group(0)
                # Replace coreference with actual entity
                replacement = f"doanh nghiệp MST {active_tax_code}"
                resolved = resolved.replace(original_ref, replacement, 1)
                resolved_entities["coreference"] = f"{original_ref} → {active_tax_code}"
                logger.debug("[ConvIntel] Coreference: '%s' → '%s'", original_ref, replacement)
                break  # Only resolve first coreference

        return resolved, resolved_entities

    # ─── Ellipsis Expansion ──────────────────────────────────────

    def _expand_ellipsis(
        self,
        message: str,
        active_tax_code: str | None,
        intent_history: list[str],
    ) -> tuple[str, bool]:
        """Expand incomplete/elliptical questions into full queries."""
        for pattern in ELLIPSIS_PATTERNS:
            match = pattern.match(message)
            if match:
                groups = match.groups()
                # Extract the topic from the ellipsis
                topic = None
                for g in groups:
                    if g and g.lower() not in ("còn", "thế", "vậy", "rồi", "thì sao", "thế nào", "ra sao", "như thế nào"):
                        topic = g.strip()
                        break

                if topic:
                    # Try to expand the topic
                    expansion = None
                    for keyword, full_query in TOPIC_EXPANSION.items():
                        if keyword.lower() in topic.lower():
                            expansion = full_query
                            break

                    if expansion:
                        if active_tax_code:
                            expanded = f"{expansion} cho doanh nghiệp MST {active_tax_code}"
                        else:
                            expanded = expansion
                        logger.debug("[ConvIntel] Ellipsis: '%s' → '%s'", message, expanded)
                        return expanded, True

        return message, False

    # ─── Comparison Detection ────────────────────────────────────

    def _check_comparison(self, message: str) -> tuple[bool, list[str], str]:
        """Detect and parse comparison queries."""
        for pattern in COMPARISON_PATTERNS:
            match = pattern.search(message)
            if match:
                groups = match.groups()
                # Extract the two entities being compared
                targets = []
                for g in groups:
                    if g and g.lower() not in (
                        "so sánh", "compare", "đối chiếu",
                        "với", "và", "vs", "vs.",
                        "khác", "giống", "khác nhau", "giống nhau", "chênh lệch",
                        "so với",
                    ):
                        cleaned = g.strip()
                        if cleaned:
                            targets.append(cleaned)

                if len(targets) >= 2:
                    return True, targets[:2], message

        return False, [], message

    # ─── Follow-up Detection ─────────────────────────────────────

    def _is_followup(
        self,
        message: str,
        recent_turns: list[dict],
    ) -> bool:
        """Detect if this is a follow-up question."""
        if not recent_turns:
            return False

        # Check follow-up patterns
        for pattern in FOLLOWUP_PATTERNS:
            if pattern.search(message):
                return True

        # Very short message after a long conversation → likely follow-up
        if len(message.split()) <= 5 and len(recent_turns) >= 2:
            return True

        return False

    # ─── Ambiguity Detection ─────────────────────────────────────

    def _detect_ambiguity(
        self,
        message: str,
        active_tax_code: str | None,
        recent_turns: list[dict],
    ) -> tuple[bool, str | None, str | None]:
        """
        Detect if the message is too ambiguous to process.

        Returns:
            (is_ambiguous, reason, clarification_prompt)
        """
        # Check ambiguous patterns
        for pattern in AMBIGUOUS_PATTERNS:
            if pattern.match(message.strip()):
                return (
                    True,
                    "query_too_short",
                    "Xin lỗi, câu hỏi quá ngắn. Bạn có thể cung cấp thêm chi tiết không? "
                    "Ví dụ: 'Phân tích rủi ro doanh nghiệp MST 0312345678'",
                )

        # No tax code and message asks about a specific company without MST
        company_analysis_keywords = [
            "phân tích", "đánh giá", "kiểm tra", "tra cứu",
            "rủi ro", "hóa đơn", "nợ đọng",
        ]
        has_analysis_keyword = any(kw in message.lower() for kw in company_analysis_keywords)
        has_mst = bool(MST_PATTERN.search(message))

        # Legal/general questions that do NOT need a specific MST
        legal_general_keywords = [
            "thuế suất", "thuế gtgt", "thuế tncn", "thuế tndn", "thuế thu nhập",
            "thuế vat", "thuế môn bài", "biểu thuế", "giảm trừ",
            "quy định", "luật", "nghị định", "thông tư", "công văn",
            "đóng thuế", "nộp thuế", "kê khai", "hoàn thuế",
            "hộ kinh doanh", "cá nhân kinh doanh", "freelance",
            "shopee", "lazada", "tiktok", "online", "bán hàng",
            "cho thuê nhà", "cho thuê", "tiền lương", "lương",
            "chi phí được trừ", "chi phí", "chứng từ", "tiền mặt",
            "phạt", "xử phạt", "chậm nộp", "trễ hạn",
            "khấu trừ", "người phụ thuộc", "gia cảnh",
            "tạp hóa", "tiệm", "quán", "cửa hàng",
        ]
        is_legal_general = any(kw in message.lower() for kw in legal_general_keywords)

        if has_analysis_keyword and not has_mst and not active_tax_code and not is_legal_general:
            return (
                True,
                "missing_tax_code",
                "Bạn muốn phân tích doanh nghiệp nào? Vui lòng cung cấp Mã Số Thuế (MST). "
                "Ví dụ: 'Phân tích MST 0312345678'",
            )

        return False, None, None
