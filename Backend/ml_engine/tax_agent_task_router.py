"""
tax_agent_task_router.py - intent focus guard for the multi-agent orchestrator.

The router converts classifier intent into an answer contract and a constrained
tool/sub-agent scope. This prevents broad "full" mode from drifting into legal
consultation when the user asked for a direct data answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerContract(str, Enum):
    SMALLTALK = "smalltalk"
    DATA_TABLE = "data_table"
    RISK_PROFILE = "risk_profile"
    LEGAL_CONSULTATION = "legal_consultation"
    VAT_GRAPH = "vat_graph"
    FILE_ANALYSIS = "file_analysis"
    CLARIFICATION = "clarification"


LEGAL_HINTS = [
    "can cu", "phap ly", "quy dinh", "luat", "nghi dinh", "thong tu",
    "cong van", "dieu", "khoan", "diem", "muc phat", "xu phat",
    "legal", "law", "citation",
]

DATA_TABLE_INTENTS = {"top_n_query", "company_name_lookup"}
LEGAL_INTENTS = {"general_tax_query", "transfer_pricing"}
VAT_GRAPH_INTENTS = {"vat_network_analysis", "vat_refund_risk"}
FILE_INTENTS = {"batch_analysis", "invoice_risk"}


@dataclass
class RoutingDecision:
    intent: str
    answer_contract: AnswerContract
    allowed_tools: set[str] | None = None
    suppressed_tools: set[str] = field(default_factory=set)
    allow_legal: bool = True
    route_confidence: float = 0.75
    focus_score: float = 1.0
    route_violation: bool = False
    reason: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "answer_contract": self.answer_contract.value,
            "allowed_tools": sorted(self.allowed_tools) if self.allowed_tools is not None else None,
            "suppressed_tools": sorted(self.suppressed_tools),
            "allow_legal": self.allow_legal,
            "route_confidence": round(float(self.route_confidence), 4),
            "focus_score": round(float(self.focus_score), 4),
            "route_violation": bool(self.route_violation),
            "reason": self.reason,
        }


class TaskRouter:
    """Deterministic routing policy between intent classification and planning."""

    def route(
        self,
        *,
        query: str,
        intent: str,
        model_mode: str = "full",
        has_attachment: bool = False,
    ) -> RoutingDecision:
        normalized = self._normalize(query)
        legal_requested = any(hint in normalized for hint in LEGAL_HINTS)

        if intent == "smalltalk":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.SMALLTALK,
                allowed_tools=set(),
                allow_legal=False,
                route_confidence=0.98,
                reason="dialogue_act",
            )

        if has_attachment or intent in FILE_INTENTS:
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.FILE_ANALYSIS,
                allowed_tools=None,
                allow_legal=legal_requested or model_mode == "legal",
                route_confidence=0.82,
                reason="file_or_document_analysis",
            )

        if intent == "top_n_query":
            allow_legal = legal_requested or model_mode == "legal"
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.LEGAL_CONSULTATION if allow_legal else AnswerContract.DATA_TABLE,
                allowed_tools={"top_n_risky_companies", "knowledge_search"} if allow_legal else {"top_n_risky_companies"},
                suppressed_tools=set() if allow_legal else {"knowledge_search"},
                allow_legal=allow_legal,
                route_confidence=0.94,
                reason="top_n_direct_data",
            )

        if intent in DATA_TABLE_INTENTS and not legal_requested:
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.DATA_TABLE,
                allowed_tools={"company_name_search", "company_risk_lookup", "top_n_risky_companies"},
                suppressed_tools={"knowledge_search"},
                allow_legal=False,
                route_confidence=0.88,
                reason="data_lookup",
            )

        if intent in VAT_GRAPH_INTENTS and "mang luoi" in normalized:
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.VAT_GRAPH,
                allowed_tools=None,
                allow_legal=legal_requested,
                route_confidence=0.82,
                reason="vat_graph",
            )

        if intent in LEGAL_INTENTS or legal_requested or model_mode == "legal":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.LEGAL_CONSULTATION,
                allowed_tools=None,
                allow_legal=True,
                route_confidence=0.82,
                reason="legal_requested",
            )

        return RoutingDecision(
            intent=intent,
            answer_contract=AnswerContract.RISK_PROFILE,
            allowed_tools=None,
            allow_legal=False,
            route_confidence=0.72,
            reason="risk_or_analysis_default",
        )

    def evaluate_focus(
        self,
        *,
        decision: RoutingDecision,
        selected_tools: list[str],
        answer_text: str = "",
    ) -> RoutingDecision:
        """Score whether selected tools/answer obeyed the answer contract."""
        selected = set(selected_tools or [])
        violation = False
        penalty = 0.0

        if not decision.allow_legal and "knowledge_search" in selected:
            violation = True
            penalty += 0.35

        if decision.allowed_tools is not None:
            unexpected = selected - decision.allowed_tools
            if unexpected:
                violation = True
                penalty += min(0.45, 0.15 * len(unexpected))
                decision.suppressed_tools.update(unexpected)

        text = self._normalize(answer_text)
        if decision.answer_contract == AnswerContract.DATA_TABLE:
            legal_markers = {"tu van phap ly", "can cu phap ly", "cong van", "quyet dinh", "nghi dinh"}
            if any(marker in text for marker in legal_markers) and not decision.allow_legal:
                violation = True
                penalty += 0.25

        decision.route_violation = violation
        decision.focus_score = max(0.0, round(1.0 - penalty, 4))
        return decision

    @staticmethod
    def _normalize(value: str) -> str:
        try:
            import unicodedata

            normalized = unicodedata.normalize("NFD", value or "")
            stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
            stripped = stripped.replace("đ", "d").replace("Đ", "D")
        except Exception:
            stripped = value or ""
        stripped = re.sub(r"[^\w\s]", " ", stripped.lower())
        return re.sub(r"\s+", " ", stripped).strip()
