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

from ml_engine.tax_agent_mode_contracts import AgentModeContractRegistry


class AnswerContract(str, Enum):
    SMALLTALK = "smalltalk"
    DATA_TABLE = "data_table"
    RISK_PROFILE = "risk_profile"
    FRAUD_ANALYSIS = "fraud_analysis"
    LEGAL_CONSULTATION = "legal_consultation"
    VAT_GRAPH = "vat_graph"
    FILE_ANALYSIS = "file_analysis"
    CLARIFICATION = "clarification"
    MODE_MISMATCH = "mode_mismatch"


LEGAL_HINTS = [
    "can cu", "phap ly", "quy dinh", "luat ", "nghi dinh", "thong tu",
    "cong van", "muc phat", "xu phat", "legal", "law", "citation",
    # Everyday citizen/business tax questions often omit "legal" wording.
    "tncn", "tndn", "nguoi phu thuoc", "thue mon bai",
    "nop to khai", "dong thue", "ho kinh doanh",
    "ban hang online", "thuong mai dien tu", "shopee",
    "tiktok", "cho thue nha", "chi phi duoc tru", "hoa don dien tu",
]

DATA_TABLE_INTENTS = {"top_n_query", "company_name_lookup"}
LEGAL_INTENTS = {"general_tax_query", "transfer_pricing"}
VAT_GRAPH_INTENTS = {"vat_network_analysis", "vat_refund_risk", "invoice_risk"}
FILE_INTENTS = {"batch_analysis", "invoice_risk"}
DOMAIN_MODES = {"legal", "fraud", "vat", "delinquency", "macro"}

MODEL_CAPABILITY_REGISTRY: dict[str, list[str]] = AgentModeContractRegistry.capability_registry()
DOMAIN_ALLOWED_TOOLS: dict[str, set[str]] = AgentModeContractRegistry.domain_allowed_tools()


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
    requested_domain: str = "general"
    selected_model_bundle: list[str] = field(default_factory=list)
    mode_validation: dict[str, Any] = field(default_factory=dict)
    mode_mismatch: bool = False
    suggested_mode: str | None = None
    suppressed_domains: list[str] = field(default_factory=list)
    route_reason: str = ""

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
            "requested_domain": self.requested_domain,
            "selected_model_bundle": list(self.selected_model_bundle),
            "mode_validation": dict(self.mode_validation or {}),
            "mode_mismatch": bool(self.mode_mismatch),
            "suggested_mode": self.suggested_mode,
            "suppressed_domains": list(self.suppressed_domains or []),
            "route_reason": self.route_reason or self.reason,
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
        attachment_analysis: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        normalized = self._normalize(query)
        mode = (model_mode or "full").lower()
        legal_requested = mode == "legal" or any(hint in normalized for hint in LEGAL_HINTS)
        requested_domain = self._infer_domain(
            normalized=normalized,
            intent=intent,
            legal_requested=legal_requested,
            attachment_analysis=attachment_analysis,
        )
        vat_legal_requested = (
            (mode == "vat" or requested_domain == "vat")
            and legal_requested
            and any(hint in normalized for hint in ("vat", "hoa don", "hoan thue", "khau tru", "thue gtgt"))
            and intent in LEGAL_INTENTS
        )
        if vat_legal_requested:
            requested_domain = "vat"
        selected_bundle = list(MODEL_CAPABILITY_REGISTRY.get(requested_domain, MODEL_CAPABILITY_REGISTRY["general"]))
        if vat_legal_requested:
            selected_bundle = list(dict.fromkeys(selected_bundle + MODEL_CAPABILITY_REGISTRY["legal"]))
        suppressed_domains = sorted(DOMAIN_MODES - {requested_domain}) if requested_domain in DOMAIN_MODES else []
        mode_validation = {
            "model_mode": mode,
            "requested_domain": requested_domain,
            "valid": True,
            "reason": "auto_or_explicit_match" if mode == "full" else "explicit_mode_match",
        }

        if intent == "smalltalk":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.SMALLTALK,
                allowed_tools=set(),
                allow_legal=False,
                route_confidence=0.98,
                reason="dialogue_act",
                requested_domain="general",
                selected_model_bundle=MODEL_CAPABILITY_REGISTRY["general"],
                mode_validation=mode_validation,
            )

        if mode in DOMAIN_MODES and requested_domain in DOMAIN_MODES and mode != requested_domain:
            suggested = requested_domain
            mode_validation = {
                "model_mode": mode,
                "requested_domain": requested_domain,
                "valid": False,
                "reason": "selected_mode_does_not_match_request_domain",
            }
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.MODE_MISMATCH,
                allowed_tools=set(),
                suppressed_tools=set().union(*DOMAIN_ALLOWED_TOOLS.values()),
                allow_legal=False,
                route_confidence=0.96,
                reason="mode_guard",
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                mode_mismatch=True,
                suggested_mode=suggested,
                suppressed_domains=suppressed_domains,
                route_reason=f"Mode '{mode}' cannot safely handle domain '{requested_domain}'.",
            )

        if intent == "top_n_query":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.LEGAL_CONSULTATION if legal_requested else AnswerContract.DATA_TABLE,
                allowed_tools={"top_n_risky_companies", "knowledge_search"} if legal_requested else {"top_n_risky_companies"},
                suppressed_tools=set() if legal_requested else {"knowledge_search"},
                allow_legal=legal_requested,
                route_confidence=0.94,
                reason="top_n_direct_data",
                requested_domain="legal" if legal_requested else "fraud",
                selected_model_bundle=MODEL_CAPABILITY_REGISTRY["legal" if legal_requested else "fraud"],
                mode_validation=mode_validation,
                suppressed_domains=sorted(DOMAIN_MODES - {"legal" if legal_requested else "fraud"}),
            )

        if has_attachment or intent in FILE_INTENTS:
            if requested_domain == "vat":
                contract = AnswerContract.VAT_GRAPH
            elif requested_domain == "legal":
                contract = AnswerContract.LEGAL_CONSULTATION
            else:
                contract = AnswerContract.FRAUD_ANALYSIS
            return RoutingDecision(
                intent=intent,
                answer_contract=contract,
                allowed_tools=set(),
                suppressed_tools={"knowledge_search"} if requested_domain != "legal" else set(),
                allow_legal=requested_domain == "legal",
                route_confidence=0.88,
                reason="file_or_document_analysis",
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                suppressed_domains=suppressed_domains,
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
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                suppressed_domains=suppressed_domains,
            )

        if requested_domain == "vat":
            if vat_legal_requested:
                return RoutingDecision(
                    intent=intent,
                    answer_contract=AnswerContract.LEGAL_CONSULTATION,
                    allowed_tools=set(DOMAIN_ALLOWED_TOOLS["vat"]) | {"knowledge_search"},
                    allow_legal=True,
                    route_confidence=0.84,
                    reason="vat_legal_requested",
                    requested_domain=requested_domain,
                    selected_model_bundle=selected_bundle,
                    mode_validation=mode_validation,
                    suppressed_domains=suppressed_domains,
                )
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.VAT_GRAPH,
                allowed_tools=set(DOMAIN_ALLOWED_TOOLS["vat"]),
                suppressed_tools={"knowledge_search"},
                allow_legal=False,
                route_confidence=0.82,
                reason="vat_graph",
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                suppressed_domains=suppressed_domains,
            )

        if requested_domain == "legal":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.LEGAL_CONSULTATION,
                allowed_tools=set(DOMAIN_ALLOWED_TOOLS["legal"]),
                allow_legal=True,
                route_confidence=0.82,
                reason="legal_requested",
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                suppressed_domains=suppressed_domains,
            )

        if requested_domain == "delinquency":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.RISK_PROFILE,
                allowed_tools=set(DOMAIN_ALLOWED_TOOLS["delinquency"]),
                suppressed_tools={"knowledge_search"},
                allow_legal=False,
                route_confidence=0.8,
                reason="delinquency_analysis",
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                suppressed_domains=suppressed_domains,
            )

        if requested_domain == "macro":
            return RoutingDecision(
                intent=intent,
                answer_contract=AnswerContract.RISK_PROFILE,
                allowed_tools=set(DOMAIN_ALLOWED_TOOLS["macro"]),
                suppressed_tools={"knowledge_search"},
                allow_legal=False,
                route_confidence=0.78,
                reason="macro_analysis",
                requested_domain=requested_domain,
                selected_model_bundle=selected_bundle,
                mode_validation=mode_validation,
                suppressed_domains=suppressed_domains,
            )

        return RoutingDecision(
            intent=intent,
            answer_contract=AnswerContract.FRAUD_ANALYSIS if intent in {"batch_analysis", "audit_selection"} else AnswerContract.RISK_PROFILE,
            allowed_tools=set(DOMAIN_ALLOWED_TOOLS["fraud"]),
            suppressed_tools={"knowledge_search"},
            allow_legal=False,
            route_confidence=0.72,
            reason="risk_or_analysis_default",
            requested_domain=requested_domain,
            selected_model_bundle=selected_bundle,
            mode_validation=mode_validation,
            suppressed_domains=suppressed_domains,
        )

    def evaluate_focus(
        self,
        *,
        decision: RoutingDecision,
        selected_tools: list[str],
        answer_text: str = "",
    ) -> RoutingDecision:
        """Score whether selected tools/answer obeyed the answer contract."""
        selected = {tool for tool in (selected_tools or []) if not str(tool).startswith("_")}
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

    def _infer_domain(
        self,
        *,
        normalized: str,
        intent: str,
        legal_requested: bool,
        attachment_analysis: dict[str, Any] | None,
    ) -> str:
        if attachment_analysis:
            explicit = str(attachment_analysis.get("requested_domain") or "").lower()
            if explicit in DOMAIN_MODES:
                return explicit
            analysis_type = str(
                attachment_analysis.get("analysis_type")
                or attachment_analysis.get("detected_schema")
                or ""
            ).lower()
            detected_schema = attachment_analysis.get("detected_schema")
            if isinstance(detected_schema, dict):
                analysis_type = str(detected_schema.get("detected_schema") or analysis_type).lower()
            if legal_requested and analysis_type in {"ocr_invoice", "document", "pdf", "image"}:
                return "legal"
            if analysis_type in {"risk_csv", "risk_scoring_csv"}:
                return "fraud"
            if analysis_type in {"vat_graph_csv", "ocr_invoice"}:
                return "vat"

        # Check delinquency and macro BEFORE legal fallback, because
        # delinquency queries often contain words like "thuế" that match legal hints.
        if intent == "delinquency" or "no dong" in normalized or "tre han" in normalized or "cham nop" in normalized:
            return "delinquency"
        if "vi mo" in normalized or "macro" in normalized or intent == "macro_forecast" or "mo phong" in normalized or "kich ban" in normalized:
            return "macro"

        if legal_requested:
            return "legal"
        if intent == "top_n_query" or intent in {"company_name_lookup", "batch_analysis", "audit_selection"}:
            return "fraud"
        if intent in VAT_GRAPH_INTENTS:
            return "vat"
        if re.fullmatch(r"\d{10}(?: \d{3})?", normalized):
            return "fraud"
        if intent in LEGAL_INTENTS:
            return "legal"
        return "fraud"

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
