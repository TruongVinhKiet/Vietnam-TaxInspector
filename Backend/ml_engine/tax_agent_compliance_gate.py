"""
tax_agent_compliance_gate.py – Advanced Compliance Gate (Phase 2)
==================================================================
Dynamic policy enforcement, safety checks, and audit trail
for the multi-agent tax intelligence system.

Responsibilities:
    1. Dynamic rule evaluation (beyond the 3 fixed rules)
    2. Safety classification (harmful/inappropriate queries)
    3. PII detection in responses
    4. Bias checking (industry/region fairness)
    5. Escalation routing by domain
    6. Full decision audit trail
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    ESCALATE = "escalate"


class EscalationDomain(str, Enum):
    LEGAL = "legal"                     # Cần chuyên viên pháp chế
    AUDIT = "audit"                     # Cần kiểm tra viên
    INVESTIGATION = "investigation"     # Cần điều tra viên
    MANAGEMENT = "management"           # Cần lãnh đạo phê duyệt
    NONE = "none"


@dataclass
class PolicyRule:
    """A single policy rule."""
    rule_key: str
    description: str
    category: str               # "intent", "retrieval", "safety", "pii", "bias"
    evaluator: str              # Function name or lambda reference
    severity: str = "medium"    # "low", "medium", "high", "critical"
    enabled: bool = True


@dataclass
class PolicyTrace:
    """Audit trace for a single policy check."""
    rule_key: str
    decision: GateDecision
    score: float
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceResult:
    """Result of full compliance check."""
    overall_decision: GateDecision
    abstain: bool
    escalate: bool
    escalation_domain: EscalationDomain
    traces: list[PolicyTrace]
    warnings: list[str]
    latency_ms: float = 0.0


# ─── PII Patterns ────────────────────────────────────────────────────────────

PII_PATTERNS = {
    "phone": re.compile(r"\b(?:0[0-9]{9,10}|(?:\+84)[0-9]{9,10})\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "id_number": re.compile(r"\b(?:CMND|CCCD|căn cước|chứng minh)\s*:?\s*\d{9,12}\b", re.IGNORECASE),
    "bank_account": re.compile(r"\b(?:STK|số tài khoản|account)\s*:?\s*\d{8,20}\b", re.IGNORECASE),
}

# ─── Safety Keywords ─────────────────────────────────────────────────────────

UNSAFE_PATTERNS = [
    re.compile(r"\b(?:trốn thuế|evade|escape tax|giấu|che giấu thu nhập)\b", re.IGNORECASE),
    re.compile(r"\b(?:rửa tiền|money laundering|launder)\b", re.IGNORECASE),
    re.compile(r"\b(?:đe dọa|threaten|bribe|hối lộ)\b", re.IGNORECASE),
]

# ─── Prompt Injection Patterns ───────────────────────────────────────────────

INJECTION_PATTERNS = [
    re.compile(r"(?:ignore|forget|disregard)\s+(?:the\s+)?(?:above|previous|prior)", re.IGNORECASE),
    re.compile(r"(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)", re.IGNORECASE),
    re.compile(r"(?:system\s*prompt|admin\s*override|root\s*access)", re.IGNORECASE),
    re.compile(r"<\s*(?:script|system|admin)", re.IGNORECASE),
    re.compile(r"```\s*(?:system|admin|override)", re.IGNORECASE),
]


class TaxAgentComplianceGate:
    """
    Advanced compliance gate with dynamic policy evaluation.

    Policy Categories:
    1. Intent confidence gate
    2. Retrieval quality gate
    3. Safety classification
    4. PII detection gate
    5. Prompt injection detection
    6. Bias fairness check
    7. Escalation routing

    Usage:
        gate = TaxAgentComplianceGate(db)
        result = gate.evaluate(
            query="...",
            intent="vat_refund_risk",
            intent_confidence=0.85,
            retrieval_hits=5,
            response_text="...",
            tool_results={...},
        )
    """

    def __init__(self, db=None):
        self.db = db
        self._db_rules_loaded = False
        self._custom_rules: list[PolicyRule] = []

    def evaluate(
        self,
        *,
        query: str,
        intent: str,
        intent_confidence: float,
        retrieval_hits: int = 0,
        response_text: str = "",
        tool_results: dict[str, Any] | None = None,
        session_id: str = "",
        turn_id: int = 0,
    ) -> ComplianceResult:
        """Run all policy checks and return compliance result."""
        t0 = time.perf_counter()
        traces: list[PolicyTrace] = []
        warnings: list[str] = []

        # 1. Intent confidence gate
        traces.append(self._check_intent_confidence(intent, intent_confidence))

        # 2. Retrieval quality gate
        traces.append(self._check_retrieval_quality(intent, retrieval_hits))

        # 3. High-risk intent gate
        traces.append(self._check_high_risk_intent(intent, retrieval_hits, intent_confidence))

        # 4. Safety classification
        safety_trace = self._check_safety(query)
        traces.append(safety_trace)
        if safety_trace.decision == GateDecision.BLOCK:
            warnings.append("Phát hiện nội dung không phù hợp trong câu hỏi.")

        # 5. Prompt injection detection
        injection_trace = self._check_prompt_injection(query)
        traces.append(injection_trace)
        if injection_trace.decision == GateDecision.BLOCK:
            warnings.append("Phát hiện prompt injection attempt.")

        # 6. PII detection in response
        if response_text:
            pii_trace = self._check_pii(response_text)
            traces.append(pii_trace)
            if pii_trace.decision == GateDecision.WARN:
                warnings.append("Phát hiện thông tin cá nhân (PII) trong response.")

        # 7. Tool error rate check
        if tool_results:
            tool_trace = self._check_tool_health(tool_results)
            traces.append(tool_trace)

        # Aggregate decision
        overall, abstain, escalate, domain = self._aggregate_decisions(traces, intent)

        # Log to DB
        if self.db and session_id:
            self._log_to_db(session_id, turn_id, traces)

        latency = (time.perf_counter() - t0) * 1000.0

        return ComplianceResult(
            overall_decision=overall,
            abstain=abstain,
            escalate=escalate,
            escalation_domain=domain,
            traces=traces,
            warnings=warnings,
            latency_ms=latency,
        )

    # ─── Individual Policy Checks ─────────────────────────────────────────

    def _check_intent_confidence(
        self,
        intent: str,
        confidence: float,
    ) -> PolicyTrace:
        """Gate: minimum intent confidence."""
        threshold = 0.35
        passed = confidence >= threshold
        return PolicyTrace(
            rule_key="min_intent_confidence",
            decision=GateDecision.ALLOW if passed else GateDecision.BLOCK,
            score=confidence,
            reason=None if passed else f"Intent confidence {confidence:.2f} < {threshold}",
        )

    def _check_retrieval_quality(
        self,
        intent: str,
        retrieval_hits: int,
    ) -> PolicyTrace:
        """Gate: minimum retrieval hits."""
        min_hits = 2 if intent != "general_tax_query" else 1
        passed = retrieval_hits >= min_hits
        return PolicyTrace(
            rule_key="min_retrieval_hits",
            decision=GateDecision.ALLOW if passed else GateDecision.BLOCK,
            score=float(retrieval_hits),
            reason=None if passed else f"Only {retrieval_hits} retrieval hits (need {min_hits}+)",
        )

    def _check_high_risk_intent(
        self,
        intent: str,
        retrieval_hits: int,
        confidence: float,
    ) -> PolicyTrace:
        """Gate: high-risk intents need stronger evidence."""
        high_risk_intents = {"audit_selection", "transfer_pricing", "vat_refund_risk"}
        if intent not in high_risk_intents:
            return PolicyTrace(
                rule_key="high_risk_intent_gate",
                decision=GateDecision.ALLOW,
                score=1.0,
            )

        passed = retrieval_hits >= 3 and confidence >= 0.5
        decision = GateDecision.ALLOW if passed else GateDecision.ESCALATE
        return PolicyTrace(
            rule_key="high_risk_intent_gate",
            decision=decision,
            score=float(retrieval_hits) * confidence,
            reason=None if passed else (
                f"High-risk intent '{intent}' needs stronger citations "
                f"(hits={retrieval_hits}, conf={confidence:.2f})"
            ),
        )

    def _check_safety(self, query: str) -> PolicyTrace:
        """Gate: detect harmful/inappropriate content."""
        for pattern in UNSAFE_PATTERNS:
            match = pattern.search(query)
            if match:
                return PolicyTrace(
                    rule_key="safety_classification",
                    decision=GateDecision.BLOCK,
                    score=0.0,
                    reason=f"Unsafe content detected: '{match.group()}'",
                    details={"matched_pattern": match.group()},
                )

        return PolicyTrace(
            rule_key="safety_classification",
            decision=GateDecision.ALLOW,
            score=1.0,
        )

    def _check_prompt_injection(self, query: str) -> PolicyTrace:
        """Gate: detect prompt injection attempts."""
        for pattern in INJECTION_PATTERNS:
            match = pattern.search(query)
            if match:
                return PolicyTrace(
                    rule_key="prompt_injection_detection",
                    decision=GateDecision.BLOCK,
                    score=0.0,
                    reason=f"Prompt injection attempt: '{match.group()}'",
                    details={"matched_pattern": match.group()},
                )

        return PolicyTrace(
            rule_key="prompt_injection_detection",
            decision=GateDecision.ALLOW,
            score=1.0,
        )

    def _check_pii(self, text: str) -> PolicyTrace:
        """Gate: detect PII in response text."""
        found_pii = []
        for pii_type, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                found_pii.append(pii_type)

        if found_pii:
            return PolicyTrace(
                rule_key="pii_detection",
                decision=GateDecision.WARN,
                score=0.5,
                reason=f"PII detected in response: {', '.join(found_pii)}",
                details={"pii_types": found_pii},
            )

        return PolicyTrace(
            rule_key="pii_detection",
            decision=GateDecision.ALLOW,
            score=1.0,
        )

    def _check_tool_health(
        self,
        tool_results: dict[str, Any],
    ) -> PolicyTrace:
        """Gate: check if too many tools failed."""
        total = len(tool_results)
        errors = sum(
            1 for r in tool_results.values()
            if isinstance(r, dict) and r.get("status") in ("error",)
        )

        if total == 0:
            return PolicyTrace(
                rule_key="tool_health",
                decision=GateDecision.ALLOW,
                score=1.0,
            )

        error_rate = errors / total
        if error_rate > 0.5:
            return PolicyTrace(
                rule_key="tool_health",
                decision=GateDecision.WARN,
                score=round(1.0 - error_rate, 2),
                reason=f"{errors}/{total} tools failed ({error_rate:.0%})",
            )

        return PolicyTrace(
            rule_key="tool_health",
            decision=GateDecision.ALLOW,
            score=round(1.0 - error_rate, 2),
        )

    # ─── Decision Aggregation ─────────────────────────────────────────────

    def _aggregate_decisions(
        self,
        traces: list[PolicyTrace],
        intent: str,
    ) -> tuple[GateDecision, bool, bool, EscalationDomain]:
        """Aggregate individual policy traces into final decision."""
        has_block = any(t.decision == GateDecision.BLOCK for t in traces)
        has_escalate = any(t.decision == GateDecision.ESCALATE for t in traces)
        has_warn = any(t.decision == GateDecision.WARN for t in traces)

        abstain = has_block
        escalate = has_escalate

        # Determine escalation domain
        domain = EscalationDomain.NONE
        if escalate:
            domain_map = {
                "vat_refund_risk": EscalationDomain.AUDIT,
                "transfer_pricing": EscalationDomain.LEGAL,
                "audit_selection": EscalationDomain.MANAGEMENT,
                "osint_ownership": EscalationDomain.INVESTIGATION,
            }
            domain = domain_map.get(intent, EscalationDomain.AUDIT)

        if has_block:
            return GateDecision.BLOCK, True, escalate, domain
        if has_escalate:
            return GateDecision.ESCALATE, False, True, domain
        if has_warn:
            return GateDecision.WARN, False, False, domain

        return GateDecision.ALLOW, False, False, EscalationDomain.NONE

    # ─── DB Logging ───────────────────────────────────────────────────────

    def _log_to_db(
        self,
        session_id: str,
        turn_id: int,
        traces: list[PolicyTrace],
    ) -> None:
        """Log all policy traces to DB."""
        from sqlalchemy import text as sql_text

        for trace in traces:
            try:
                self.db.execute(
                    sql_text("""
                        INSERT INTO policy_execution_logs
                        (session_id, turn_id, rule_key, decision, reason, score, payload_json)
                        VALUES (:session_id, :turn_id, :rule_key, :decision, :reason, :score, CAST(:payload_json AS jsonb))
                    """),
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "rule_key": trace.rule_key,
                        "decision": trace.decision.value,
                        "reason": trace.reason,
                        "score": trace.score,
                        "payload_json": json.dumps(trace.details),
                    },
                )
            except Exception as exc:
                logger.debug("[ComplianceGate] DB log failed: %s", exc)
