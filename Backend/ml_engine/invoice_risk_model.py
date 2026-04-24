from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List


@dataclass
class InvoiceRiskResult:
    invoice_number: str
    risk_score: float
    risk_level: str
    reason_codes: List[str]
    explanations: Dict[str, Any]
    linked_invoice_ids: List[str]
    model_version: str = "invoice-risk-heuristic-v1"


class InvoiceRiskScorer:
    """
    Baseline invoice-level fraud/anomaly scorer.
    This scorer is deterministic and auditable, suitable for initial rollout.
    """

    def __init__(self, model_version: str = "invoice-risk-heuristic-v1"):
        self.model_version = model_version

    def score(self, invoice: Dict[str, Any], context: Dict[str, Any]) -> InvoiceRiskResult:
        amount = float(invoice.get("amount") or 0.0)
        vat_rate = float(invoice.get("vat_rate") or 0.0)
        payment_status = str(invoice.get("payment_status") or "unknown").lower()
        is_adjustment = bool(invoice.get("is_adjustment"))
        event_count = int(context.get("event_count") or 0)
        near_dup_count = int(context.get("near_dup_count") or 0)
        sibling_count = int(context.get("same_day_pair_count") or 0)
        linked_ids = list(context.get("linked_invoice_ids") or [])

        score = 0.0
        reasons: List[str] = []

        if vat_rate <= 0 or vat_rate > 15:
            score += 25
            reasons.append("vat_rate_anomaly")
        if payment_status in {"overdue", "failed", "rejected"}:
            score += 20
            reasons.append("payment_status_risky")
        if is_adjustment:
            score += 10
            reasons.append("adjustment_invoice")
        if event_count >= 2:
            score += min(25, 8 * event_count)
            reasons.append("cancel_replace_pattern")
        if near_dup_count >= 1:
            score += min(20, 10 * near_dup_count)
            reasons.append("near_duplicate_cluster")
        if sibling_count >= 3:
            score += 15
            reasons.append("smurfing_same_day")
        if amount >= 5_000_000_000:
            score += 15
            reasons.append("large_amount_outlier")

        score = max(0.0, min(100.0, score))
        if score >= 75:
            level = "critical"
        elif score >= 55:
            level = "high"
        elif score >= 30:
            level = "medium"
        else:
            level = "low"

        return InvoiceRiskResult(
            invoice_number=str(invoice.get("invoice_number") or ""),
            risk_score=round(score, 2),
            risk_level=level,
            reason_codes=reasons,
            explanations={
                "amount": amount,
                "vat_rate": vat_rate,
                "payment_status": payment_status,
                "event_count": event_count,
                "near_dup_count": near_dup_count,
                "same_day_pair_count": sibling_count,
            },
            linked_invoice_ids=linked_ids,
            model_version=self.model_version,
        )

