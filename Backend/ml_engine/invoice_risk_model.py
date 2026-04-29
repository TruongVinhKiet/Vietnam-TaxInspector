from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List
import json

import joblib
import numpy as np


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

    MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
    FEATURE_COLS = (
        "amount_log",
        "vat_rate",
        "payment_status_overdue",
        "payment_status_failed",
        "is_adjustment",
        "event_count",
        "near_dup_count",
        "same_day_pair_count",
        "seller_risk_score",
        "buyer_risk_score",
        "counterparty_gap",
    )

    def __init__(self, model_version: str = "invoice-risk-heuristic-v1"):
        self.model_version = model_version
        self.model = None
        self.model_config: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = self.MODEL_DIR / "invoice_risk_model.joblib"
        config_path = self.MODEL_DIR / "invoice_risk_config.json"
        if not model_path.exists():
            return
        try:
            self.model = joblib.load(model_path)
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self.model_config = json.load(f)
            self.model_version = str(self.model_config.get("model_version") or "invoice-risk-learned-v1")
        except Exception:
            self.model = None
            self.model_config = {}

    def _build_feature_dict(self, invoice: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        amount = float(invoice.get("amount") or 0.0)
        vat_rate = float(invoice.get("vat_rate") or 0.0)
        payment_status = str(invoice.get("payment_status") or "unknown").lower()
        seller_risk = float(context.get("seller_risk_score") or 0.0)
        buyer_risk = float(context.get("buyer_risk_score") or 0.0)
        return {
            "amount_log": float(np.log1p(max(amount, 0.0))),
            "vat_rate": vat_rate,
            "payment_status_overdue": 1.0 if payment_status in {"overdue", "partial"} else 0.0,
            "payment_status_failed": 1.0 if payment_status in {"failed", "rejected"} else 0.0,
            "is_adjustment": 1.0 if bool(invoice.get("is_adjustment")) else 0.0,
            "event_count": float(context.get("event_count") or 0.0),
            "near_dup_count": float(context.get("near_dup_count") or 0.0),
            "same_day_pair_count": float(context.get("same_day_pair_count") or 0.0),
            "seller_risk_score": seller_risk,
            "buyer_risk_score": buyer_risk,
            "counterparty_gap": abs(seller_risk - buyer_risk),
        }

    def _predict_learned(self, invoice: Dict[str, Any], context: Dict[str, Any]) -> InvoiceRiskResult | None:
        if self.model is None:
            return None
        features = self._build_feature_dict(invoice, context)
        vec = np.asarray([[features.get(col, 0.0) for col in self.FEATURE_COLS]], dtype=float)
        try:
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(vec)[0][1])
            else:
                prob = float(self.model.predict(vec)[0])
        except Exception:
            return None

        score = round(max(0.0, min(100.0, prob * 100.0)), 2)
        if score >= 80:
            level = "critical"
        elif score >= 60:
            level = "high"
        elif score >= 35:
            level = "medium"
        else:
            level = "low"

        importance_pairs = []
        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None:
            for name, weight in sorted(zip(self.FEATURE_COLS, importances), key=lambda item: abs(item[1]), reverse=True)[:4]:
                importance_pairs.append({"feature": name, "importance": round(float(weight), 4), "value": round(float(features.get(name, 0.0)), 4)})

        reason_codes = [item["feature"] for item in importance_pairs[:3]] or ["learned_invoice_pattern"]
        return InvoiceRiskResult(
            invoice_number=str(invoice.get("invoice_number") or ""),
            risk_score=score,
            risk_level=level,
            reason_codes=reason_codes,
            explanations={
                "mode": "learned_model",
                "feature_values": features,
                "top_features": importance_pairs,
                "model_confidence": round(max(prob, 1 - prob) * 100.0, 2),
            },
            linked_invoice_ids=list(context.get("linked_invoice_ids") or []),
            model_version=self.model_version,
        )

    def score(self, invoice: Dict[str, Any], context: Dict[str, Any]) -> InvoiceRiskResult:
        learned = self._predict_learned(invoice, context)
        if learned is not None:
            return learned

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

