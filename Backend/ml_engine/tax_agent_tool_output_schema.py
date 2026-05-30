"""
Standardized analytics output envelope for TaxInspector agent tools.

Individual tools keep their domain-specific payloads, but the orchestrator can
attach this normalized summary for dashboards, audits, calibration and active
learning.
"""

from __future__ import annotations

from typing import Any


def _first_number(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _risk_level(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def build_standard_tool_output(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, stable output envelope for any agent tool."""
    raw_score = _first_number(
        payload,
        (
            "risk_score",
            "fraud_probability",
            "prob_90d",
            "anomaly_ratio",
            "ring_score",
            "confidence",
            "score",
        ),
    )
    score = None
    if raw_score is not None:
        score = raw_score / 100.0 if raw_score > 1.0 else raw_score
        score = max(0.0, min(1.0, score))

    evidence_count = 0
    for key in ("hits", "results", "companies", "cases", "motifs", "rings", "top_flags", "top_matches"):
        value = payload.get(key)
        if isinstance(value, list):
            evidence_count += len(value)
        elif isinstance(value, dict):
            evidence_count += len(value)

    return {
        "schema_version": "tool-output-v1",
        "tool_name": tool_name,
        "status": str(payload.get("status") or "success"),
        "score": None if score is None else round(score, 4),
        "risk_level": _risk_level(score),
        "uncertainty": None if score is None else round(1.0 - abs(score - 0.5) * 2.0, 4),
        "evidence_count": evidence_count,
        "recommended_action": payload.get("recommended_action") or payload.get("action") or "",
        "explanation": payload.get("message") or payload.get("summary") or payload.get("reason") or "",
    }
