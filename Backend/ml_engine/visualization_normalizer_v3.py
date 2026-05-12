from __future__ import annotations

from typing import Any


def normalize_visualization_v3(viz: dict[str, Any] | None) -> dict[str, Any]:
    """
    VisualizationNormalizer v3
    --------------------------
    Unifies domain payloads into a small, chat-friendly envelope.

    Notes:
    - This is intentionally lightweight and backward-compatible: existing viz keys remain unchanged.
    - Frontend can progressively adopt `viz.v3` without breaking older renderers.
    """
    viz = viz or {}
    out: dict[str, Any] = {
        "contract_version": "viz-normalizer-v3",
        "domains": [],
        "cards": [],
        "raw": {},
    }

    def _push(domain: str, payload: Any) -> None:
        if payload is None:
            return
        out["domains"].append(domain)
        out["raw"][domain] = payload

    _push("fraud", viz.get("fraud"))
    _push("vat", viz.get("vat"))
    _push("delinquency", viz.get("delinquency"))
    _push("macro", viz.get("macro"))
    _push("knowledge_graph", viz.get("knowledge_graph"))

    # Provide compact "cards" hints (frontend may ignore)
    if viz.get("fraud"):
        out["cards"].append({"domain": "fraud", "layout": "compact", "max_height": 220})
    if viz.get("vat"):
        out["cards"].append({"domain": "vat", "layout": "compact", "max_height": 220})
    if viz.get("delinquency_timeline") or viz.get("delinquency"):
        out["cards"].append({"domain": "delinquency", "layout": "compact", "max_height": 200})
    if viz.get("macro"):
        out["cards"].append({"domain": "macro", "layout": "compact", "max_height": 240})

    return out

