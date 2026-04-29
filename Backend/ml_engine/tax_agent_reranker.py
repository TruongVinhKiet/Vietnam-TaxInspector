from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RerankWeights:
    model_version: str
    w_bm25: float
    w_dense: float
    w_lexical: float
    w_doc_type: float


class TaxAgentReranker:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.weights = RerankWeights(model_version="tax-agent-rerank-v1", w_bm25=0.35, w_dense=0.45, w_lexical=0.15, w_doc_type=0.05)

    def load(self) -> bool:
        path = self.model_dir / "tax_agent_reranker_weights.json"
        if not path.exists():
            return False
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            self.weights = RerankWeights(
                model_version=str(obj.get("model_version") or self.weights.model_version),
                w_bm25=float(obj.get("w_bm25", self.weights.w_bm25)),
                w_dense=float(obj.get("w_dense", self.weights.w_dense)),
                w_lexical=float(obj.get("w_lexical", self.weights.w_lexical)),
                w_doc_type=float(obj.get("w_doc_type", self.weights.w_doc_type)),
            )
            return True
        except Exception:
            return False

    def rerank(self, items: list[dict[str, Any]], *, preferred_doc_types: list[str] | None = None) -> list[dict[str, Any]]:
        pref = set(preferred_doc_types or [])
        w = self.weights

        def _score(it: dict[str, Any]) -> float:
            comp = it.get("components") or {}
            doc_type = str(it.get("doc_type") or "")
            dt = 1.0 if (pref and doc_type in pref) else 0.0
            return (
                w.w_bm25 * float(comp.get("bm25") or 0.0)
                + w.w_dense * float(comp.get("dense") or 0.0)
                + w.w_lexical * float(comp.get("lexical") or 0.0)
                + w.w_doc_type * dt
            )

        ranked = list(items)
        ranked.sort(key=_score, reverse=True)
        return ranked

