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
        self._cross_encoder = None

    def load(self) -> bool:
        path = self.model_dir / "tax_agent_reranker_weights.json"
        loaded = False
        try:
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                self.weights = RerankWeights(
                    model_version=str(obj.get("model_version") or self.weights.model_version),
                    w_bm25=float(obj.get("w_bm25", self.weights.w_bm25)),
                    w_dense=float(obj.get("w_dense", self.weights.w_dense)),
                    w_lexical=float(obj.get("w_lexical", self.weights.w_lexical)),
                    w_doc_type=float(obj.get("w_doc_type", self.weights.w_doc_type)),
                )
                loaded = True
        except Exception:
            loaded = False
        cross_encoder_dir = self.model_dir / "cross_encoder"
        if cross_encoder_dir.exists():
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(str(cross_encoder_dir))
                loaded = True
            except Exception:
                self._cross_encoder = None
        return loaded

    def rerank(self, items: list[dict[str, Any]], *, preferred_doc_types: list[str] | None = None, query: str | None = None) -> list[dict[str, Any]]:
        pref = set(preferred_doc_types or [])
        w = self.weights
        ce_scores: dict[int, float] = {}
        if query and self._cross_encoder is not None and items:
            try:
                pairs = [
                    [
                        query,
                        str(it.get("full_text") or it.get("text") or it.get("chunk_text") or it.get("title") or ""),
                    ]
                    for it in items
                ]
                raw_scores = self._cross_encoder.predict(pairs)
                vals = [float(x) for x in raw_scores]
                lo = min(vals)
                hi = max(vals)
                span = max(1e-9, hi - lo)
                ce_scores = {idx: (val - lo) / span for idx, val in enumerate(vals)}
            except Exception:
                ce_scores = {}

        def _score(idx: int, it: dict[str, Any]) -> float:
            comp = it.get("components") or {}
            doc_type = str(it.get("doc_type") or "")
            dt = 1.0 if (pref and doc_type in pref) else 0.0
            base = (
                w.w_bm25 * float(comp.get("bm25") or 0.0)
                + w.w_dense * float(comp.get("dense") or 0.0)
                + w.w_lexical * float(comp.get("lexical") or 0.0)
                + w.w_doc_type * dt
            )
            if ce_scores:
                return 0.72 * base + 0.28 * ce_scores.get(idx, 0.0)
            return base

        ranked_with_idx = list(enumerate(items))
        ranked_with_idx.sort(key=lambda pair: _score(pair[0], pair[1]), reverse=True)
        return [item for _, item in ranked_with_idx]
