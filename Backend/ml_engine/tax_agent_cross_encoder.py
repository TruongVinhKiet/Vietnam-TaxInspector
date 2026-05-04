"""
tax_agent_cross_encoder.py – Cross-Encoder Reranker (Phase 1)
==============================================================
Production-grade reranker for the Tax Agent RAG pipeline.

Architecture:
    Tier 1: Cross-Encoder (multilingual MiniLM) – true semantic relevance
    Tier 2: Trained LightGBM reranker on multi-signal features
    Tier 3: Weighted linear combination (backward compat fallback)

Designed for: Core i7-8th gen, 12GB RAM, CPU-only.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, multilingual)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# ─── Configuration ────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL = os.getenv(
    "TAX_AGENT_CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_BATCH_SIZE = int(os.getenv("TAX_AGENT_RERANK_BATCH", "16"))
RERANK_TOP_N = int(os.getenv("TAX_AGENT_RERANK_TOP_N", "20"))


@dataclass
class RerankCandidate:
    """A retrieval candidate to be reranked."""
    chunk_id: int
    chunk_key: str
    title: str
    doc_type: str | None
    text: str
    # Original retrieval scores (for feature-based fallback)
    bm25_score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    original_rank: int = 0
    # Reranker output
    rerank_score: float = 0.0
    rerank_tier: str = "unranked"


@dataclass
class RerankResult:
    """Result of reranking."""
    candidates: list[RerankCandidate]
    model_tier: str
    latency_ms: float = 0.0
    top_score: float = 0.0
    score_gap: float = 0.0  # gap between #1 and #2 (confidence proxy)


class TaxAgentCrossEncoder:
    """
    Multi-tier reranker for enterprise RAG.

    Tier 1: Cross-Encoder neural reranker
        - Input: (query, passage) pair → relevance score
        - Model: ms-marco-MiniLM-L-6-v2 (multilingual, ~80MB)

    Tier 2: LightGBM learned reranker
        - Features: BM25 + dense + lexical + doc_type + query-passage overlap
        - Trained on historical click/relevance data

    Tier 3: Weighted linear combination (fallback)
        - Static weights on BM25, dense, lexical scores

    Usage:
        reranker = TaxAgentCrossEncoder()
        reranker.load()
        result = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        model_dir: Optional[Path] = None,
        batch_size: int = RERANK_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.model_dir = model_dir or MODEL_DIR
        self.batch_size = batch_size

        self._cross_encoder = None
        self._lgbm_model = None
        self._model_tier: str = "weighted"
        self._loaded = False

    def load(self) -> str:
        """Load the best available reranker model."""
        tier = self._try_load_cross_encoder()
        if tier:
            return tier

        tier = self._try_load_lgbm()
        if tier:
            return tier

        self._model_tier = "weighted"
        self._loaded = True
        logger.warning(
            "[CrossEncoder] Using weighted fallback. "
            "Install sentence-transformers for neural reranking: "
            "pip install sentence-transformers"
        )
        return "weighted"

    def _try_load_cross_encoder(self) -> Optional[str]:
        """Try loading cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            local_path = self.model_dir / "reranker" / self.model_name.replace("/", "_")
            if local_path.exists():
                model_path = str(local_path)
                logger.info("[CrossEncoder] Loading from local cache: %s", local_path)
            else:
                model_path = self.model_name
                logger.info("[CrossEncoder] Loading from HuggingFace: %s", self.model_name)

            self._cross_encoder = CrossEncoder(
                model_path,
                max_length=256,
                device="cpu",
            )
            self._model_tier = "cross_encoder"
            self._loaded = True

            # Cache locally for offline use
            if not local_path.exists():
                try:
                    local_path.mkdir(parents=True, exist_ok=True)
                    self._cross_encoder.save(str(local_path))
                    logger.info("[CrossEncoder] Model cached to: %s", local_path)
                except Exception as exc:
                    logger.warning("[CrossEncoder] Could not cache model: %s", exc)

            logger.info(
                "[CrossEncoder] ✓ Loaded cross-encoder (%s)", self.model_name
            )
            return "cross_encoder"

        except ImportError:
            logger.info("[CrossEncoder] sentence-transformers not available")
            return None
        except Exception as exc:
            logger.warning("[CrossEncoder] Cross-encoder load failed: %s", exc)
            return None

    def _try_load_lgbm(self) -> Optional[str]:
        """Try loading trained LightGBM reranker."""
        try:
            import joblib

            lgbm_path = self.model_dir / "tax_agent_reranker_lgbm.joblib"
            if not lgbm_path.exists():
                return None

            self._lgbm_model = joblib.load(lgbm_path)
            self._model_tier = "lgbm"
            self._loaded = True
            logger.info("[CrossEncoder] ✓ Loaded LightGBM reranker")
            return "lgbm"

        except Exception as exc:
            logger.warning("[CrossEncoder] LightGBM reranker load failed: %s", exc)
            return None

    @property
    def model_tier(self) -> str:
        return self._model_tier

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
        preferred_doc_types: list[str] | None = None,
    ) -> RerankResult:
        """
        Rerank candidates using the best available model.

        Args:
            query: The user query
            candidates: List of retrieval candidates to rerank
            top_k: Number of top results to return
            preferred_doc_types: Preferred document types (boost factor)

        Returns:
            RerankResult with sorted candidates
        """
        if not self._loaded:
            self.load()

        if not candidates:
            return RerankResult(candidates=[], model_tier=self._model_tier)

        t0 = time.perf_counter()

        if self._model_tier == "cross_encoder":
            scored = self._rerank_cross_encoder(query, candidates, preferred_doc_types)
        elif self._model_tier == "lgbm":
            scored = self._rerank_lgbm(query, candidates, preferred_doc_types)
        else:
            scored = self._rerank_weighted(query, candidates, preferred_doc_types)

        # Sort by rerank score descending
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        top_results = scored[:top_k]

        latency = (time.perf_counter() - t0) * 1000.0

        # Compute score gap (confidence proxy)
        top_score = top_results[0].rerank_score if top_results else 0.0
        second_score = top_results[1].rerank_score if len(top_results) > 1 else 0.0
        score_gap = top_score - second_score

        return RerankResult(
            candidates=top_results,
            model_tier=self._model_tier,
            latency_ms=latency,
            top_score=top_score,
            score_gap=score_gap,
        )

    def _rerank_cross_encoder(
        self,
        query: str,
        candidates: list[RerankCandidate],
        preferred_doc_types: list[str] | None,
    ) -> list[RerankCandidate]:
        """Rerank using cross-encoder (Tier 1)."""
        pairs = [(query, c.text[:512]) for c in candidates]

        # Score in batches for memory efficiency
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores = self._cross_encoder.predict(batch)
            all_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else list(scores))

        pref = set(preferred_doc_types or [])

        for i, c in enumerate(candidates):
            # Apply sigmoid to raw logits from cross-encoder to get absolute probability
            raw_score = all_scores[i]
            base_score = 1.0 / (1.0 + np.exp(-raw_score))
            
            # Small bonus for preferred doc types
            doc_bonus = 0.02 if (pref and c.doc_type in pref) else 0.0
            c.rerank_score = float(base_score + doc_bonus)
            c.rerank_tier = "cross_encoder"

        return candidates

    def _rerank_lgbm(
        self,
        query: str,
        candidates: list[RerankCandidate],
        preferred_doc_types: list[str] | None,
    ) -> list[RerankCandidate]:
        """Rerank using trained LightGBM (Tier 2)."""
        import re

        pref = set(preferred_doc_types or [])
        query_tokens = set(
            t for t in re.split(r"[^a-zA-ZÀ-ỹ0-9_]+", query.lower()) if t
        )

        features_list = []
        for c in candidates:
            doc_tokens = set(
                t for t in re.split(r"[^a-zA-ZÀ-ỹ0-9_]+", c.text.lower()) if t
            )
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)

            features = [
                c.bm25_score,
                c.dense_score,
                c.lexical_score,
                overlap,
                1.0 if (pref and c.doc_type in pref) else 0.0,
                len(c.text) / 1000.0,  # text length feature
                c.original_rank / max(len(candidates), 1),  # rank position
            ]
            features_list.append(features)

        X = np.array(features_list, dtype=np.float32)

        if hasattr(self._lgbm_model, "predict_proba"):
            scores = self._lgbm_model.predict_proba(X)[:, 1]
        else:
            scores = self._lgbm_model.predict(X)

        for i, c in enumerate(candidates):
            c.rerank_score = float(scores[i])
            c.rerank_tier = "lgbm"

        return candidates

    def _rerank_weighted(
        self,
        query: str,
        candidates: list[RerankCandidate],
        preferred_doc_types: list[str] | None,
    ) -> list[RerankCandidate]:
        """Rerank using weighted combination (Tier 3 fallback)."""
        import re

        pref = set(preferred_doc_types or [])
        query_tokens = set(
            t for t in re.split(r"[^a-zA-ZÀ-ỹ0-9_]+", query.lower()) if t
        )

        # Enhanced weights (better than original static weights)
        W_BM25 = 0.35
        W_DENSE = 0.40
        W_LEXICAL = 0.10
        W_DOC_TYPE = 0.05
        W_OVERLAP = 0.10

        for c in candidates:
            doc_tokens = set(
                t for t in re.split(r"[^a-zA-ZÀ-ỹ0-9_]+", c.text.lower()) if t
            )
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            doc_type_bonus = 1.0 if (pref and c.doc_type in pref) else 0.0

            score = (
                W_BM25 * c.bm25_score
                + W_DENSE * c.dense_score
                + W_LEXICAL * c.lexical_score
                + W_DOC_TYPE * doc_type_bonus
                + W_OVERLAP * overlap
            )
            c.rerank_score = float(score)
            c.rerank_tier = "weighted"

        return candidates

    def status(self) -> dict[str, Any]:
        """Return reranker status."""
        return {
            "model_name": self.model_name,
            "model_tier": self._model_tier,
            "loaded": self._loaded,
            "batch_size": self.batch_size,
        }


_default_cross_encoder: TaxAgentCrossEncoder | None = None


def get_cross_encoder() -> TaxAgentCrossEncoder:
    """Return a process-wide reranker singleton."""
    global _default_cross_encoder
    if _default_cross_encoder is None:
        _default_cross_encoder = TaxAgentCrossEncoder()
        _default_cross_encoder.load()
    return _default_cross_encoder
