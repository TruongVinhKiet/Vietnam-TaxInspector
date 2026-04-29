"""
tax_agent_embeddings.py – Vietnamese Semantic Embedding Engine (Phase 1)
=========================================================================
Enterprise-grade dense embedding service for the Tax Agent RAG pipeline.

Architecture:
    Tier 1: sentence-transformers (multilingual-e5-small, 118MB) — true semantic
    Tier 2: Cached ONNX Runtime inference (CPU-optimized, i7-8th gen friendly)
    Tier 3: Hash-TFIDF fallback (backward compat, zero-dependency)

Memory budget:
    - Model: ~120MB (e5-small) or ~60MB (ONNX quantized)
    - Embedding cache: LRU bounded by MAX_CACHE_ENTRIES
    - Batch processing: chunked to fit 12GB RAM

Designed for: Core i7-8th gen, 12GB RAM, CPU-only inference.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
EMBEDDING_CACHE_DIR = MODEL_DIR / "embedding_cache"

# ─── Configuration ────────────────────────────────────────────────────────────
# Tuned for i7-8th gen + 12GB RAM
EMBEDDING_MODEL_NAME = os.getenv(
    "TAX_AGENT_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)
EMBEDDING_DIM = int(os.getenv("TAX_AGENT_EMBEDDING_DIM", "384"))
MAX_SEQ_LENGTH = int(os.getenv("TAX_AGENT_MAX_SEQ_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("TAX_AGENT_EMBEDDING_BATCH", "32"))
MAX_CACHE_ENTRIES = int(os.getenv("TAX_AGENT_EMBEDDING_CACHE_SIZE", "50000"))
HASH_TFIDF_DIM = int(os.getenv("TAX_AGENT_HASH_TFIDF_DIM", "96"))

# Query prefix for E5 family models (required by e5 for asymmetric retrieval)
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass
class EmbeddingResult:
    """Single embedding result with metadata."""
    text: str
    vector: np.ndarray
    model_tier: str           # "semantic", "onnx", "hash_tfidf"
    latency_ms: float = 0.0
    dim: int = 384
    cached: bool = False


@dataclass
class BatchEmbeddingResult:
    """Batch embedding results with statistics."""
    embeddings: list[EmbeddingResult]
    total_latency_ms: float = 0.0
    model_tier: str = "hash_tfidf"
    batch_size: int = 0
    throughput_docs_per_sec: float = 0.0


class EmbeddingCache:
    """
    LRU-bounded in-memory cache for embeddings.
    Avoids re-computing embeddings for repeated queries/chunks.
    Thread-safe via dict operations (CPython GIL).
    """

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES):
        self._cache: dict[str, np.ndarray] = {}
        self._access_order: list[str] = []
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_key(text: str, prefix: str, model_tier: str) -> str:
        raw = f"{model_tier}::{prefix}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def get(self, text: str, prefix: str, model_tier: str) -> Optional[np.ndarray]:
        key = self._hash_key(text, prefix, model_tier)
        vec = self._cache.get(key)
        if vec is not None:
            self.hits += 1
            return vec
        self.misses += 1
        return None

    def put(self, text: str, prefix: str, model_tier: str, vector: np.ndarray) -> None:
        key = self._hash_key(text, prefix, model_tier)
        if key in self._cache:
            return
        if len(self._cache) >= self.max_entries:
            # Evict oldest 10%
            evict_count = max(1, self.max_entries // 10)
            for old_key in list(self._cache.keys())[:evict_count]:
                del self._cache[old_key]
        self._cache[key] = vector

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / max(total, 1), 4),
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0


# Global embedding cache (shared across requests within the process)
_embedding_cache = EmbeddingCache()


def _hash_tfidf_embed(text: str, dim: int = HASH_TFIDF_DIM) -> np.ndarray:
    """
    Hash-based TF-IDF embedding (Tier 3 fallback).
    Backward compatible with existing retrieval pipeline.
    """
    import re

    tokens = [t for t in re.split(r"[^a-zA-ZÀ-ỹ0-9_]+", (text or "").lower()) if t]
    vec = np.zeros((dim,), dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


class TaxAgentEmbeddingEngine:
    """
    Multi-tier Vietnamese embedding engine for enterprise RAG.

    Loading priority:
        1. sentence-transformers (true semantic, requires library)
        2. ONNX Runtime (CPU-optimized, smaller footprint)
        3. Hash-TFIDF (zero-dependency fallback)

    Usage:
        engine = TaxAgentEmbeddingEngine()
        engine.load()

        # Single query
        result = engine.embed_query("hoàn thuế VAT điều kiện gì?")

        # Single passage
        result = engine.embed_passage("Điều 13 Luật thuế GTGT...")

        # Batch
        batch = engine.embed_passages_batch(["chunk1", "chunk2", ...])
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        model_dir: Optional[Path] = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        batch_size: int = BATCH_SIZE,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model_dir = model_dir or MODEL_DIR
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

        self._st_model = None      # sentence-transformers model
        self._onnx_session = None  # ONNX runtime session
        self._tokenizer = None     # tokenizer for ONNX path
        self._model_tier: str = "hash_tfidf"
        self._loaded = False
        self._embedding_dim: int = EMBEDDING_DIM

    def load(self) -> str:
        """
        Attempt to load the best available embedding model.
        Returns the tier that was loaded: "semantic", "onnx", or "hash_tfidf".
        """
        # Tier 1: sentence-transformers
        tier = self._try_load_sentence_transformers()
        if tier:
            return tier

        # Tier 2: ONNX Runtime
        tier = self._try_load_onnx()
        if tier:
            return tier

        # Tier 3: Hash-TFIDF fallback
        self._model_tier = "hash_tfidf"
        self._embedding_dim = HASH_TFIDF_DIM
        self._loaded = True
        logger.warning(
            "[EmbeddingEngine] Using hash-TFIDF fallback (dim=%d). "
            "Install sentence-transformers for semantic embeddings: "
            "pip install sentence-transformers",
            HASH_TFIDF_DIM,
        )
        return "hash_tfidf"

    def _try_load_sentence_transformers(self) -> Optional[str]:
        """Try loading sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Check for local cached model first
            local_path = self.model_dir / "embeddings" / self.model_name.replace("/", "_")
            if local_path.exists():
                model_path = str(local_path)
                logger.info("[EmbeddingEngine] Loading from local cache: %s", local_path)
            else:
                model_path = self.model_name
                logger.info("[EmbeddingEngine] Loading from HuggingFace: %s", self.model_name)

            self._st_model = SentenceTransformer(
                model_path,
                device=self.device,
            )
            self._st_model.max_seq_length = self.max_seq_length
            self._embedding_dim = self._st_model.get_embedding_dimension()
            self._model_tier = "semantic"
            self._loaded = True

            # Save locally for future offline use
            if not local_path.exists():
                try:
                    local_path.mkdir(parents=True, exist_ok=True)
                    self._st_model.save(str(local_path))
                    logger.info("[EmbeddingEngine] Model cached to: %s", local_path)
                except Exception as exc:
                    logger.warning("[EmbeddingEngine] Could not cache model: %s", exc)

            logger.info(
                "[EmbeddingEngine] ✓ Loaded sentence-transformers (%s, dim=%d, device=%s)",
                self.model_name, self._embedding_dim, self.device,
            )
            return "semantic"

        except ImportError:
            logger.info("[EmbeddingEngine] sentence-transformers not available")
            return None
        except Exception as exc:
            logger.warning("[EmbeddingEngine] sentence-transformers load failed: %s", exc)
            return None

    def _try_load_onnx(self) -> Optional[str]:
        """Try loading ONNX Runtime model."""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            onnx_path = self.model_dir / "embeddings" / "model.onnx"
            if not onnx_path.exists():
                logger.info("[EmbeddingEngine] No ONNX model found at %s", onnx_path)
                return None

            self._onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            tokenizer_path = self.model_dir / "embeddings" / "tokenizer"
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path) if tokenizer_path.exists() else self.model_name
            )
            self._model_tier = "onnx"
            self._loaded = True

            logger.info("[EmbeddingEngine] ✓ Loaded ONNX model (CPU-optimized)")
            return "onnx"

        except ImportError:
            logger.info("[EmbeddingEngine] ONNX runtime / transformers not available")
            return None
        except Exception as exc:
            logger.warning("[EmbeddingEngine] ONNX load failed: %s", exc)
            return None

    @property
    def model_tier(self) -> str:
        return self._model_tier

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_semantic(self) -> bool:
        return self._model_tier in ("semantic", "onnx")

    def _encode_sentence_transformers(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode using sentence-transformers (Tier 1)."""
        return self._st_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _encode_onnx(self, texts: list[str]) -> np.ndarray:
        """Encode using ONNX Runtime (Tier 2)."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            outputs = self._onnx_session.run(
                None,
                {
                    "input_ids": encoded["input_ids"].astype(np.int64),
                    "attention_mask": encoded["attention_mask"].astype(np.int64),
                },
            )
            # Mean pooling over token embeddings
            token_embeddings = outputs[0]  # (batch, seq_len, dim)
            attention_mask = encoded["attention_mask"][..., np.newaxis]
            pooled = (token_embeddings * attention_mask).sum(axis=1) / attention_mask.sum(axis=1).clip(min=1e-9)
            # L2 normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-9, None)
            all_embeddings.append(pooled.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def _encode_hash_tfidf(self, texts: list[str]) -> np.ndarray:
        """Encode using hash-TFIDF (Tier 3 fallback)."""
        return np.array(
            [_hash_tfidf_embed(t, dim=HASH_TFIDF_DIM) for t in texts],
            dtype=np.float32,
        )

    def _encode_raw(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Route to the appropriate encoder based on loaded tier."""
        if not self._loaded:
            self.load()

        if self._model_tier == "semantic":
            return self._encode_sentence_transformers(texts, show_progress)
        elif self._model_tier == "onnx":
            return self._encode_onnx(texts)
        else:
            return self._encode_hash_tfidf(texts)

    def embed_query(self, query: str) -> EmbeddingResult:
        """
        Embed a single query (with query prefix for asymmetric retrieval).
        Uses cache for repeated queries.
        """
        cached = _embedding_cache.get(query, QUERY_PREFIX, self._model_tier)
        if cached is not None:
            return EmbeddingResult(
                text=query, vector=cached, model_tier=self._model_tier,
                dim=len(cached), cached=True,
            )

        t0 = time.perf_counter()
        if self._model_tier in ("semantic", "onnx"):
            prefixed = f"{QUERY_PREFIX}{query}"
        else:
            prefixed = query

        vec = self._encode_raw([prefixed])[0]
        latency = (time.perf_counter() - t0) * 1000.0

        _embedding_cache.put(query, QUERY_PREFIX, self._model_tier, vec)

        return EmbeddingResult(
            text=query, vector=vec, model_tier=self._model_tier,
            latency_ms=latency, dim=len(vec),
        )

    def embed_passage(self, passage: str) -> EmbeddingResult:
        """
        Embed a single passage (with passage prefix for asymmetric retrieval).
        """
        cached = _embedding_cache.get(passage, PASSAGE_PREFIX, self._model_tier)
        if cached is not None:
            return EmbeddingResult(
                text=passage, vector=cached, model_tier=self._model_tier,
                dim=len(cached), cached=True,
            )

        t0 = time.perf_counter()
        if self._model_tier in ("semantic", "onnx"):
            prefixed = f"{PASSAGE_PREFIX}{passage}"
        else:
            prefixed = passage

        vec = self._encode_raw([prefixed])[0]
        latency = (time.perf_counter() - t0) * 1000.0

        _embedding_cache.put(passage, PASSAGE_PREFIX, self._model_tier, vec)

        return EmbeddingResult(
            text=passage, vector=vec, model_tier=self._model_tier,
            latency_ms=latency, dim=len(vec),
        )

    def embed_passages_batch(
        self,
        passages: list[str],
        show_progress: bool = False,
    ) -> BatchEmbeddingResult:
        """
        Embed a batch of passages. Memory-efficient: processes in chunks.
        Skips cached entries.
        """
        t0 = time.perf_counter()
        results: list[EmbeddingResult] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        for i, passage in enumerate(passages):
            cached = _embedding_cache.get(passage, PASSAGE_PREFIX, self._model_tier)
            if cached is not None:
                results.append(EmbeddingResult(
                    text=passage, vector=cached, model_tier=self._model_tier,
                    dim=len(cached), cached=True,
                ))
            else:
                results.append(None)  # placeholder
                uncached_indices.append(i)
                if self._model_tier in ("semantic", "onnx"):
                    uncached_texts.append(f"{PASSAGE_PREFIX}{passage}")
                else:
                    uncached_texts.append(passage)

        # Encode uncached in batches
        if uncached_texts:
            vecs = self._encode_raw(uncached_texts, show_progress)
            for j, idx in enumerate(uncached_indices):
                vec = vecs[j]
                passage = passages[idx]
                _embedding_cache.put(passage, PASSAGE_PREFIX, self._model_tier, vec)
                results[idx] = EmbeddingResult(
                    text=passage, vector=vec, model_tier=self._model_tier,
                    dim=len(vec),
                )

        total_latency = (time.perf_counter() - t0) * 1000.0
        throughput = len(passages) / max(total_latency / 1000.0, 1e-6)

        return BatchEmbeddingResult(
            embeddings=results,
            total_latency_ms=total_latency,
            model_tier=self._model_tier,
            batch_size=len(passages),
            throughput_docs_per_sec=round(throughput, 1),
        )

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)

    def cosine_similarity_batch(
        self,
        query_vec: np.ndarray,
        passage_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query and multiple passages.
        Vectorized for speed.
        """
        if passage_vecs.ndim == 1:
            passage_vecs = passage_vecs.reshape(1, -1)

        # Handle dimension mismatch (hash_tfidf=96 vs semantic=384)
        q_dim = len(query_vec)
        p_dim = passage_vecs.shape[1]
        if q_dim != p_dim:
            min_dim = min(q_dim, p_dim)
            query_vec = query_vec[:min_dim]
            passage_vecs = passage_vecs[:, :min_dim]

        dots = passage_vecs @ query_vec
        norms_p = np.linalg.norm(passage_vecs, axis=1)
        norm_q = np.linalg.norm(query_vec)
        denom = norms_p * norm_q
        denom = np.clip(denom, 1e-9, None)
        return (dots / denom).astype(np.float32)

    def get_cache_stats(self) -> dict[str, Any]:
        """Return embedding cache statistics."""
        return _embedding_cache.stats()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        _embedding_cache.clear()

    def status(self) -> dict[str, Any]:
        """Return engine status for monitoring."""
        return {
            "model_name": self.model_name,
            "model_tier": self._model_tier,
            "is_semantic": self.is_semantic,
            "embedding_dim": self._embedding_dim,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "loaded": self._loaded,
            "cache": _embedding_cache.stats(),
        }


# ─── Singleton for shared use across the application ──────────────────────────

_engine_instance: Optional[TaxAgentEmbeddingEngine] = None


def get_embedding_engine() -> TaxAgentEmbeddingEngine:
    """Get or create the singleton embedding engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TaxAgentEmbeddingEngine()
        _engine_instance.load()
    return _engine_instance


# ─── Vietnamese Query Expansion (Domain-specific) ─────────────────────────────

VIETNAMESE_TAX_SYNONYMS: dict[str, list[str]] = {
    "hoàn thuế": ["refund thuế", "khấu trừ thuế", "hoàn VAT", "xin hoàn"],
    "hóa đơn": ["invoice", "hóa đơn điện tử", "HĐĐT", "phiếu xuất kho"],
    "nợ đọng": ["nợ thuế", "chậm nộp", "quá hạn nộp", "delinquency"],
    "chuyển giá": ["transfer pricing", "giá giao dịch liên kết", "GDLK"],
    "thanh tra": ["kiểm tra thuế", "audit", "rà soát", "kiểm toán"],
    "sở hữu": ["ownership", "UBO", "chủ sở hữu hưởng lợi", "cổ đông"],
    "offshore": ["công ty nước ngoài", "pháp nhân ngoại", "BVI", "Cayman"],
    "MST": ["mã số thuế", "tax code", "TIN"],
    "NNT": ["người nộp thuế", "taxpayer", "doanh nghiệp"],
    "VAT": ["thuế GTGT", "thuế giá trị gia tăng", "value added tax"],
    "TNCN": ["thuế thu nhập cá nhân", "PIT", "personal income tax"],
    "TNDN": ["thuế thu nhập doanh nghiệp", "CIT", "corporate income tax"],
}

VIETNAMESE_TAX_ACRONYMS: dict[str, str] = {
    "HĐĐT": "hóa đơn điện tử",
    "GTGT": "giá trị gia tăng",
    "TNCN": "thu nhập cá nhân",
    "TNDN": "thu nhập doanh nghiệp",
    "GDLK": "giao dịch liên kết",
    "NNT": "người nộp thuế",
    "MST": "mã số thuế",
    "TCT": "tổng cục thuế",
    "CCT": "chi cục thuế",
    "CT": "cục thuế",
    "KTT": "kiểm tra thuế",
    "UBO": "ultimate beneficial owner",
}


def expand_query(query: str) -> str:
    """
    Expand a Vietnamese tax query with domain synonyms and acronym resolution.
    Returns the expanded query for better retrieval recall.
    """
    expanded_parts = [query]

    query_lower = query.lower()

    # Add synonym expansions
    for key, synonyms in VIETNAMESE_TAX_SYNONYMS.items():
        if key.lower() in query_lower:
            # Add top 2 most different synonyms
            for syn in synonyms[:2]:
                if syn.lower() not in query_lower:
                    expanded_parts.append(syn)

    # Resolve acronyms
    for acronym, full_form in VIETNAMESE_TAX_ACRONYMS.items():
        if acronym in query and full_form.lower() not in query_lower:
            expanded_parts.append(full_form)

    return " ".join(expanded_parts)
