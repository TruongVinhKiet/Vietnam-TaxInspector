from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


def tokenize(text_value: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text_value or "").lower()) if t]


def embed_hash_tfidf(text_value: str, dim: int = 96) -> np.ndarray:
    import hashlib

    vec = np.zeros((dim,), dtype=float)
    toks = tokenize(text_value)
    if not toks:
        return vec
    for tok in toks:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class RetrievalCandidate:
    chunk_id: int
    chunk_key: str
    title: str
    doc_type: str | None
    text: str
    embedding: list[float] | None


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]], *, k1: float = 1.4, b: float = 0.75) -> list[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]
    q = list(dict.fromkeys(query_tokens))
    N = len(docs_tokens)
    df = {t: 0 for t in q}
    doc_lens = [len(d) for d in docs_tokens]
    avgdl = (sum(doc_lens) / max(N, 1)) if N else 0.0
    for d in docs_tokens:
        seen = set(d)
        for t in q:
            if t in seen:
                df[t] += 1
    idf = {t: math.log(1.0 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in q}

    scores: list[float] = []
    for d in docs_tokens:
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        dl = len(d)
        denom_norm = (1.0 - b) + b * (dl / max(avgdl, 1.0))
        s = 0.0
        for t in q:
            f = float(tf.get(t, 0))
            if f <= 0:
                continue
            num = f * (k1 + 1.0)
            denom = f + k1 * denom_norm
            s += idf[t] * (num / max(denom, 1e-9))
        scores.append(float(s))
    return scores


def hybrid_score(
    *,
    query_text: str,
    candidate: RetrievalCandidate,
    bm25: float,
    dense: float,
    lexical: float,
    w_bm25: float = 0.45,
    w_dense: float = 0.40,
    w_lexical: float = 0.15,
) -> tuple[float, dict[str, Any]]:
    score = w_bm25 * bm25 + w_dense * dense + w_lexical * lexical
    return float(score), {"bm25": float(bm25), "dense": float(dense), "lexical": float(lexical)}

