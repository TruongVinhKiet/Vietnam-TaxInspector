from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import text

from app.database import SessionLocal


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    index_key = "tax_knowledge_hash_tfidf_v1"
    with SessionLocal() as db:
        # Derive a stable corpus hash from (count, max created_at, embedding model).
        agg = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS n,
                    COALESCE(MAX(kce.created_at), NOW()) AS max_ts,
                    COALESCE(MAX(kce.embedding_dim), 0) AS dim,
                    COALESCE(MAX(kce.embedding_model), 'unknown') AS embedding_model
                FROM knowledge_chunk_embeddings kce
                """
            )
        ).mappings().first()
        n = int((agg or {}).get("n") or 0)
        dim = int((agg or {}).get("dim") or 96)
        embedding_model = str((agg or {}).get("embedding_model") or "hash-tfidf-v1")
        max_ts = str((agg or {}).get("max_ts") or "")
        corpus_hash = _sha({"n": n, "max_ts": max_ts, "embedding_model": embedding_model, "dim": dim})

        db.execute(
            text(
                """
                INSERT INTO vector_index_registry
                (index_key, corpus_key, corpus_version, embedding_model, embedding_dim, index_type, build_params, corpus_hash, status)
                VALUES
                (:index_key, 'tax_knowledge', :corpus_version, :embedding_model, :embedding_dim, 'none', CAST(:build_params AS jsonb), :corpus_hash, 'ready')
                ON CONFLICT (index_key) DO UPDATE SET
                    corpus_version = EXCLUDED.corpus_version,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    corpus_hash = EXCLUDED.corpus_hash,
                    status = EXCLUDED.status,
                    build_params = EXCLUDED.build_params
                """
            ),
            {
                "index_key": index_key,
                "corpus_version": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                "embedding_model": embedding_model,
                "embedding_dim": dim,
                "build_params": json.dumps({"note": "registry-only; ANN not enabled yet"}),
                "corpus_hash": corpus_hash,
            },
        )
        db.commit()
        print(f"[OK] vector index registered index_key={index_key} n={n} dim={dim} corpus_hash={corpus_hash[:12]}...")


if __name__ == "__main__":
    main()

