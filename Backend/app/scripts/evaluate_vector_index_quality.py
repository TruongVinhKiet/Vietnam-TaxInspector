from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import text

from app.database import SessionLocal


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def main() -> None:
    index_key = "tax_knowledge_hash_tfidf_v1"
    run_key = f"{index_key}-quality-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    benchmark_key = "tax_agent_core_v1"
    window_days = 14

    with SessionLocal() as db:
        idx = db.execute(text("SELECT id, corpus_hash FROM vector_index_registry WHERE index_key = :k"), {"k": index_key}).fetchone()
        if not idx:
            raise RuntimeError(f"vector index not registered: {index_key}")
        index_id = int(idx[0])
        corpus_hash = str(idx[1] or "")

        since = datetime.utcnow() - timedelta(days=window_days)
        rows = db.execute(
            text(
                """
                SELECT
                    latency_ms,
                    jsonb_array_length(COALESCE(retrieved_chunks, '[]'::jsonb)) AS hits,
                    retrieval_scores
                FROM retrieval_logs
                WHERE created_at >= :since
                ORDER BY created_at DESC
                LIMIT 3000
                """
            ),
            {"since": since},
        ).mappings().all()

        if not rows:
            metrics = {"sample_size": 0, "window_days": window_days, "corpus_hash": corpus_hash}
        else:
            lat = [float(r.get("latency_ms") or 0.0) for r in rows]
            hits = [int(r.get("hits") or 0) for r in rows]
            top_scores = []
            for r in rows:
                scores = r.get("retrieval_scores")
                if isinstance(scores, list) and scores:
                    try:
                        top_scores.append(float(scores[0]))
                    except Exception:
                        pass
            metrics = {
                "sample_size": int(len(rows)),
                "window_days": window_days,
                "retrieval_hit_rate_ge2": round(float(np.mean(np.asarray(hits) >= 2)), 4),
                "retrieval_latency_p95_ms": round(_p95(lat), 3),
                "top1_score_mean": round(float(np.mean(top_scores)) if top_scores else 0.0, 6),
                "top1_score_p10": round(float(np.percentile(np.asarray(top_scores, dtype=float), 10)) if top_scores else 0.0, 6),
                "corpus_hash": corpus_hash,
            }

        db.execute(
            text(
                """
                INSERT INTO vector_index_quality_runs (index_id, run_key, benchmark_key, metrics_json)
                VALUES (:index_id, :run_key, :benchmark_key, CAST(:metrics_json AS jsonb))
                """
            ),
            {"index_id": index_id, "run_key": run_key, "benchmark_key": benchmark_key, "metrics_json": json.dumps(metrics, ensure_ascii=True)},
        )
        db.commit()
        print(f"[OK] vector index quality logged run_key={run_key}")


if __name__ == "__main__":
    main()

