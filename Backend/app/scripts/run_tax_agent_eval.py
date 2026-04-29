from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import InternalError, ProgrammingError

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


@dataclass
class EvalCase:
    case_id: str
    message: str
    expected_intent: str | None = None
    min_hits: int = 2


def _tokenize(text_value: str) -> list[str]:
    import re

    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", (text_value or "").lower()) if tok]


def _embed(text_value: str, dim: int = 96) -> np.ndarray:
    vec = np.zeros((dim,), dtype=float)
    toks = _tokenize(text_value)
    if not toks:
        return vec
    for tok in toks:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


INTENT_RULES = {
    "vat_refund_risk": ["hoan thue", "vat", "ho so hoan", "refund", "đề nghị hoàn"],
    "invoice_risk": ["hoa don", "invoice", "xuat hoa don", "mua vao", "ban ra"],
    "delinquency": ["no dong", "cham nop", "delinquency", "qua han", "thu no"],
    "osint_ownership": ["offshore", "so huu", "ubo", "phoenix", "cong ty me"],
    "transfer_pricing": ["chuyen gia", "transfer pricing", "gia giao dich lien ket", "mispricing"],
    "audit_selection": ["thanh tra", "audit", "kiem tra", "xep hang ho so"],
}


def _ensure_agent_eval_schema(db) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS agent_eval_suites (
            id SERIAL PRIMARY KEY,
            suite_key VARCHAR(120) NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_eval_runs (
            id SERIAL PRIMARY KEY,
            suite_id INTEGER NOT NULL REFERENCES agent_eval_suites(id) ON DELETE CASCADE,
            run_key VARCHAR(120) NOT NULL UNIQUE,
            model_version VARCHAR(80),
            metrics_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS retrieval_logs (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(120) NOT NULL,
            session_id VARCHAR(120),
            query_text TEXT,
            query_hash VARCHAR(64),
            intent VARCHAR(80),
            entity_scope JSONB,
            retrieved_chunks JSONB,
            retrieval_scores JSONB,
            top_k INTEGER,
            latency_ms FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_agent_eval_runs_suite_created ON agent_eval_runs (suite_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_retrieval_logs_created ON retrieval_logs (created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_retrieval_logs_intent_created ON retrieval_logs (intent, created_at DESC)",
    ]
    for stmt in statements:
        db.execute(text(stmt))


def _ensure_knowledge_schema(db) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id SERIAL PRIMARY KEY,
            document_key VARCHAR(120) NOT NULL UNIQUE,
            title VARCHAR(400) NOT NULL,
            doc_type VARCHAR(80) NOT NULL,
            authority VARCHAR(200),
            language_code VARCHAR(10) NOT NULL DEFAULT 'vi',
            effective_from DATE,
            effective_to DATE,
            status VARCHAR(30) NOT NULL DEFAULT 'active',
            source_uri VARCHAR(500),
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_document_versions (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
            version_tag VARCHAR(80) NOT NULL,
            content_hash VARCHAR(64),
            raw_text TEXT,
            parsed_json JSONB,
            ingestion_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_doc_versions_doc_tag ON knowledge_document_versions (document_id, version_tag)",
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id SERIAL PRIMARY KEY,
            version_id INTEGER NOT NULL REFERENCES knowledge_document_versions(id) ON DELETE CASCADE,
            chunk_key VARCHAR(120) NOT NULL UNIQUE,
            chunk_index INTEGER NOT NULL,
            heading VARCHAR(300),
            chunk_text TEXT NOT NULL,
            token_count INTEGER,
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunk_embeddings (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER NOT NULL UNIQUE REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
            embedding_model VARCHAR(80) NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding_json JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    for stmt in statements:
        db.execute(text(stmt))


def _infer_intent(message: str) -> tuple[str, float]:
    # Keep eval consistent with production router: learned-first intent if available.
    try:
        from pathlib import Path

        from ml_engine.tax_agent_intent_model import TaxAgentIntentModel

        model_dir = Path(__file__).resolve().parents[2] / "data" / "models"
        model = TaxAgentIntentModel(model_dir)
        if model.load():
            intent, conf, _ = model.predict(message)
            if intent and conf >= 0.45:
                return intent, min(0.95, max(0.15, float(conf)))
    except Exception:
        pass

    normalized = message.lower()
    best = ("general_tax_query", 0.15)
    for intent, keywords in INTENT_RULES.items():
        score = sum(1 for kw in keywords if kw in normalized)
        if score > best[1]:
            best = (intent, float(score))
    conf = min(0.95, 0.25 + 0.15 * best[1])
    if best[0] == "general_tax_query":
        conf = 0.22
    return best[0], conf


def _run_retrieval(db, *, session_id: str, query_text: str, intent: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    query_vec = _embed(query_text)
    try:
        rows = db.execute(
            text(
                """
                SELECT
                    kc.id AS chunk_id,
                    kc.chunk_key,
                    kc.chunk_text,
                    kd.title,
                    kd.doc_type,
                    kce.embedding_json
                FROM knowledge_chunks kc
                JOIN knowledge_document_versions kdv ON kdv.id = kc.version_id
                JOIN knowledge_documents kd ON kd.id = kdv.document_id
                LEFT JOIN knowledge_chunk_embeddings kce ON kce.chunk_id = kc.id
                WHERE kd.status = 'active'
                ORDER BY kc.created_at DESC
                LIMIT 400
                """
            )
        ).mappings().all()
    except (ProgrammingError, InternalError):
        # Knowledge schema may not be present in a fresh DB; still allow intent-only evaluation.
        try:
            db.rollback()
        except Exception:
            pass
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ([], latency_ms)
    scored: list[tuple[float, dict[str, Any]]] = []
    query_tokens = set(_tokenize(query_text))
    for row in rows:
        emb = row.get("embedding_json")
        if not isinstance(emb, list) or not emb:
            continue
        vec = np.asarray(emb, dtype=float)
        dot = float(np.dot(query_vec[: len(vec)], vec[: len(query_vec)]))
        text_tokens = set(_tokenize(str(row.get("chunk_text") or "")))
        lexical = len(query_tokens.intersection(text_tokens)) / max(1, len(query_tokens))
        score = 0.65 * dot + 0.35 * lexical
        scored.append(
            (
                score,
                {
                    "chunk_id": int(row["chunk_id"]),
                    "chunk_key": row["chunk_key"],
                    "title": row["title"],
                    "doc_type": row["doc_type"],
                    "text": str(row["chunk_text"])[:900],
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:top_k]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    db.execute(
        text(
            """
            INSERT INTO retrieval_logs
            (request_id, session_id, query_text, query_hash, intent, entity_scope, retrieved_chunks, retrieval_scores, top_k, latency_ms)
            VALUES
            (:request_id, :session_id, :query_text, :query_hash, :intent, CAST(:entity_scope AS jsonb), CAST(:retrieved_chunks AS jsonb), CAST(:retrieval_scores AS jsonb), :top_k, :latency_ms)
            """
        ),
        {
            "request_id": f"ret-eval-{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "query_text": query_text,
            "query_hash": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            "intent": intent,
            "entity_scope": json.dumps({"eval": True}),
            "retrieved_chunks": json.dumps([item[1]["chunk_key"] for item in selected]),
            "retrieval_scores": json.dumps([round(item[0], 6) for item in selected]),
            "top_k": top_k,
            "latency_ms": latency_ms,
        },
    )
    return ([{"score": float(score), **payload} for score, payload in selected], latency_ms)


def _apply_policy(*, intent: str, intent_conf: float, retrieval_hits: int) -> tuple[bool, bool]:
    abstain = False
    escalate = False
    rules = [
        intent_conf >= 0.35,
        retrieval_hits >= 2,
        (intent not in {"general_tax_query", "transfer_pricing"}) or retrieval_hits >= 3,
    ]
    if not all(rules):
        abstain = True
        if intent in {"audit_selection", "transfer_pricing", "vat_refund_risk"}:
            escalate = True
    return abstain, escalate


DEFAULT_SUITE_CASES: list[EvalCase] = [
    EvalCase(case_id="vat_refund_1", message="Đánh giá rủi ro hồ sơ hoàn thuế VAT kỳ 2025-Q4 cần lưu ý gì?", expected_intent="vat_refund_risk"),
    EvalCase(case_id="invoice_1", message="Hóa đơn mua vào có dấu hiệu trùng lặp thì hệ thống kiểm tra thế nào?", expected_intent="invoice_risk"),
    EvalCase(case_id="delinq_1", message="Doanh nghiệp chậm nộp thuế nhiều kỳ thì dự báo nợ đọng ra sao?", expected_intent="delinquency"),
    EvalCase(case_id="osint_1", message="Phát hiện sở hữu offshore/UBO qua proxy company như thế nào?", expected_intent="osint_ownership"),
    EvalCase(case_id="tp_1", message="Dấu hiệu chuyển giá khi giao dịch liên kết có thể nhận diện thế nào?", expected_intent="transfer_pricing"),
    EvalCase(case_id="audit_1", message="Tiêu chí xếp hạng hồ sơ để thanh tra kiểm tra là gì?", expected_intent="audit_selection"),
    EvalCase(case_id="general_1", message="Giải thích nghĩa vụ kê khai VAT theo quý cho doanh nghiệp mới thành lập.", expected_intent="general_tax_query", min_hits=1),
]


def _ensure_suite(db, *, suite_key: str, description: str) -> int:
    row = db.execute(text("SELECT id FROM agent_eval_suites WHERE suite_key = :k"), {"k": suite_key}).fetchone()
    if row:
        return int(row[0])
    row = db.execute(
        text("INSERT INTO agent_eval_suites (suite_key, description) VALUES (:k, :d) RETURNING id"),
        {"k": suite_key, "d": description},
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to create agent_eval_suite")
    return int(row[0])


def _compute_offline_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    intent_total = len(records)
    intent_correct = sum(1 for r in records if r.get("expected_intent") and r.get("intent") == r.get("expected_intent"))
    abstained = sum(1 for r in records if r.get("abstained"))
    escalated = sum(1 for r in records if r.get("escalation_required"))
    hit_ge_min = sum(1 for r in records if int(r.get("hits") or 0) >= int(r.get("min_hits") or 2))
    cited = sum(1 for r in records if int(r.get("hits") or 0) >= 1)
    latencies = [float(r.get("retrieval_latency_ms") or 0.0) for r in records if r.get("retrieval_latency_ms") is not None]
    p95 = float(np.percentile(np.asarray(latencies, dtype=float), 95)) if latencies else 0.0
    return {
        "offline.intent_accuracy": round(intent_correct / max(intent_total, 1), 4),
        "offline.abstain_rate": round(abstained / max(intent_total, 1), 4),
        "offline.escalation_rate": round(escalated / max(intent_total, 1), 4),
        "offline.retrieval_hit_rate": round(hit_ge_min / max(intent_total, 1), 4),
        "offline.citation_rate": round(cited / max(intent_total, 1), 4),
        "offline.retrieval_latency_p95_ms": round(p95, 3),
        "offline.sample_size": intent_total,
    }


def _compute_online_metrics(db, *, days: int = 7) -> dict[str, Any]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.execute(
        text(
            """
            SELECT
                rl.intent,
                rl.top_k,
                rl.latency_ms,
                jsonb_array_length(COALESCE(rl.retrieved_chunks, '[]'::jsonb)) AS hits
            FROM retrieval_logs rl
            WHERE rl.created_at >= :since
            ORDER BY rl.created_at DESC
            LIMIT 2000
            """
        ),
        {"since": since},
    ).mappings().all()
    if not rows:
        return {"online.sample_size": 0}
    lat = np.asarray([float(r.get("latency_ms") or 0.0) for r in rows], dtype=float)
    hits = np.asarray([int(r.get("hits") or 0) for r in rows], dtype=int)
    return {
        "online.sample_size": int(len(rows)),
        "online.retrieval_hit_rate_ge2": round(float(np.mean(hits >= 2)), 4),
        "online.retrieval_latency_p95_ms": round(float(np.percentile(lat, 95)), 3),
    }


def main() -> None:
    suite_key = "tax_agent_core_v1"
    model_version = "tax-agent-v1"
    run_key = f"{suite_key}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    cases_path = Path(__file__).resolve().parent / "tax_agent_eval_cases.jsonl"

    cases: list[EvalCase] = []
    if cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    case_id=str(obj.get("case_id")),
                    message=str(obj.get("message")),
                    expected_intent=obj.get("expected_intent"),
                    min_hits=int(obj.get("min_hits") or 2),
                )
            )
    if not cases:
        cases = DEFAULT_SUITE_CASES

    with SessionLocal() as db:
        _ensure_agent_eval_schema(db)
        _ensure_knowledge_schema(db)
        # DDL must be committed before any later rollback (e.g., missing knowledge schema).
        db.commit()
        suite_id = _ensure_suite(db, suite_key=suite_key, description="Core tax-agent eval: intent/rag/policy baseline.")
        # Suite row must be committed before any downstream rollback.
        db.commit()

        # Offline-style: run current tax-agent logic against curated queries (no external calls).
        session_id = f"sess-eval-{uuid.uuid4().hex[:8]}"
        records: list[dict[str, Any]] = []
        for c in cases:
            intent, intent_conf = _infer_intent(c.message)
            ctx, latency_ms = _run_retrieval(db, session_id=session_id, query_text=c.message, intent=intent, top_k=5)
            abstain, escalate = _apply_policy(intent=intent, intent_conf=intent_conf, retrieval_hits=len(ctx))
            records.append(
                {
                    "case_id": c.case_id,
                    "expected_intent": c.expected_intent,
                    "intent": intent,
                    "intent_confidence": round(float(intent_conf), 4),
                    "hits": len(ctx),
                    "min_hits": c.min_hits,
                    "abstained": bool(abstain),
                    "escalation_required": bool(escalate),
                    "retrieval_latency_ms": round(float(latency_ms), 3),
                    "top_chunk_key": ctx[0]["chunk_key"] if ctx else None,
                }
            )

        offline_metrics = _compute_offline_metrics(records)
        online_metrics = _compute_online_metrics(db, days=7)
        metrics = {
            **offline_metrics,
            **online_metrics,
            "suite_key": suite_key,
            "run_key": run_key,
            "model_version": model_version,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        db.execute(
            text(
                """
                INSERT INTO agent_eval_runs (suite_id, run_key, model_version, metrics_json)
                VALUES (:suite_id, :run_key, :model_version, CAST(:metrics_json AS jsonb))
                """
            ),
            {
                "suite_id": suite_id,
                "run_key": run_key,
                "model_version": model_version,
                "metrics_json": json.dumps(metrics, ensure_ascii=True),
            },
        )
        db.commit()

        registry = ModelRegistryService(db)
        registry.log_inference(
            model_name="tax_agent_eval",
            model_version=model_version,
            entity_type="eval_suite",
            entity_id=suite_key,
            input_features={"cases": len(cases), "online_window_days": 7},
            outputs={"metrics": metrics},
        )

    out_dir = Path(__file__).resolve().parents[2] / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tax_agent_eval_latest.json"
    out_path.write_text(json.dumps({"records": records, "metrics": metrics}, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[OK] tax-agent eval run_key={run_key} wrote {out_path}")


if __name__ == "__main__":
    main()

