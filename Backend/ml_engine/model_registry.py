from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class AuditContext:
    request_id: str
    actor_user_id: Optional[int] = None
    actor_badge_id: Optional[str] = None


class ModelRegistryService:
    """
    Minimal model registry + inference audit trail.
    - Registry stores governance metadata per model version.
    - Audit logs store per-inference trace for compliance.
    """

    def __init__(self, db: Session):
        self.db = db

    def resolve_active_version(self, model_name: str) -> Optional[str]:
        row = self.db.execute(
            text(
                "SELECT model_version FROM model_registry "
                "WHERE model_name = :model_name AND status = 'prod' "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"model_name": model_name},
        ).fetchone()
        return str(row[0]) if row else None

    def upsert_registry_entry(
        self,
        *,
        model_name: str,
        model_version: str,
        artifact_path: str = "",
        feature_set_id: Optional[int] = None,
        train_data_hash: str = "",
        code_hash: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        gates: Optional[Dict[str, Any]] = None,
        status: str = "staging",
    ) -> None:
        metrics_json = json.dumps(metrics or {}, default=str)
        gates_json = json.dumps(gates or {}, default=str)
        self.db.execute(
            text(
                "INSERT INTO model_registry "
                "(model_name, model_version, artifact_path, feature_set_id, train_data_hash, code_hash, metrics_json, gates_json, status) "
                "VALUES (:model_name, :model_version, :artifact_path, :feature_set_id, :train_data_hash, :code_hash, :metrics_json::jsonb, :gates_json::jsonb, :status) "
                "ON CONFLICT (model_name, model_version) DO UPDATE SET "
                "artifact_path = EXCLUDED.artifact_path, "
                "feature_set_id = COALESCE(EXCLUDED.feature_set_id, model_registry.feature_set_id), "
                "train_data_hash = COALESCE(NULLIF(EXCLUDED.train_data_hash,''), model_registry.train_data_hash), "
                "code_hash = COALESCE(NULLIF(EXCLUDED.code_hash,''), model_registry.code_hash), "
                "metrics_json = EXCLUDED.metrics_json, "
                "gates_json = EXCLUDED.gates_json, "
                "status = EXCLUDED.status"
            ),
            {
                "model_name": model_name,
                "model_version": model_version,
                "artifact_path": artifact_path or None,
                "feature_set_id": feature_set_id,
                "train_data_hash": train_data_hash or None,
                "code_hash": code_hash or None,
                "metrics_json": metrics_json,
                "gates_json": gates_json,
                "status": status,
            },
        )
        self.db.commit()

    def log_inference(
        self,
        *,
        model_name: str,
        model_version: str,
        entity_type: str,
        entity_id: str,
        as_of_date: Optional[str] = None,
        input_features: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        explanation_ref: str = "",
        latency_ms: Optional[float] = None,
        ctx: Optional[AuditContext] = None,
    ) -> None:
        input_hash = _sha256_json(input_features or {})
        output_hash = _sha256_json(outputs or {})
        outputs_json = json.dumps(outputs or {}, default=str)
        self.db.execute(
            text(
                "INSERT INTO inference_audit_logs "
                "(model_name, model_version, request_id, actor_badge_id, actor_user_id, entity_type, entity_id, as_of_date, "
                "input_feature_hash, output_hash, outputs_json, explanation_ref, latency_ms) "
                "VALUES "
                "(:model_name, :model_version, :request_id, :actor_badge_id, :actor_user_id, :entity_type, :entity_id, :as_of_date, "
                ":input_feature_hash, :output_hash, :outputs_json::jsonb, :explanation_ref, :latency_ms)"
            ),
            {
                "model_name": model_name,
                "model_version": model_version or "unknown",
                "request_id": ctx.request_id if ctx else None,
                "actor_badge_id": (ctx.actor_badge_id if ctx else None),
                "actor_user_id": (ctx.actor_user_id if ctx else None),
                "entity_type": entity_type,
                "entity_id": entity_id,
                "as_of_date": as_of_date,
                "input_feature_hash": input_hash,
                "output_hash": output_hash,
                "outputs_json": outputs_json,
                "explanation_ref": explanation_ref or None,
                "latency_ms": latency_ms,
            },
        )
        self.db.commit()


class InferenceTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

