from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

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
                "VALUES (:model_name, :model_version, :artifact_path, :feature_set_id, :train_data_hash, :code_hash, CAST(:metrics_json AS jsonb), CAST(:gates_json AS jsonb), :status) "
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

    def ensure_experiment(
        self,
        *,
        experiment_key: str,
        model_name: str,
        objective: str = "",
        owner: str = "system",
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        row = self.db.execute(
            text("SELECT id FROM ml_experiments WHERE experiment_key = :experiment_key"),
            {"experiment_key": experiment_key},
        ).fetchone()
        if row:
            self.db.execute(
                text(
                    "UPDATE ml_experiments "
                    "SET model_name = :model_name, objective = :objective, owner = :owner, status = :status, "
                    "metadata_json = CAST(:metadata_json AS jsonb), updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = :id"
                ),
                {
                    "id": row[0],
                    "model_name": model_name,
                    "objective": objective or None,
                    "owner": owner or None,
                    "status": status,
                    "metadata_json": json.dumps(metadata or {}, default=str),
                },
            )
            self.db.commit()
            return int(row[0])

        inserted = self.db.execute(
            text(
                "INSERT INTO ml_experiments (experiment_key, model_name, objective, owner, status, metadata_json) "
                "VALUES (:experiment_key, :model_name, :objective, :owner, :status, CAST(:metadata_json AS jsonb)) "
                "RETURNING id"
            ),
            {
                "experiment_key": experiment_key,
                "model_name": model_name,
                "objective": objective or None,
                "owner": owner or None,
                "status": status,
                "metadata_json": json.dumps(metadata or {}, default=str),
            },
        ).fetchone()
        self.db.commit()
        return int(inserted[0])

    def register_dataset_version(
        self,
        *,
        dataset_key: str,
        dataset_version: str,
        entity_type: str = "",
        row_count: Optional[int] = None,
        source_tables: Optional[list[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        data_hash: str = "",
        created_by: str = "system",
    ) -> int:
        row = self.db.execute(
            text(
                "SELECT id FROM dataset_versions WHERE dataset_key = :dataset_key AND dataset_version = :dataset_version"
            ),
            {"dataset_key": dataset_key, "dataset_version": dataset_version},
        ).fetchone()
        if row:
            return int(row[0])
        inserted = self.db.execute(
            text(
                "INSERT INTO dataset_versions "
                "(dataset_key, dataset_version, entity_type, row_count, source_tables_json, filters_json, data_hash, created_by) "
                "VALUES (:dataset_key, :dataset_version, :entity_type, :row_count, CAST(:source_tables_json AS jsonb), "
                "CAST(:filters_json AS jsonb), :data_hash, :created_by) RETURNING id"
            ),
            {
                "dataset_key": dataset_key,
                "dataset_version": dataset_version,
                "entity_type": entity_type or None,
                "row_count": row_count,
                "source_tables_json": json.dumps(source_tables or [], default=str),
                "filters_json": json.dumps(filters or {}, default=str),
                "data_hash": data_hash or None,
                "created_by": created_by or None,
            },
        ).fetchone()
        self.db.commit()
        return int(inserted[0])

    def register_label_version(
        self,
        *,
        label_key: str,
        label_version: str,
        entity_type: str = "",
        label_source: str = "",
        positive_count: Optional[int] = None,
        negative_count: Optional[int] = None,
        label_hash: str = "",
        notes: str = "",
    ) -> int:
        row = self.db.execute(
            text("SELECT id FROM label_versions WHERE label_key = :label_key AND label_version = :label_version"),
            {"label_key": label_key, "label_version": label_version},
        ).fetchone()
        if row:
            return int(row[0])
        inserted = self.db.execute(
            text(
                "INSERT INTO label_versions "
                "(label_key, label_version, entity_type, label_source, positive_count, negative_count, label_hash, notes) "
                "VALUES (:label_key, :label_version, :entity_type, :label_source, :positive_count, :negative_count, :label_hash, :notes) "
                "RETURNING id"
            ),
            {
                "label_key": label_key,
                "label_version": label_version,
                "entity_type": entity_type or None,
                "label_source": label_source or None,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "label_hash": label_hash or None,
                "notes": notes or None,
            },
        ).fetchone()
        self.db.commit()
        return int(inserted[0])

    def start_training_run(
        self,
        *,
        model_name: str,
        experiment_id: Optional[int] = None,
        model_version: str = "",
        dataset_version_id: Optional[int] = None,
        label_version_id: Optional[int] = None,
        feature_set_id: Optional[int] = None,
        seed: Optional[int] = None,
        code_hash: str = "",
        hyperparams: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        resolved_run_id = run_id or f"{model_name}-{uuid4().hex[:12]}"
        self.db.execute(
            text(
                "INSERT INTO ml_training_runs "
                "(run_id, experiment_id, model_name, model_version, dataset_version_id, label_version_id, feature_set_id, "
                "status, seed, code_hash, hyperparams_json) "
                "VALUES (:run_id, :experiment_id, :model_name, :model_version, :dataset_version_id, :label_version_id, "
                ":feature_set_id, 'running', :seed, :code_hash, CAST(:hyperparams_json AS jsonb))"
            ),
            {
                "run_id": resolved_run_id,
                "experiment_id": experiment_id,
                "model_name": model_name,
                "model_version": model_version or None,
                "dataset_version_id": dataset_version_id,
                "label_version_id": label_version_id,
                "feature_set_id": feature_set_id,
                "seed": seed,
                "code_hash": code_hash or None,
                "hyperparams_json": json.dumps(hyperparams or {}, default=str),
            },
        )
        self.db.commit()
        return resolved_run_id

    def complete_training_run(
        self,
        *,
        run_id: str,
        status: str = "completed",
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db.execute(
            text(
                "UPDATE ml_training_runs "
                "SET status = :status, metrics_json = CAST(:metrics_json AS jsonb), "
                "artifacts_json = CAST(:artifacts_json AS jsonb), completed_at = CURRENT_TIMESTAMP "
                "WHERE run_id = :run_id"
            ),
            {
                "run_id": run_id,
                "status": status,
                "metrics_json": json.dumps(metrics or {}, default=str),
                "artifacts_json": json.dumps(artifacts or {}, default=str),
            },
        )
        self.db.commit()

    def register_rollout(
        self,
        *,
        model_name: str,
        model_version: str,
        environment: str = "staging",
        rollout_type: str = "shadow",
        status: str = "planned",
        approved_by: str = "",
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db.execute(
            text(
                "INSERT INTO deployment_rollouts "
                "(model_name, model_version, environment, rollout_type, status, approved_by, rollout_notes, rollout_metadata) "
                "VALUES (:model_name, :model_version, :environment, :rollout_type, :status, :approved_by, :rollout_notes, CAST(:rollout_metadata AS jsonb))"
            ),
            {
                "model_name": model_name,
                "model_version": model_version,
                "environment": environment,
                "rollout_type": rollout_type,
                "status": status,
                "approved_by": approved_by or None,
                "rollout_notes": notes or None,
                "rollout_metadata": json.dumps(metadata or {}, default=str),
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
                ":input_feature_hash, :output_hash, CAST(:outputs_json AS jsonb), :explanation_ref, :latency_ms)"
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

