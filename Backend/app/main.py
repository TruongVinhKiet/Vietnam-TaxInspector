"""
main.py – TaxInspector API Entry Point (Security Hardened)
============================================================
Changes:
    1. CORS: allow_credentials=True (required for HttpOnly Cookie)
    2. Security Headers middleware added
    3. Rate Limiter (slowapi) integrated
    4. Swagger /docs disabled in production (optional)
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .database import engine, Base
from .routers import (
    scoring,
    graph,
    delinquency,
    auth,
    ai_analysis,
    monitoring,
    simulation,
    osint,
    invoice_risk,
    vat_refund,
    entity_resolution,
    transfer_pricing,
    audit_selection,
    collections,
    case_triage,
    tax_agent,
    ml_api,
)
from .security import limiter, SecurityHeadersMiddleware, rate_limit_exceeded_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup, cleanup on shutdown."""
    try:
        Base.metadata.create_all(bind=engine)
        print("[OK] Database tables verified / created.")
    except Exception as e:
        print(f"[WARN] Database not reachable, starting without DB: {e}")
    # --- Auto-migration: add columns that may not exist yet ---
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            pgvector_ready = False
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                pgvector_ready = True
            except Exception as vector_exc:
                try:
                    conn.rollback()
                except Exception:
                    pass
                print("[INFO] Môi trường không có pgvector. Hệ thống tự động chuyển sang chế độ Local RAG (BM25 + JSON).")

            # Original migrations
            conn.execute(text(
                "ALTER TABLE ai_risk_assessments "
                "ADD COLUMN IF NOT EXISTS model_confidence FLOAT;"
            ))
            conn.execute(text(
                "ALTER TABLE ai_risk_assessments "
                "ADD COLUMN IF NOT EXISTS yearly_history JSON;"
            ))
            conn.execute(text(
                "ALTER TABLE ai_risk_assessments "
                "ADD COLUMN IF NOT EXISTS model_version VARCHAR(80);"
            ))

            # Flagship model migrations (Phase 0.1)
            conn.execute(text(
                "ALTER TABLE companies "
                "ADD COLUMN IF NOT EXISTS province VARCHAR(100);"
            ))
            conn.execute(text(
                "ALTER TABLE invoices "
                "ADD COLUMN IF NOT EXISTS payment_status VARCHAR(30) DEFAULT 'unknown';"
            ))
            conn.execute(text(
                "ALTER TABLE invoices "
                "ADD COLUMN IF NOT EXISTS goods_category VARCHAR(100);"
            ))
            conn.execute(text(
                "ALTER TABLE invoices "
                "ADD COLUMN IF NOT EXISTS is_adjustment BOOLEAN DEFAULT FALSE;"
            ))
            conn.execute(text(
                "ALTER TABLE tax_returns "
                "ADD COLUMN IF NOT EXISTS due_date DATE;"
            ))
            conn.execute(text(
                "ALTER TABLE tax_returns "
                "ADD COLUMN IF NOT EXISTS tax_type VARCHAR(50) DEFAULT 'VAT';"
            ))
            conn.execute(text(
                "ALTER TABLE tax_returns "
                "ADD COLUMN IF NOT EXISTS amendment_number INTEGER DEFAULT 0;"
            ))

            # KPI loop foundation migrations (Sprint 1)
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS model_version VARCHAR(80);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS label_origin VARCHAR(40) DEFAULT 'manual_inspector';"
            ))
            conn.execute(text(
                "UPDATE inspector_labels "
                "SET label_origin = 'manual_inspector' "
                "WHERE label_origin IS NULL OR btrim(label_origin) = '';"
            ))
            conn.execute(text(
                "UPDATE inspector_labels l "
                "SET model_version = a.model_version "
                "FROM ai_risk_assessments a "
                "WHERE l.assessment_id = a.id "
                "AND (l.model_version IS NULL OR btrim(l.model_version) = '') "
                "AND a.model_version IS NOT NULL "
                "AND btrim(a.model_version) <> '';"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS intervention_action VARCHAR(50);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS intervention_attempted BOOLEAN DEFAULT FALSE;"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS outcome_status VARCHAR(30);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS predicted_collection_uplift NUMERIC(18, 2);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS expected_recovery NUMERIC(18, 2);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS expected_net_recovery NUMERIC(18, 2);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS estimated_audit_cost NUMERIC(18, 2);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS actual_audit_cost NUMERIC(18, 2);"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS actual_audit_hours FLOAT;"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS outcome_recorded_at TIMESTAMP;"
            ))
            conn.execute(text(
                "ALTER TABLE inspector_labels "
                "ADD COLUMN IF NOT EXISTS kpi_window_days INTEGER DEFAULT 90;"
            ))

            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_execution_plans ("
                "id SERIAL PRIMARY KEY, "
                "plan_id VARCHAR(64) NOT NULL UNIQUE, "
                "session_id VARCHAR(64) NOT NULL, "
                "turn_id INT, "
                "query_text TEXT, "
                "intent VARCHAR(64), "
                "complexity VARCHAR(32), "
                "reasoning_trace TEXT, "
                "budget_ms INT, "
                "max_react_iterations INT, "
                "retry_policy_json JSONB, "
                "evidence_contract_json JSONB, "
                "steps_json JSONB, "
                "tool_results_json JSONB, "
                "synthesis_json JSONB, "
                "compliance_json JSONB, "
                "latency_ms FLOAT, "
                "latency_breakdown JSONB, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_agent_exec_plans_session "
                "ON agent_execution_plans (session_id);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_agent_exec_plans_intent "
                "ON agent_execution_plans (intent);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_agent_exec_plans_created "
                "ON agent_execution_plans (created_at DESC);"
            ))
            conn.execute(text("ALTER TABLE agent_execution_plans ADD COLUMN IF NOT EXISTS budget_ms INT;"))
            conn.execute(text("ALTER TABLE agent_execution_plans ADD COLUMN IF NOT EXISTS max_react_iterations INT;"))
            conn.execute(text("ALTER TABLE agent_execution_plans ADD COLUMN IF NOT EXISTS retry_policy_json JSONB;"))
            conn.execute(text("ALTER TABLE agent_execution_plans ADD COLUMN IF NOT EXISTS evidence_contract_json JSONB;"))

            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_quality_metrics ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(64) NOT NULL, "
                "turn_id INT, "
                "intent VARCHAR(64), "
                "retrieval_hits INT, "
                "retrieval_mrr FLOAT, "
                "retrieval_ndcg FLOAT, "
                "synthesis_confidence FLOAT, "
                "evidence_count INT, "
                "citation_count INT, "
                "compliance_decision VARCHAR(16), "
                "warnings_count INT, "
                "total_latency_ms FLOAT, "
                "embedding_tier VARCHAR(32), "
                "reranker_tier VARCHAR(32), "
                "synthesis_tier VARCHAR(32), "
                "tools_invoked INT, "
                "tools_succeeded INT, "
                "tools_failed INT, "
                "user_rating INT, "
                "user_feedback TEXT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))

            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS kpi_trigger_policies ("
                "id SERIAL PRIMARY KEY, "
                "track_name VARCHAR(50) NOT NULL, "
                "metric_name VARCHAR(80) NOT NULL, "
                "comparator VARCHAR(8) NOT NULL DEFAULT '>=', "
                "threshold FLOAT NOT NULL, "
                "min_sample INTEGER NOT NULL DEFAULT 50, "
                "window_days INTEGER NOT NULL DEFAULT 28, "
                "cooldown_days INTEGER NOT NULL DEFAULT 14, "
                "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "rationale TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_kpi_policy_track_metric "
                "ON kpi_trigger_policies (track_name, metric_name);"
            ))

            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS kpi_metric_snapshots ("
                "id SERIAL PRIMARY KEY, "
                "track_name VARCHAR(50) NOT NULL, "
                "metric_name VARCHAR(80) NOT NULL, "
                "metric_value FLOAT, "
                "sample_size INTEGER NOT NULL DEFAULT 0, "
                "comparator VARCHAR(8), "
                "threshold FLOAT, "
                "status VARCHAR(30) NOT NULL DEFAULT 'no_metric', "
                "window_days INTEGER NOT NULL DEFAULT 28, "
                "source VARCHAR(80) NOT NULL DEFAULT 'split_trigger_status', "
                "details JSONB, "
                "generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))

            # Feature store foundation (point-in-time snapshots)
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS feature_sets ("
                "id SERIAL PRIMARY KEY, "
                "name VARCHAR(80) NOT NULL, "
                "version VARCHAR(40) NOT NULL, "
                "owner VARCHAR(80), "
                "description TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_sets_name_version "
                "ON feature_sets (name, version);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS feature_snapshots ("
                "id SERIAL PRIMARY KEY, "
                "entity_type VARCHAR(20) NOT NULL, "
                "entity_id VARCHAR(120) NOT NULL, "
                "as_of_date DATE NOT NULL, "
                "feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE, "
                "features_json JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "source_hash VARCHAR(64), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_feature_snapshots_entity_asof "
                "ON feature_snapshots (entity_type, entity_id, as_of_date DESC);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_feature_snapshots_feature_set_asof "
                "ON feature_snapshots (feature_set_id, as_of_date DESC);"
            ))
            conn.execute(text(
                "DO $$ "
                "BEGIN "
                "IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='feature_snapshots_feature_set_id_fkey') THEN "
                "ALTER TABLE feature_snapshots "
                "ADD CONSTRAINT feature_snapshots_feature_set_id_fkey "
                "FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id) ON DELETE CASCADE; "
                "END IF; "
                "END $$;"
            ))

            # Model registry + inference audit trail
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS model_registry ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80) NOT NULL, "
                "artifact_path VARCHAR(400), "
                "feature_set_id INTEGER REFERENCES feature_sets(id) ON DELETE SET NULL, "
                "train_data_hash VARCHAR(64), "
                "code_hash VARCHAR(64), "
                "metrics_json JSONB, "
                "gates_json JSONB, "
                "status VARCHAR(20) NOT NULL DEFAULT 'staging', "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_name_version "
                "ON model_registry (model_name, model_version);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_model_registry_status "
                "ON model_registry (model_name, status);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_model_registry_created "
                "ON model_registry (created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS inference_audit_logs ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80) NOT NULL, "
                "request_id VARCHAR(64), "
                "actor_badge_id VARCHAR(50), "
                "actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, "
                "entity_type VARCHAR(20) NOT NULL, "
                "entity_id VARCHAR(120) NOT NULL, "
                "as_of_date DATE, "
                "input_feature_hash VARCHAR(64), "
                "output_hash VARCHAR(64), "
                "outputs_json JSONB, "
                "explanation_ref VARCHAR(200), "
                "latency_ms FLOAT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_infer_audit_model_ts "
                "ON inference_audit_logs (model_name, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_infer_audit_entity_ts "
                "ON inference_audit_logs (entity_type, entity_id, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_infer_audit_request "
                "ON inference_audit_logs (request_id);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS model_quality_snapshots ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80), "
                "window_start TIMESTAMP, "
                "window_end TIMESTAMP, "
                "quality_json JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "status VARCHAR(20) DEFAULT 'unknown', "
                "status_reason VARCHAR(120), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_model_quality_snapshots_model_ts "
                "ON model_quality_snapshots (model_name, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS feature_drift_stats ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80), "
                "feature_name VARCHAR(120) NOT NULL, "
                "window_start TIMESTAMP, "
                "window_end TIMESTAMP, "
                "psi FLOAT, "
                "ks FLOAT, "
                "missing_rate FLOAT, "
                "mean FLOAT, "
                "std FLOAT, "
                "baseline_mean FLOAT, "
                "baseline_std FLOAT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_feature_drift_stats_model_feature_ts "
                "ON feature_drift_stats (model_name, feature_name, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS ml_experiments ("
                "id SERIAL PRIMARY KEY, "
                "experiment_key VARCHAR(120) UNIQUE NOT NULL, "
                "model_name VARCHAR(80) NOT NULL, "
                "objective TEXT, "
                "owner VARCHAR(80), "
                "status VARCHAR(20) NOT NULL DEFAULT 'draft', "
                "metadata_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS dataset_versions ("
                "id SERIAL PRIMARY KEY, "
                "dataset_key VARCHAR(120) NOT NULL, "
                "dataset_version VARCHAR(80) NOT NULL, "
                "entity_type VARCHAR(30), "
                "row_count INTEGER, "
                "source_tables_json JSONB, "
                "filters_json JSONB, "
                "data_hash VARCHAR(64), "
                "created_by VARCHAR(80), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_versions_key_version "
                "ON dataset_versions (dataset_key, dataset_version);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS label_versions ("
                "id SERIAL PRIMARY KEY, "
                "label_key VARCHAR(120) NOT NULL, "
                "label_version VARCHAR(80) NOT NULL, "
                "entity_type VARCHAR(30), "
                "label_source VARCHAR(80), "
                "positive_count INTEGER, "
                "negative_count INTEGER, "
                "label_hash VARCHAR(64), "
                "notes TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_label_versions_key_version "
                "ON label_versions (label_key, label_version);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS ml_training_runs ("
                "id SERIAL PRIMARY KEY, "
                "run_id VARCHAR(120) UNIQUE NOT NULL, "
                "experiment_id INTEGER REFERENCES ml_experiments(id) ON DELETE SET NULL, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80), "
                "dataset_version_id INTEGER REFERENCES dataset_versions(id) ON DELETE SET NULL, "
                "label_version_id INTEGER REFERENCES label_versions(id) ON DELETE SET NULL, "
                "feature_set_id INTEGER REFERENCES feature_sets(id) ON DELETE SET NULL, "
                "status VARCHAR(20) NOT NULL DEFAULT 'running', "
                "seed INTEGER, "
                "code_hash VARCHAR(64), "
                "hyperparams_json JSONB, "
                "metrics_json JSONB, "
                "artifacts_json JSONB, "
                "started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "completed_at TIMESTAMP, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS deployment_rollouts ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80) NOT NULL, "
                "environment VARCHAR(30) NOT NULL DEFAULT 'staging', "
                "rollout_type VARCHAR(30) NOT NULL DEFAULT 'shadow', "
                "status VARCHAR(20) NOT NULL DEFAULT 'planned', "
                "approved_by VARCHAR(80), "
                "rollout_notes TEXT, "
                "rollout_metadata JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "ALTER TABLE case_triage_predictions ADD COLUMN IF NOT EXISTS confidence FLOAT;"
            ))
            conn.execute(text(
                "ALTER TABLE case_triage_predictions ADD COLUMN IF NOT EXISTS cohort_tags JSONB;"
            ))
            conn.execute(text(
                "ALTER TABLE audit_selection_predictions ADD COLUMN IF NOT EXISTS fusion_score FLOAT;"
            ))
            conn.execute(text(
                "ALTER TABLE nba_predictions ADD COLUMN IF NOT EXISTS uncertainty_score FLOAT;"
            ))
            conn.execute(text(
                "ALTER TABLE nba_predictions ADD COLUMN IF NOT EXISTS ranked_actions JSONB;"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS entity_risk_fusion_predictions ("
                "id SERIAL PRIMARY KEY, "
                "tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE, "
                "as_of_date DATE NOT NULL, "
                "model_version VARCHAR(80), "
                "fusion_score FLOAT, "
                "risk_band VARCHAR(20), "
                "confidence FLOAT, "
                "component_scores JSONB, "
                "driver_summary JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS evaluation_slices ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80), "
                "slice_name VARCHAR(80) NOT NULL, "
                "slice_value VARCHAR(120) NOT NULL, "
                "metric_name VARCHAR(80) NOT NULL, "
                "metric_value FLOAT, "
                "sample_size INTEGER, "
                "window_start TIMESTAMP, "
                "window_end TIMESTAMP, "
                "details JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS champion_challenger_results ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "champion_version VARCHAR(80), "
                "challenger_version VARCHAR(80), "
                "decision VARCHAR(30), "
                "metric_summary JSONB, "
                "notes TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS calibration_bins ("
                "id SERIAL PRIMARY KEY, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80), "
                "bin_label VARCHAR(40) NOT NULL, "
                "lower_bound FLOAT, "
                "upper_bound FLOAT, "
                "predicted_mean FLOAT, "
                "observed_rate FLOAT, "
                "sample_size INTEGER, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS feature_validation_rules ("
                "id SERIAL PRIMARY KEY, "
                "feature_set_id INTEGER REFERENCES feature_sets(id) ON DELETE CASCADE, "
                "feature_name VARCHAR(120) NOT NULL, "
                "rule_type VARCHAR(40) NOT NULL, "
                "rule_config JSONB, "
                "severity VARCHAR(20) NOT NULL DEFAULT 'warning', "
                "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_nodes ("
                "id SERIAL PRIMARY KEY, "
                "node_id VARCHAR(120) NOT NULL UNIQUE, "
                "node_type VARCHAR(40) NOT NULL, "
                "native_table VARCHAR(80), "
                "native_key VARCHAR(120), "
                "display_name VARCHAR(255), "
                "status VARCHAR(30), "
                "risk_score FLOAT, "
                "country_code VARCHAR(40), "
                "attributes_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_nodes_native ON graph_nodes (native_table, native_key);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_edges ("
                "id SERIAL PRIMARY KEY, "
                "edge_id VARCHAR(120) NOT NULL UNIQUE, "
                "src_node_id VARCHAR(120) NOT NULL, "
                "dst_node_id VARCHAR(120) NOT NULL, "
                "edge_type VARCHAR(40) NOT NULL, "
                "directed BOOLEAN NOT NULL DEFAULT TRUE, "
                "weight FLOAT, "
                "confidence FLOAT, "
                "valid_from TIMESTAMP, "
                "valid_to TIMESTAMP, "
                "observed_at TIMESTAMP, "
                "attributes_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges (src_node_id);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges (dst_node_id);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges (edge_type);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_edge_evidence ("
                "id SERIAL PRIMARY KEY, "
                "edge_id VARCHAR(120) NOT NULL, "
                "source_type VARCHAR(80), "
                "source_ref VARCHAR(160), "
                "source_url VARCHAR(400), "
                "observed_at TIMESTAMP, "
                "ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "extractor_version VARCHAR(80), "
                "confidence FLOAT, "
                "raw_payload_json JSONB"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_edge_evidence_edge ON graph_edge_evidence (edge_id);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_snapshots ("
                "id SERIAL PRIMARY KEY, "
                "snapshot_id VARCHAR(120) NOT NULL UNIQUE, "
                "graph_family VARCHAR(80) NOT NULL, "
                "as_of_timestamp TIMESTAMP NOT NULL, "
                "extraction_policy_json JSONB, "
                "seed_set_json JSONB, "
                "source_tables_json JSONB, "
                "resolution_ruleset_version VARCHAR(80), "
                "lineage_hash VARCHAR(64), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_snapshots_family_time "
                "ON graph_snapshots (graph_family, as_of_timestamp DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_snapshot_nodes ("
                "id SERIAL PRIMARY KEY, "
                "snapshot_id VARCHAR(120) NOT NULL, "
                "node_id VARCHAR(120) NOT NULL, "
                "node_type VARCHAR(40) NOT NULL, "
                "feature_payload JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_snapshot_nodes_snapshot "
                "ON graph_snapshot_nodes (snapshot_id, node_type);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_snapshot_edges ("
                "id SERIAL PRIMARY KEY, "
                "snapshot_id VARCHAR(120) NOT NULL, "
                "edge_id VARCHAR(120) NOT NULL, "
                "edge_type VARCHAR(40) NOT NULL, "
                "src_node_id VARCHAR(120) NOT NULL, "
                "dst_node_id VARCHAR(120) NOT NULL, "
                "attributes_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_snapshot_edges_snapshot "
                "ON graph_snapshot_edges (snapshot_id, edge_type);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_node_versions ("
                "id SERIAL PRIMARY KEY, "
                "node_id VARCHAR(120) NOT NULL, "
                "valid_from TIMESTAMP NOT NULL, "
                "valid_to TIMESTAMP, "
                "attributes_json JSONB, "
                "source_hash VARCHAR(64), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_node_versions_node "
                "ON graph_node_versions (node_id, valid_from DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_labels ("
                "id SERIAL PRIMARY KEY, "
                "entity_type VARCHAR(30) NOT NULL, "
                "entity_id VARCHAR(120) NOT NULL, "
                "label_name VARCHAR(80) NOT NULL, "
                "label_value VARCHAR(40) NOT NULL, "
                "trust_tier VARCHAR(20) NOT NULL, "
                "label_source VARCHAR(80), "
                "annotator VARCHAR(120), "
                "confidence FLOAT, "
                "valid_from TIMESTAMP, "
                "valid_to TIMESTAMP, "
                "evidence_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_labels_entity "
                "ON graph_labels (entity_type, entity_id, label_name);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_graph_labels_tier "
                "ON graph_labels (trust_tier, label_name);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS graph_benchmark_specs ("
                "id SERIAL PRIMARY KEY, "
                "benchmark_key VARCHAR(120) NOT NULL UNIQUE, "
                "graph_family VARCHAR(80) NOT NULL, "
                "baseline_models JSONB, "
                "split_strategy JSONB, "
                "metric_contract JSONB, "
                "slice_contract JSONB, "
                "promotion_gate JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS knowledge_documents ("
                "id SERIAL PRIMARY KEY, "
                "document_key VARCHAR(120) NOT NULL UNIQUE, "
                "title VARCHAR(400) NOT NULL, "
                "doc_type VARCHAR(80) NOT NULL, "
                "authority VARCHAR(200), "
                "language_code VARCHAR(10) NOT NULL DEFAULT 'vi', "
                "effective_from DATE, "
                "effective_to DATE, "
                "status VARCHAR(30) NOT NULL DEFAULT 'active', "
                "source_uri VARCHAR(500), "
                "metadata_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS knowledge_document_versions ("
                "id SERIAL PRIMARY KEY, "
                "document_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE, "
                "version_tag VARCHAR(80) NOT NULL, "
                "content_hash VARCHAR(64), "
                "raw_text TEXT, "
                "parsed_json JSONB, "
                "ingestion_notes TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_doc_versions_doc_tag "
                "ON knowledge_document_versions (document_id, version_tag);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS knowledge_chunks ("
                "id SERIAL PRIMARY KEY, "
                "version_id INTEGER NOT NULL REFERENCES knowledge_document_versions(id) ON DELETE CASCADE, "
                "chunk_key VARCHAR(120) NOT NULL UNIQUE, "
                "chunk_index INTEGER NOT NULL, "
                "heading VARCHAR(300), "
                "chunk_text TEXT NOT NULL, "
                "token_count INTEGER, "
                "metadata_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS knowledge_citations ("
                "id SERIAL PRIMARY KEY, "
                "chunk_id INTEGER NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE, "
                "citation_key VARCHAR(140) NOT NULL UNIQUE, "
                "legal_reference VARCHAR(300), "
                "citation_text TEXT, "
                "confidence FLOAT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS knowledge_chunk_embeddings ("
                "id SERIAL PRIMARY KEY, "
                "chunk_id INTEGER NOT NULL UNIQUE REFERENCES knowledge_chunks(id) ON DELETE CASCADE, "
                "embedding_model VARCHAR(80) NOT NULL, "
                "embedding_dim INTEGER NOT NULL, "
                "embedding_json JSONB NOT NULL, "
                "embedding_source VARCHAR(40) NOT NULL DEFAULT 'ingestion', "
                "content_hash VARCHAR(64), "
                "indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text("ALTER TABLE knowledge_chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_source VARCHAR(40) NOT NULL DEFAULT 'ingestion';"))
            conn.execute(text("ALTER TABLE knowledge_chunk_embeddings ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);"))
            conn.execute(text("ALTER TABLE knowledge_chunk_embeddings ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"))
            if pgvector_ready:
                conn.execute(text("ALTER TABLE knowledge_chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_vector vector(384);"))
                conn.execute(text("ALTER TABLE knowledge_chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_hash_vector vector(96);"))
                for index_stmt in [
                    (
                        "CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_vector_hnsw "
                        "ON knowledge_chunk_embeddings USING hnsw (embedding_vector vector_cosine_ops) "
                        "WHERE embedding_vector IS NOT NULL;"
                    ),
                    (
                        "CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_vector_ivfflat "
                        "ON knowledge_chunk_embeddings USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100) "
                        "WHERE embedding_vector IS NOT NULL;"
                    ),
                    (
                        "CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_hash_hnsw "
                        "ON knowledge_chunk_embeddings USING hnsw (embedding_hash_vector vector_cosine_ops) "
                        "WHERE embedding_hash_vector IS NOT NULL;"
                    ),
                ]:
                    nested = None
                    try:
                        if hasattr(conn, "begin_nested"):
                            nested = conn.begin_nested()
                        conn.execute(text(index_stmt))
                        if nested is not None:
                            nested.commit()
                    except Exception as index_exc:
                        if nested is not None:
                            try:
                                nested.rollback()
                            except Exception:
                                pass
                        print(f"[WARN] pgvector ANN index skipped: {index_exc}")
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS vector_index_registry ("
                "id SERIAL PRIMARY KEY, "
                "index_key VARCHAR(120) NOT NULL UNIQUE, "
                "corpus_key VARCHAR(120) NOT NULL DEFAULT 'tax_knowledge', "
                "corpus_version VARCHAR(80), "
                "embedding_model VARCHAR(80) NOT NULL, "
                "embedding_dim INTEGER NOT NULL, "
                "index_type VARCHAR(40) NOT NULL DEFAULT 'none', "
                "build_params JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "corpus_hash VARCHAR(64), "
                "status VARCHAR(20) NOT NULL DEFAULT 'ready', "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS vector_index_quality_runs ("
                "id SERIAL PRIMARY KEY, "
                "index_id INTEGER NOT NULL REFERENCES vector_index_registry(id) ON DELETE CASCADE, "
                "run_key VARCHAR(120) NOT NULL UNIQUE, "
                "benchmark_key VARCHAR(120) NOT NULL DEFAULT 'tax_agent_core_v1', "
                "metrics_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_vector_index_quality_runs_index_created "
                "ON vector_index_quality_runs (index_id, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS retrieval_logs ("
                "id SERIAL PRIMARY KEY, "
                "request_id VARCHAR(120), "
                "session_id VARCHAR(120), "
                "query_text TEXT NOT NULL, "
                "query_hash VARCHAR(64), "
                "intent VARCHAR(80), "
                "entity_scope JSONB, "
                "retrieved_chunks JSONB, "
                "retrieval_scores JSONB, "
                "top_k INTEGER NOT NULL DEFAULT 5, "
                "corpus_version VARCHAR(200), "
                "index_key VARCHAR(120), "
                "embedding_tier VARCHAR(40), "
                "reranker_tier VARCHAR(40), "
                "query_embedding_hash VARCHAR(64), "
                "candidate_count INTEGER, "
                "citation_spans JSONB, "
                "latency_breakdown JSONB, "
                "latency_ms FLOAT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS corpus_version VARCHAR(200);"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS index_key VARCHAR(120);"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS embedding_tier VARCHAR(40);"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS reranker_tier VARCHAR(40);"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS query_embedding_hash VARCHAR(64);"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS candidate_count INTEGER;"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS citation_spans JSONB;"))
            conn.execute(text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS latency_breakdown JSONB;"))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(120) NOT NULL UNIQUE, "
                "user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, "
                "channel VARCHAR(40) NOT NULL DEFAULT 'chat', "
                "status VARCHAR(30) NOT NULL DEFAULT 'active', "
                "started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "ended_at TIMESTAMP, "
                "metadata_json JSONB"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_turns ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE, "
                "turn_index INTEGER NOT NULL, "
                "role VARCHAR(20) NOT NULL, "
                "message_text TEXT NOT NULL, "
                "normalized_intent VARCHAR(80), "
                "confidence FLOAT, "
                "citations_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_tool_calls ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE, "
                "turn_id INTEGER REFERENCES agent_turns(id) ON DELETE SET NULL, "
                "tool_name VARCHAR(120) NOT NULL, "
                "tool_input JSONB, "
                "tool_output JSONB, "
                "status VARCHAR(20) NOT NULL DEFAULT 'ok', "
                "latency_ms FLOAT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS policy_rules ("
                "id SERIAL PRIMARY KEY, "
                "rule_key VARCHAR(120) NOT NULL UNIQUE, "
                "rule_name VARCHAR(200) NOT NULL, "
                "rule_type VARCHAR(80) NOT NULL, "
                "severity VARCHAR(20) NOT NULL DEFAULT 'warning', "
                "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "config_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS policy_execution_logs ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(120), "
                "turn_id INTEGER REFERENCES agent_turns(id) ON DELETE SET NULL, "
                "rule_key VARCHAR(120), "
                "decision VARCHAR(40) NOT NULL, "
                "reason TEXT, "
                "score FLOAT, "
                "payload_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_decision_traces ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE, "
                "turn_id INTEGER REFERENCES agent_turns(id) ON DELETE SET NULL, "
                "intent VARCHAR(80), "
                "selected_track VARCHAR(80), "
                "confidence FLOAT, "
                "abstained BOOLEAN NOT NULL DEFAULT FALSE, "
                "escalation_required BOOLEAN NOT NULL DEFAULT FALSE, "
                "evidence_json JSONB, "
                "answer_text TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_feedback_events ("
                "id SERIAL PRIMARY KEY, "
                "session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE, "
                "turn_id INTEGER REFERENCES agent_turns(id) ON DELETE SET NULL, "
                "feedback_type VARCHAR(60) NOT NULL, "
                "rating FLOAT, "
                "notes TEXT, "
                "actor VARCHAR(80), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_eval_suites ("
                "id SERIAL PRIMARY KEY, "
                "suite_key VARCHAR(120) NOT NULL UNIQUE, "
                "description TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS agent_eval_runs ("
                "id SERIAL PRIMARY KEY, "
                "suite_id INTEGER NOT NULL REFERENCES agent_eval_suites(id) ON DELETE CASCADE, "
                "run_key VARCHAR(120) NOT NULL UNIQUE, "
                "model_version VARCHAR(80), "
                "metrics_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS prompt_registry ("
                "id SERIAL PRIMARY KEY, "
                "prompt_key VARCHAR(120) NOT NULL UNIQUE, "
                "description TEXT, "
                "owner VARCHAR(80), "
                "current_version VARCHAR(80), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS prompt_versions ("
                "id SERIAL PRIMARY KEY, "
                "prompt_id INTEGER NOT NULL REFERENCES prompt_registry(id) ON DELETE CASCADE, "
                "version_tag VARCHAR(80) NOT NULL, "
                "template_text TEXT NOT NULL, "
                "variables_json JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_versions_prompt_tag "
                "ON prompt_versions (prompt_id, version_tag);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS prompt_rollouts ("
                "id SERIAL PRIMARY KEY, "
                "prompt_key VARCHAR(120) NOT NULL, "
                "version_tag VARCHAR(80) NOT NULL, "
                "environment VARCHAR(40) NOT NULL DEFAULT 'staging', "
                "traffic_pct FLOAT NOT NULL DEFAULT 1.0, "
                "status VARCHAR(30) NOT NULL DEFAULT 'planned', "
                "notes TEXT, "
                "metadata_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_prompt_rollouts_key_env_created "
                "ON prompt_rollouts (prompt_key, environment, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS policy_rule_versions ("
                "id SERIAL PRIMARY KEY, "
                "rule_key VARCHAR(120) NOT NULL, "
                "version_tag VARCHAR(80) NOT NULL, "
                "config_json JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "changed_by VARCHAR(80), "
                "approved_by VARCHAR(80), "
                "effective_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "effective_to TIMESTAMP, "
                "change_reason TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_policy_rule_versions_key_created "
                "ON policy_rule_versions (rule_key, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS tool_execution_outcomes ("
                "id SERIAL PRIMARY KEY, "
                "tool_call_id INTEGER NOT NULL REFERENCES agent_tool_calls(id) ON DELETE CASCADE, "
                "outcome_type VARCHAR(40) NOT NULL, "
                "error_class VARCHAR(120), "
                "retry_count INTEGER NOT NULL DEFAULT 0, "
                "side_effect_level VARCHAR(30) NOT NULL DEFAULT 'none', "
                "metadata_json JSONB, "
                "finalized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_tool_execution_outcomes_tool_call "
                "ON tool_execution_outcomes (tool_call_id);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS adjudication_cases ("
                "id SERIAL PRIMARY KEY, "
                "entity_type VARCHAR(40) NOT NULL, "
                "entity_id VARCHAR(120) NOT NULL, "
                "model_name VARCHAR(80), "
                "model_version VARCHAR(80), "
                "model_label VARCHAR(80), "
                "human_label VARCHAR(80), "
                "final_label VARCHAR(80), "
                "status VARCHAR(30) NOT NULL DEFAULT 'open', "
                "resolver_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, "
                "dispute_reason TEXT, "
                "resolution_notes TEXT, "
                "resolved_at TIMESTAMP, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_adjudication_entity_status "
                "ON adjudication_cases (entity_type, entity_id, status);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS redteam_scenarios ("
                "id SERIAL PRIMARY KEY, "
                "scenario_key VARCHAR(120) NOT NULL UNIQUE, "
                "taxonomy VARCHAR(80) NOT NULL, "
                "prompt_text TEXT NOT NULL, "
                "expected_guardrail VARCHAR(120), "
                "severity VARCHAR(20) NOT NULL DEFAULT 'medium', "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS redteam_run_results ("
                "id SERIAL PRIMARY KEY, "
                "run_key VARCHAR(120) NOT NULL, "
                "scenario_id INTEGER NOT NULL REFERENCES redteam_scenarios(id) ON DELETE CASCADE, "
                "model_name VARCHAR(80) NOT NULL, "
                "model_version VARCHAR(80), "
                "outcome VARCHAR(40) NOT NULL, "
                "severity VARCHAR(20), "
                "trace_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_redteam_run_results_key_created "
                "ON redteam_run_results (run_key, created_at DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS invoice_events ("
                "id SERIAL PRIMARY KEY, "
                "invoice_number VARCHAR(50) NOT NULL REFERENCES invoices(invoice_number) ON DELETE CASCADE, "
                "event_type VARCHAR(30) NOT NULL, "
                "event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "reason VARCHAR(200), "
                "replaced_invoice_number VARCHAR(50), "
                "payload_json JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_invoice_events_invoice_time "
                "ON invoice_events (invoice_number, event_time DESC);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_invoice_events_type_time "
                "ON invoice_events (event_type, event_time DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS invoice_fingerprints ("
                "id SERIAL PRIMARY KEY, "
                "invoice_number VARCHAR(50) NOT NULL UNIQUE REFERENCES invoices(invoice_number) ON DELETE CASCADE, "
                "hash_near_dup VARCHAR(64), "
                "hash_line_items VARCHAR(64), "
                "hash_counterparty VARCHAR(64), "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_invoice_fingerprints_near_dup "
                "ON invoice_fingerprints (hash_near_dup);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS invoice_risk_predictions ("
                "id SERIAL PRIMARY KEY, "
                "invoice_number VARCHAR(50) NOT NULL REFERENCES invoices(invoice_number) ON DELETE CASCADE, "
                "as_of_date DATE NOT NULL, "
                "model_version VARCHAR(80), "
                "risk_score FLOAT NOT NULL DEFAULT 0.0, "
                "risk_level VARCHAR(20) NOT NULL DEFAULT 'low', "
                "reason_codes JSONB, "
                "explanations JSONB, "
                "linked_invoice_ids JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_invoice_risk_predictions_invoice_date "
                "ON invoice_risk_predictions (invoice_number, as_of_date DESC);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_invoice_risk_predictions_score "
                "ON invoice_risk_predictions (risk_score DESC);"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS vat_refund_cases ("
                "case_id VARCHAR(40) PRIMARY KEY, "
                "tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE, "
                "period VARCHAR(20) NOT NULL, "
                "requested_amount NUMERIC(18, 2) NOT NULL DEFAULT 0.0, "
                "submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "status VARCHAR(30) NOT NULL DEFAULT 'submitted', "
                "channel VARCHAR(30), "
                "documents_score FLOAT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS vat_refund_case_links ("
                "id SERIAL PRIMARY KEY, "
                "case_id VARCHAR(40) NOT NULL REFERENCES vat_refund_cases(case_id) ON DELETE CASCADE, "
                "invoice_number VARCHAR(50) NOT NULL REFERENCES invoices(invoice_number) ON DELETE CASCADE, "
                "link_type VARCHAR(20) NOT NULL DEFAULT 'supporting', "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS vat_refund_predictions ("
                "id SERIAL PRIMARY KEY, "
                "case_id VARCHAR(40) NOT NULL REFERENCES vat_refund_cases(case_id) ON DELETE CASCADE, "
                "as_of_date DATE NOT NULL, "
                "model_version VARCHAR(80), "
                "risk_score FLOAT NOT NULL DEFAULT 0.0, "
                "expected_loss NUMERIC(18, 2) NOT NULL DEFAULT 0.0, "
                "reason_codes JSONB, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_kpi_snapshot_track_metric_ts "
                "ON kpi_metric_snapshots (track_name, metric_name, generated_at DESC);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_kpi_snapshot_generated_at "
                "ON kpi_metric_snapshots (generated_at DESC);"
            ))

            conn.execute(text(
                "INSERT INTO kpi_trigger_policies (track_name, metric_name, comparator, threshold, min_sample, window_days, cooldown_days, enabled, rationale) "
                "SELECT 'audit_value', 'precision_top_50', '>=', 0.70, 50, 28, 14, TRUE, 'Top-50 hồ sơ Audit Value cần precision ổn định trước split.' "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM kpi_trigger_policies WHERE track_name='audit_value' AND metric_name='precision_top_50'"
                ");"
            ))
            conn.execute(text(
                "INSERT INTO kpi_trigger_policies (track_name, metric_name, comparator, threshold, min_sample, window_days, cooldown_days, enabled, rationale) "
                "SELECT 'audit_value', 'roi_positive_rate', '>=', 0.80, 50, 28, 14, TRUE, 'Tối thiểu 80% case can thiệp phải có net recovery dương.' "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM kpi_trigger_policies WHERE track_name='audit_value' AND metric_name='roi_positive_rate'"
                ");"
            ))
            conn.execute(text(
                "INSERT INTO kpi_trigger_policies (track_name, metric_name, comparator, threshold, min_sample, window_days, cooldown_days, enabled, rationale) "
                "SELECT 'vat_refund', 'precision_top_100', '>=', 0.65, 80, 28, 14, TRUE, 'Top-100 VAT queue cần precision đủ mạnh trước split.' "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM kpi_trigger_policies WHERE track_name='vat_refund' AND metric_name='precision_top_100'"
                ");"
            ))
            conn.execute(text(
                "INSERT INTO kpi_trigger_policies (track_name, metric_name, comparator, threshold, min_sample, window_days, cooldown_days, enabled, rationale) "
                "SELECT 'vat_refund', 'false_negative_rate_high_risk', '<=', 0.12, 50, 28, 14, TRUE, 'Giữ FN high-risk VAT ở mức thấp để tránh bỏ sót hồ sơ.' "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM kpi_trigger_policies WHERE track_name='vat_refund' AND metric_name='false_negative_rate_high_risk'"
                ");"
            ))

            conn.commit()
        print("[OK] Schema migration: all flagship columns verified.")
    except Exception as e:
        print(f"[WARN] Schema migration skipped (table may not exist yet): {e}")

    # --- Periodic cache cleanup scheduler (every 24 hours) ---
    try:
        import threading
        import time as _time
        from app.tasks import cleanup_stale_assessments

        def _cache_cleanup_loop():
            """Run cleanup immediately on startup, then every 24 hours."""
            while True:
                try:
                    cleanup_stale_assessments(max_age_days=30)
                except Exception as exc:
                    print(f"[WARN] Periodic cache cleanup error: {exc}")
                _time.sleep(86400)  # 24 hours

        t = threading.Thread(target=_cache_cleanup_loop, daemon=True)
        t.start()
        print("[OK] Cache cleanup scheduler started (runs every 24h).")
    except Exception as e:
        print(f"[WARN] Cache cleanup scheduler failed to start: {e}")

    # --- Periodic KPI snapshot capture scheduler (every 6 hours by default) ---
    try:
        import threading
        import time as _time

        kpi_snapshot_interval = int(os.getenv("KPI_SNAPSHOT_INTERVAL_SECONDS", "21600"))
        kpi_snapshot_interval = max(900, kpi_snapshot_interval)  # minimum 15 minutes

        def _kpi_snapshot_loop():
            """Persist split-trigger KPI snapshots on a fixed cadence for trend alerting."""
            while True:
                try:
                    payload = monitoring.get_split_trigger_status_snapshot(
                        persist_snapshot=True,
                        snapshot_source="scheduler_periodic_capture",
                    )
                    captured = int(payload.get("snapshots_captured") or 0)
                    ready = bool(payload.get("ready", False))
                    print(f"[OK] KPI snapshot captured rows={captured}, ready={ready}.")
                except Exception as exc:
                    print(f"[WARN] Periodic KPI snapshot capture error: {exc}")
                _time.sleep(kpi_snapshot_interval)

        t_kpi = threading.Thread(target=_kpi_snapshot_loop, daemon=True)
        t_kpi.start()
        print(f"[OK] KPI snapshot scheduler started (runs every {kpi_snapshot_interval}s).")
    except Exception as e:
        print(f"[WARN] KPI snapshot scheduler failed to start: {e}")

    yield

app = FastAPI(
    title="TaxInspector ML API",
    description="API hệ thống giám sát thuế tích hợp Machine Learning: "
                "Fraud Risk Scoring, VAT Invoice Graph, Delinquency Prediction, "
                "Temporal Compliance Intelligence, Graph Intelligence 2.0, "
                "Investigator Decision Intelligence.",
    version="3.0.0-FLAGSHIP",
    lifespan=lifespan,
)

# --- Rate Limiter ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# --- CORS (allow_credentials=True for HttpOnly Cookie) ---
default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://[::1]:3000,http://[::]:3000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,        # CRITICAL: Required for cookie-based auth
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security Headers Middleware ---
app.add_middleware(SecurityHeadersMiddleware)

# --- Register Routers ---
app.include_router(auth.router)
app.include_router(scoring.router)
app.include_router(graph.router)
app.include_router(delinquency.router)
app.include_router(ai_analysis.router)
app.include_router(monitoring.router)
app.include_router(simulation.router)
app.include_router(osint.router)
app.include_router(invoice_risk.router)
app.include_router(vat_refund.router)
app.include_router(entity_resolution.router)
app.include_router(transfer_pricing.router)
app.include_router(audit_selection.router)
app.include_router(collections.router)
app.include_router(case_triage.router)
app.include_router(tax_agent.router)
app.include_router(ml_api.router)


@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "version": "3.0.0-FLAGSHIP",
        "security": {
            "cookie_auth": True,
            "rate_limiting": True,
            "security_headers": True,
            "data_encryption": True,
            "audit_logging": True,
        },
        "flagship_models": {
            "temporal_compliance": "Program A – active",
            "graph_intelligence": "Program B – active",
            "decision_intelligence": "Program C – active",
        },
        "message": "TaxInspector API is running with full security hardening + flagship models.",
    }
