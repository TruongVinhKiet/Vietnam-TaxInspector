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
from .routers import scoring, graph, delinquency, auth, ai_analysis, monitoring
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
