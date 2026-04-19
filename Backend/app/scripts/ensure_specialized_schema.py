"""
ensure_specialized_schema.py

Check specialized monitoring/training schema and apply missing SQL objects.

Scope:
- ai_risk_assessments required columns used by specialized training/monitoring
- inspector_labels required columns used by import/train/monitoring
- data_reality_audit_logs table + indexes
- selected indexes for specialized operations

Usage (from Backend):
    python app/scripts/ensure_specialized_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import engine


ASSESSMENT_ALTERS = [
    "ALTER TABLE ai_risk_assessments ADD COLUMN IF NOT EXISTS model_confidence FLOAT;",
    "ALTER TABLE ai_risk_assessments ADD COLUMN IF NOT EXISTS yearly_history JSON;",
    "ALTER TABLE ai_risk_assessments ADD COLUMN IF NOT EXISTS model_version VARCHAR(80);",
]

LABEL_ALTERS = [
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS label_origin VARCHAR(40) DEFAULT 'manual_inspector';",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS model_version VARCHAR(80);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS intervention_action VARCHAR(50);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS intervention_attempted BOOLEAN DEFAULT FALSE;",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS outcome_status VARCHAR(30);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS predicted_collection_uplift NUMERIC(18, 2);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS expected_recovery NUMERIC(18, 2);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS expected_net_recovery NUMERIC(18, 2);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS estimated_audit_cost NUMERIC(18, 2);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS actual_audit_cost NUMERIC(18, 2);",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS actual_audit_hours FLOAT;",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS outcome_recorded_at TIMESTAMP;",
    "ALTER TABLE inspector_labels ADD COLUMN IF NOT EXISTS kpi_window_days INTEGER DEFAULT 90;",
]

DATA_REALITY_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS data_reality_audit_logs (
    id SERIAL PRIMARY KEY,
    source_endpoint VARCHAR(80) NOT NULL,
    status VARCHAR(20) NOT NULL,
    ready_for_real_ops BOOLEAN NOT NULL,
    hard_ready BOOLEAN NOT NULL,
    reasons JSONB,
    hard_checks JSONB,
    soft_checks JSONB,
    metrics JSONB,
    generated_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_inspector_labels_model_version ON inspector_labels (model_version);",
    "CREATE INDEX IF NOT EXISTS idx_inspector_labels_label_origin ON inspector_labels (label_origin);",
    "CREATE INDEX IF NOT EXISTS idx_data_reality_audit_logs_created ON data_reality_audit_logs (created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_data_reality_audit_logs_source_status ON data_reality_audit_logs (source_endpoint, status, created_at DESC);",
]

POST_MIGRATION_SQL = [
    "UPDATE inspector_labels SET label_origin = 'manual_inspector' WHERE label_origin IS NULL OR btrim(label_origin) = '';",
    """
    UPDATE inspector_labels l
    SET model_version = a.model_version
    FROM ai_risk_assessments a
    WHERE l.assessment_id = a.id
      AND (l.model_version IS NULL OR btrim(l.model_version) = '')
      AND a.model_version IS NOT NULL
      AND btrim(a.model_version) <> '';
    """,
]


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
            LIMIT 1
            """
        ),
        {"table_name": table_name},
    ).first()
    return row is not None


def _count_table_columns(conn, table_name: str) -> int:
    row = conn.execute(
        text(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name},
    ).first()
    return int((row[0] if row else 0) or 0)


def run() -> int:
    print("=" * 72)
    print("  Ensure Specialized Schema")
    print("=" * 72)

    executed = 0
    with engine.connect() as conn:
        for sql in ASSESSMENT_ALTERS:
            conn.execute(text(sql))
            executed += 1

        for sql in LABEL_ALTERS:
            conn.execute(text(sql))
            executed += 1

        conn.execute(text(DATA_REALITY_AUDIT_TABLE_SQL))
        executed += 1

        for sql in INDEX_SQL:
            conn.execute(text(sql))
            executed += 1

        for sql in POST_MIGRATION_SQL:
            conn.execute(text(sql))
            executed += 1

        conn.commit()

        tables = ["ai_risk_assessments", "inspector_labels", "data_reality_audit_logs"]
        print(f"[OK] sql_statements_executed={executed}")
        for table_name in tables:
            exists = _table_exists(conn, table_name)
            col_count = _count_table_columns(conn, table_name) if exists else 0
            print(f"[OK] table={table_name} exists={exists} column_count={col_count}")

    print("[DONE] specialized schema check and migration completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
