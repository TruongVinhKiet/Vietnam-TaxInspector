"""Shared SQL bootstrap helpers for durable agent chat state."""

from __future__ import annotations

from sqlalchemy import text


def ensure_agent_resilience_schema(conn) -> None:
    """Create/upgrade agent resilience tables shared by startup and migration."""
    try:
        conn.execute(text("""
            WITH ranked AS (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY session_id
                    ORDER BY turn_index ASC, id ASC
                ) AS new_turn_index
                FROM agent_turns
            )
            UPDATE agent_turns AS t
            SET turn_index = ranked.new_turn_index
            FROM ranked
            WHERE t.id = ranked.id
              AND t.turn_index <> ranked.new_turn_index;
        """))
    except Exception:
        # Best effort: older dev DBs may not support the CTE/window syntax.
        pass
    conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_turns_session_turn
        ON agent_turns (session_id, turn_index);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_agent_turns_session_turn_desc
        ON agent_turns (session_id, turn_index DESC);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS agent_prior_answer_facts (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
            turn_id INTEGER REFERENCES agent_turns(id) ON DELETE SET NULL,
            mode VARCHAR(40) NOT NULL DEFAULT 'full',
            intent VARCHAR(80),
            subject_key VARCHAR(160),
            fact_type VARCHAR(80) NOT NULL DEFAULT 'claim',
            claim_text TEXT NOT NULL,
            value_json JSONB,
            source_tool VARCHAR(120),
            confidence FLOAT NOT NULL DEFAULT 0.75,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMPTZ
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_agent_prior_answer_facts_session_mode_exp
        ON agent_prior_answer_facts (session_id, mode, expires_at);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_agent_prior_answer_facts_subject
        ON agent_prior_answer_facts (session_id, subject_key, created_at DESC);
    """))
    conn.execute(text("""
        ALTER TABLE agent_async_file_jobs
        ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ;
    """))

    # PostgreSQL-only governance checks. They are NOT VALID where possible to
    # avoid breaking existing bootstrap data; new rows still get clear contracts.
    checks = [
        (
            "ck_agent_route_events_model_mode",
            "agent_route_events",
            "model_mode IN ('full','fraud','vat','delinquency','macro','legal')",
        ),
        (
            "ck_agent_route_events_dialogue_act",
            "agent_route_events",
            "dialogue_act IN ('task','greeting','smalltalk','thanks','goodbye','help','clarification')",
        ),
        (
            "ck_agent_route_events_answer_contract",
            "agent_route_events",
            "answer_contract IN ('smalltalk','data_table','risk_profile','fraud_analysis','legal_consultation','vat_graph','file_analysis','clarification','mode_mismatch')",
        ),
        (
            "ck_agent_async_file_jobs_status",
            "agent_async_file_jobs",
            "status IN ('pending','processing','done','error','cancelled')",
        ),
        (
            "ck_agent_prior_answer_facts_mode",
            "agent_prior_answer_facts",
            "mode IN ('full','fraud','vat','delinquency','macro','legal')",
        ),
    ]
    for name, table, expr in checks:
        try:
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{name}') THEN
                        ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({expr}) NOT VALID;
                    END IF;
                END $$;
            """))
        except Exception:
            # SQLite/dev fixtures do not support DO blocks or NOT VALID.
            pass
