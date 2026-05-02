import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from Backend.app.database import engine


def _run_safe_user_profile_migration(conn) -> None:
    for col_sql in [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS signature_data TEXT;",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS signature_verified BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_data TEXT;",
    ]:
        conn.execute(text(col_sql))


def _run_offshore_proxy_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS offshore_entities (
            id SERIAL PRIMARY KEY,
            entity_code VARCHAR(30) UNIQUE NOT NULL,
            proxy_tax_code VARCHAR(20),
            name VARCHAR(255) NOT NULL,
            country VARCHAR(100) NOT NULL,
            jurisdiction_risk_weight FLOAT DEFAULT 0.5,
            risk_score FLOAT DEFAULT 50.0,
            entity_type VARCHAR(50) DEFAULT 'shell_company',
            registration_date DATE,
            status VARCHAR(30) DEFAULT 'active',
            data_source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("ALTER TABLE offshore_entities ADD COLUMN IF NOT EXISTS proxy_tax_code VARCHAR(20);"))

    conn.execute(text("""
        WITH unresolved AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY id) AS rn
            FROM offshore_entities
            WHERE proxy_tax_code IS NULL OR proxy_tax_code !~ '^[0-9]{10}$'
        ),
        current_max AS (
            SELECT COALESCE(MAX((SUBSTRING(code FROM 3 FOR 8))::int), 0) AS max_suffix
            FROM (
                SELECT proxy_tax_code AS code
                FROM offshore_entities
                WHERE proxy_tax_code ~ '^99[0-9]{8}$'

                UNION

                SELECT tax_code AS code
                FROM companies
                WHERE tax_code ~ '^99[0-9]{8}$'
            ) existing_codes
        ),
        generated AS (
            SELECT u.id,
                   ('99' || LPAD((current_max.max_suffix + u.rn)::text, 8, '0')) AS generated_proxy
            FROM unresolved u
            CROSS JOIN current_max
        ),
        seeded_companies AS (
            INSERT INTO companies (tax_code, name, industry, province, risk_score, is_active)
            SELECT
                g.generated_proxy,
                COALESCE('[OFFSHORE] ' || oe.name, '[OFFSHORE] ' || oe.entity_code),
                'Offshore Entity',
                oe.country,
                COALESCE(oe.risk_score, 70),
                TRUE
            FROM generated g
            JOIN offshore_entities oe ON oe.id = g.id
            ON CONFLICT (tax_code) DO UPDATE
            SET name = EXCLUDED.name,
                industry = EXCLUDED.industry,
                province = EXCLUDED.province,
                risk_score = EXCLUDED.risk_score,
                is_active = TRUE
            RETURNING tax_code
        )
        UPDATE offshore_entities oe
        SET proxy_tax_code = g.generated_proxy
        FROM generated g
        JOIN seeded_companies sc ON sc.tax_code = g.generated_proxy
        WHERE oe.id = g.id;
    """))

    conn.execute(text("""
        INSERT INTO companies (tax_code, name, industry, province, risk_score, is_active)
        SELECT
            oe.proxy_tax_code,
            COALESCE('[OFFSHORE] ' || oe.name, '[OFFSHORE] ' || oe.entity_code),
            'Offshore Entity',
            oe.country,
            COALESCE(oe.risk_score, 70),
            TRUE
        FROM offshore_entities oe
        WHERE oe.proxy_tax_code ~ '^[0-9]{10}$'
        ON CONFLICT (tax_code) DO UPDATE
        SET name = EXCLUDED.name,
            industry = 'Offshore Entity',
            province = EXCLUDED.province,
            risk_score = GREATEST(COALESCE(companies.risk_score, 0), COALESCE(EXCLUDED.risk_score, 0)),
            is_active = TRUE;
    """))

    conn.execute(text("""
        UPDATE ownership_links ol
        SET parent_tax_code = oe.proxy_tax_code
        FROM offshore_entities oe
        WHERE ol.parent_tax_code = oe.entity_code
          AND oe.proxy_tax_code ~ '^[0-9]{10}$';
    """))

    conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'offshore_entities_proxy_tax_code_fkey'
            ) THEN
                ALTER TABLE offshore_entities
                    ADD CONSTRAINT offshore_entities_proxy_tax_code_fkey
                    FOREIGN KEY (proxy_tax_code) REFERENCES companies(tax_code) ON DELETE SET NULL;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'offshore_entities_proxy_tax_code_unique'
            ) THEN
                ALTER TABLE offshore_entities
                    ADD CONSTRAINT offshore_entities_proxy_tax_code_unique
                    UNIQUE (proxy_tax_code);
            END IF;
        END $$;
    """))


def _run_numeric_tax_code_contract_migration(conn) -> None:
    conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_companies_tax_code_numeric10'
            ) THEN
                ALTER TABLE companies
                    ADD CONSTRAINT ck_companies_tax_code_numeric10
                    CHECK (tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_tax_returns_tax_code_numeric10'
            ) THEN
                ALTER TABLE tax_returns
                    ADD CONSTRAINT ck_tax_returns_tax_code_numeric10
                    CHECK (tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_invoices_seller_tax_code_numeric10'
            ) THEN
                ALTER TABLE invoices
                    ADD CONSTRAINT ck_invoices_seller_tax_code_numeric10
                    CHECK (seller_tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_invoices_buyer_tax_code_numeric10'
            ) THEN
                ALTER TABLE invoices
                    ADD CONSTRAINT ck_invoices_buyer_tax_code_numeric10
                    CHECK (buyer_tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_ownership_links_parent_tax_code_numeric10'
            ) THEN
                ALTER TABLE ownership_links
                    ADD CONSTRAINT ck_ownership_links_parent_tax_code_numeric10
                    CHECK (parent_tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_ownership_links_child_tax_code_numeric10'
            ) THEN
                ALTER TABLE ownership_links
                    ADD CONSTRAINT ck_ownership_links_child_tax_code_numeric10
                    CHECK (child_tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_offshore_entities_proxy_tax_code_numeric10'
            ) THEN
                ALTER TABLE offshore_entities
                    ADD CONSTRAINT ck_offshore_entities_proxy_tax_code_numeric10
                    CHECK (proxy_tax_code IS NULL OR proxy_tax_code ~ '^[0-9]{10}$') NOT VALID;
            END IF;
        END $$;
    """))


def _run_feature_store_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_sets (
            id SERIAL PRIMARY KEY,
            name VARCHAR(80) NOT NULL,
            version VARCHAR(40) NOT NULL,
            owner VARCHAR(80),
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))


def _run_model_registry_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80) NOT NULL,
            artifact_path VARCHAR(400),
            feature_set_id INTEGER REFERENCES feature_sets(id) ON DELETE SET NULL,
            train_data_hash VARCHAR(64),
            code_hash VARCHAR(64),
            metrics_json JSONB,
            gates_json JSONB,
            status VARCHAR(20) NOT NULL DEFAULT 'staging',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_name_version
        ON model_registry (model_name, model_version);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_model_registry_status
        ON model_registry (model_name, status);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_model_registry_created
        ON model_registry (created_at DESC);
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS inference_audit_logs (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80) NOT NULL,
            request_id VARCHAR(64),
            actor_badge_id VARCHAR(50),
            actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            entity_type VARCHAR(20) NOT NULL,
            entity_id VARCHAR(120) NOT NULL,
            as_of_date DATE,
            input_feature_hash VARCHAR(64),
            output_hash VARCHAR(64),
            outputs_json JSONB,
            explanation_ref VARCHAR(200),
            latency_ms FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_infer_audit_model_ts
        ON inference_audit_logs (model_name, created_at DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_infer_audit_entity_ts
        ON inference_audit_logs (entity_type, entity_id, created_at DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_infer_audit_request
        ON inference_audit_logs (request_id);
    """))


def _run_drift_telemetry_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS model_quality_snapshots (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80),
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status VARCHAR(20) DEFAULT 'unknown',
            status_reason VARCHAR(120),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_model_quality_snapshots_model_ts
        ON model_quality_snapshots (model_name, created_at DESC);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_drift_stats (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80),
            feature_name VARCHAR(120) NOT NULL,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            psi FLOAT,
            ks FLOAT,
            missing_rate FLOAT,
            mean FLOAT,
            std FLOAT,
            baseline_mean FLOAT,
            baseline_std FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_feature_drift_stats_model_feature_ts
        ON feature_drift_stats (model_name, feature_name, created_at DESC);
    """))


def _run_invoice_risk_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS invoice_events (
            id SERIAL PRIMARY KEY,
            invoice_number VARCHAR(50) NOT NULL REFERENCES invoices(invoice_number) ON DELETE CASCADE,
            event_type VARCHAR(30) NOT NULL,
            event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            reason VARCHAR(200),
            replaced_invoice_number VARCHAR(50),
            payload_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_invoice_events_invoice_time
        ON invoice_events (invoice_number, event_time DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_invoice_events_type_time
        ON invoice_events (event_type, event_time DESC);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS invoice_fingerprints (
            id SERIAL PRIMARY KEY,
            invoice_number VARCHAR(50) NOT NULL UNIQUE REFERENCES invoices(invoice_number) ON DELETE CASCADE,
            hash_near_dup VARCHAR(64),
            hash_line_items VARCHAR(64),
            hash_counterparty VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_invoice_fingerprints_near_dup
        ON invoice_fingerprints (hash_near_dup);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS invoice_risk_predictions (
            id SERIAL PRIMARY KEY,
            invoice_number VARCHAR(50) NOT NULL REFERENCES invoices(invoice_number) ON DELETE CASCADE,
            as_of_date DATE NOT NULL,
            model_version VARCHAR(80),
            risk_score FLOAT NOT NULL DEFAULT 0.0,
            risk_level VARCHAR(20) NOT NULL DEFAULT 'low',
            reason_codes JSONB,
            explanations JSONB,
            linked_invoice_ids JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_invoice_risk_predictions_invoice_date
        ON invoice_risk_predictions (invoice_number, as_of_date DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_invoice_risk_predictions_score
        ON invoice_risk_predictions (risk_score DESC);
    """))


def _run_vat_refund_case_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vat_refund_cases (
            case_id VARCHAR(40) PRIMARY KEY,
            tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            period VARCHAR(20) NOT NULL,
            requested_amount NUMERIC(18, 2) NOT NULL DEFAULT 0.0,
            submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(30) NOT NULL DEFAULT 'submitted',
            channel VARCHAR(30),
            documents_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_vat_refund_cases_tax_period
        ON vat_refund_cases (tax_code, period);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vat_refund_case_links (
            id SERIAL PRIMARY KEY,
            case_id VARCHAR(40) NOT NULL REFERENCES vat_refund_cases(case_id) ON DELETE CASCADE,
            invoice_number VARCHAR(50) NOT NULL REFERENCES invoices(invoice_number) ON DELETE CASCADE,
            link_type VARCHAR(20) NOT NULL DEFAULT 'supporting',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_vat_refund_case_links_case
        ON vat_refund_case_links (case_id);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vat_refund_predictions (
            id SERIAL PRIMARY KEY,
            case_id VARCHAR(40) NOT NULL REFERENCES vat_refund_cases(case_id) ON DELETE CASCADE,
            as_of_date DATE NOT NULL,
            model_version VARCHAR(80),
            risk_score FLOAT NOT NULL DEFAULT 0.0,
            expected_loss NUMERIC(18, 2) NOT NULL DEFAULT 0.0,
            reason_codes JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_vat_refund_predictions_case_date
        ON vat_refund_predictions (case_id, as_of_date DESC);
    """))


def _run_entity_resolution_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS entity_identities (
            id SERIAL PRIMARY KEY,
            tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            legal_name VARCHAR(255),
            normalized_name VARCHAR(255),
            address TEXT,
            phone VARCHAR(30),
            email VARCHAR(120),
            representative_name VARCHAR(255),
            representative_id VARCHAR(50),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_entity_identities_tax_code ON entity_identities (tax_code);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_entity_identities_rep_id ON entity_identities (representative_id);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS entity_alias_edges (
            id SERIAL PRIMARY KEY,
            src_tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            dst_tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            edge_type VARCHAR(30) NOT NULL,
            score FLOAT NOT NULL DEFAULT 0.0,
            evidence_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_entity_alias_src_dst ON entity_alias_edges (src_tax_code, dst_tax_code);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_entity_alias_type_score ON entity_alias_edges (edge_type, score DESC);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS phoenix_candidates (
            id SERIAL PRIMARY KEY,
            old_tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            new_tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
            score FLOAT NOT NULL DEFAULT 0.0,
            signals_json JSONB,
            as_of_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phoenix_candidates_old_new ON phoenix_candidates (old_tax_code, new_tax_code);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phoenix_candidates_score ON phoenix_candidates (score DESC);"))
    conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_sets_name_version
        ON feature_sets (name, version);
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_snapshots (
            id SERIAL PRIMARY KEY,
            entity_type VARCHAR(20) NOT NULL,
            entity_id VARCHAR(120) NOT NULL,
            as_of_date DATE NOT NULL,
            feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
            features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_entity_asof
        ON feature_snapshots (entity_type, entity_id, as_of_date DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_feature_set_asof
        ON feature_snapshots (feature_set_id, as_of_date DESC);
    """))
    conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'feature_snapshots_feature_set_id_fkey'
            ) THEN
                ALTER TABLE feature_snapshots
                    ADD CONSTRAINT feature_snapshots_feature_set_id_fkey
                    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id) ON DELETE CASCADE;
            END IF;
        END $$;
    """))


def _run_multimodal_upload_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS analysis_uploads (
            id SERIAL PRIMARY KEY,
            source VARCHAR(80) NOT NULL DEFAULT 'unknown',
            batch_type VARCHAR(80) NOT NULL DEFAULT 'generic',
            original_filename VARCHAR(500) NOT NULL,
            stored_filename VARCHAR(500),
            file_path VARCHAR(1000),
            content_type VARCHAR(120),
            file_size_bytes INTEGER NOT NULL DEFAULT 0,
            sha256 VARCHAR(64) NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'received',
            error_message TEXT,
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_analysis_uploads_source_created ON analysis_uploads (source, created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_analysis_uploads_sha256 ON analysis_uploads (sha256);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_analysis_uploads_status ON analysis_uploads (status);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vat_graph_analysis_batches (
            id SERIAL PRIMARY KEY,
            upload_id INTEGER REFERENCES analysis_uploads(id) ON DELETE SET NULL,
            filename VARCHAR(500) NOT NULL,
            detected_schema VARCHAR(80),
            total_rows INTEGER DEFAULT 0,
            processed_rows INTEGER DEFAULT 0,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            error_message TEXT,
            warnings JSONB,
            result_summary JSONB,
            result_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vat_graph_batches_status_created ON vat_graph_analysis_batches (status, created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vat_graph_batches_upload ON vat_graph_analysis_batches (upload_id);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vat_graph_batch_results (
            id SERIAL PRIMARY KEY,
            batch_id INTEGER NOT NULL REFERENCES vat_graph_analysis_batches(id) ON DELETE CASCADE,
            invoice_number VARCHAR(80) NOT NULL,
            seller_tax_code VARCHAR(20) NOT NULL,
            buyer_tax_code VARCHAR(20) NOT NULL,
            amount NUMERIC(18, 2) NOT NULL DEFAULT 0.0,
            vat_rate NUMERIC(5, 2) NOT NULL DEFAULT 10.0,
            invoice_date DATE NOT NULL,
            edge_risk_score DOUBLE PRECISION,
            edge_risk_level VARCHAR(20),
            signals JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vat_graph_batch_results_batch_score ON vat_graph_batch_results (batch_id, edge_risk_score DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vat_graph_batch_results_seller ON vat_graph_batch_results (seller_tax_code);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vat_graph_batch_results_buyer ON vat_graph_batch_results (buyer_tax_code);"))


def run_migration() -> None:
    with engine.begin() as conn:
        _run_safe_user_profile_migration(conn)
        _run_offshore_proxy_migration(conn)
        _run_numeric_tax_code_contract_migration(conn)
        _run_feature_store_migration(conn)
        _run_model_registry_migration(conn)
        _run_drift_telemetry_migration(conn)
        _run_invoice_risk_migration(conn)
        _run_vat_refund_case_migration(conn)
        _run_entity_resolution_migration(conn)
        _run_multimodal_upload_migration(conn)

    print("[OK] Completed migration: user profile columns + offshore proxy mapping + numeric tax_code contract + multimodal uploads.")


if __name__ == "__main__":
    run_migration()
