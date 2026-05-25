import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from Backend.app.agent_schema import ensure_agent_resilience_schema
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


def _run_legal_agent_v2_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS kg_entities (
            id SERIAL PRIMARY KEY,
            entity_key VARCHAR(200) NOT NULL UNIQUE,
            entity_type VARCHAR(60) NOT NULL,
            display_name VARCHAR(500) NOT NULL,
            description TEXT,
            authority_rank INTEGER DEFAULT 50,
            effective_from DATE,
            effective_to DATE,
            status VARCHAR(30) DEFAULT 'active',
            chunk_ids INTEGER[],
            attributes_json JSONB DEFAULT '{}'::jsonb,
            embedding_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_entities_status ON kg_entities(status);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_entities_authority ON kg_entities(authority_rank DESC);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS kg_relations (
            id SERIAL PRIMARY KEY,
            source_entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
            target_entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
            relation_type VARCHAR(60) NOT NULL,
            weight FLOAT DEFAULT 1.0,
            confidence FLOAT DEFAULT 0.8,
            evidence_text TEXT,
            attributes_json JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_entity_id, target_entity_id, relation_type)
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_entity_id);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_entity_id);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS kg_communities (
            id SERIAL PRIMARY KEY,
            community_key VARCHAR(120) NOT NULL UNIQUE,
            level INTEGER NOT NULL DEFAULT 0,
            title VARCHAR(400),
            summary TEXT,
            entity_ids INTEGER[],
            parent_community_id INTEGER REFERENCES kg_communities(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_kg_communities_level ON kg_communities(level);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS legal_kg_extraction_runs (
            id SERIAL PRIMARY KEY,
            document_key VARCHAR(120) NOT NULL,
            extractor_version VARCHAR(80) NOT NULL,
            entities_count INTEGER NOT NULL DEFAULT 0,
            relations_count INTEGER NOT NULL DEFAULT 0,
            citations_count INTEGER NOT NULL DEFAULT 0,
            status VARCHAR(30) NOT NULL DEFAULT 'success',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_legal_kg_extraction_runs_doc_created ON legal_kg_extraction_runs (document_key, created_at DESC);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS agent_case_workspace (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(120) NOT NULL,
            turn_id INT,
            facts_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            assumptions_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            open_questions_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            claim_verification_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            escalation_reason TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_case_workspace_session ON agent_case_workspace (session_id, created_at DESC);"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS legal_claim_verifications (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(120),
            turn_id INT,
            claim_text TEXT NOT NULL,
            support_score FLOAT NOT NULL DEFAULT 0.0,
            evidence_ref VARCHAR(200),
            status VARCHAR(30) NOT NULL DEFAULT 'review',
            verifier_version VARCHAR(80) NOT NULL DEFAULT 'legal-faithfulness-v1',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_legal_claim_verifications_session ON legal_claim_verifications (session_id, turn_id);"))


def _run_agent_routing_telemetry_migration(conn) -> None:
    conn.execute(text("ALTER TABLE agent_feedback_events ADD COLUMN IF NOT EXISTS intent VARCHAR(80);"))
    conn.execute(text("ALTER TABLE agent_feedback_events ADD COLUMN IF NOT EXISTS confidence FLOAT;"))
    conn.execute(text("ALTER TABLE agent_feedback_events ADD COLUMN IF NOT EXISTS correction_text TEXT;"))
    conn.execute(text("ALTER TABLE agent_feedback_events ADD COLUMN IF NOT EXISTS suggested_intent VARCHAR(80);"))
    conn.execute(text("ALTER TABLE agent_feedback_events ADD COLUMN IF NOT EXISTS metadata_json JSONB;"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS agent_route_events (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
            turn_id INTEGER REFERENCES agent_turns(id) ON DELETE SET NULL,
            dialogue_act VARCHAR(40) NOT NULL DEFAULT 'task',
            intent VARCHAR(80) NOT NULL,
            answer_contract VARCHAR(80) NOT NULL,
            model_mode VARCHAR(40) NOT NULL DEFAULT 'full',
            selected_tools_json JSONB,
            suppressed_tools_json JSONB,
            requested_domain VARCHAR(40),
            selected_model_bundle_json JSONB,
            mode_validation_json JSONB,
            mode_mismatch BOOLEAN NOT NULL DEFAULT FALSE,
            suggested_mode VARCHAR(40),
            suppressed_domains_json JSONB,
            route_confidence FLOAT,
            focus_score FLOAT,
            route_violation BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("ALTER TABLE agent_route_events ADD COLUMN IF NOT EXISTS requested_domain VARCHAR(40);"))
    conn.execute(text("ALTER TABLE agent_route_events ADD COLUMN IF NOT EXISTS selected_model_bundle_json JSONB;"))
    conn.execute(text("ALTER TABLE agent_route_events ADD COLUMN IF NOT EXISTS mode_validation_json JSONB;"))
    conn.execute(text("ALTER TABLE agent_route_events ADD COLUMN IF NOT EXISTS mode_mismatch BOOLEAN NOT NULL DEFAULT FALSE;"))
    conn.execute(text("ALTER TABLE agent_route_events ADD COLUMN IF NOT EXISTS suggested_mode VARCHAR(40);"))
    conn.execute(text("ALTER TABLE agent_route_events ADD COLUMN IF NOT EXISTS suppressed_domains_json JSONB;"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_route_events_created ON agent_route_events (created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_route_events_contract ON agent_route_events (answer_contract, created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_route_events_domain ON agent_route_events (requested_domain, created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_route_events_mismatch ON agent_route_events (mode_mismatch, created_at DESC);"))


def _run_agent_session_resilience_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS agent_session_snapshots (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(120) NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
            scope VARCHAR(60) NOT NULL,
            payload_json JSONB,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMPTZ
        );
    """))
    conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_session_snapshot_scope
        ON agent_session_snapshots (session_id, scope);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_agent_session_snapshots_exp
        ON agent_session_snapshots (expires_at);
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS agent_async_file_jobs (
            id SERIAL PRIMARY KEY,
            job_id VARCHAR(64) NOT NULL UNIQUE,
            session_id VARCHAR(120) REFERENCES agent_sessions(session_id) ON DELETE SET NULL,
            filename VARCHAR(500),
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            phase VARCHAR(80),
            progress FLOAT NOT NULL DEFAULT 0.0,
            error_message TEXT,
            response_json JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_agent_async_file_jobs_status
        ON agent_async_file_jobs (status, updated_at DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_agent_async_file_jobs_session
        ON agent_async_file_jobs (session_id, updated_at DESC);
    """))
    ensure_agent_resilience_schema(conn)


def _run_macro_map_state_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_data_sources (
            id BIGSERIAL PRIMARY KEY,
            source_key VARCHAR(120) NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            source_url TEXT,
            source_type VARCHAR(60) NOT NULL DEFAULT 'official',
            observed_level VARCHAR(40) NOT NULL DEFAULT 'national',
            review_status VARCHAR(30) NOT NULL DEFAULT 'approved',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_timeseries_observations (
            id BIGSERIAL PRIMARY KEY,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            province_code VARCHAR(20),
            indicator_key VARCHAR(100) NOT NULL,
            indicator_label TEXT,
            year INTEGER NOT NULL,
            quarter INTEGER NOT NULL DEFAULT 0,
            value_num DOUBLE PRECISION NOT NULL,
            unit VARCHAR(80),
            is_observed BOOLEAN NOT NULL DEFAULT TRUE,
            source_key VARCHAR(120) REFERENCES macro_data_sources(source_key) ON DELETE SET NULL,
            source_quality VARCHAR(40) NOT NULL DEFAULT 'official',
            provenance_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (boundary_version, province_code, indicator_key, year, quarter, source_key)
        );
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_macro_timeseries_province_year_indicator
        ON macro_timeseries_observations (province_code, year, indicator_key);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_macro_timeseries_source_observed
        ON macro_timeseries_observations (source_key, observed_at DESC);
    """))
    conn.execute(text("ALTER TABLE macro_timeseries_observations ALTER COLUMN quarter SET DEFAULT 0;"))
    conn.execute(text("UPDATE macro_timeseries_observations SET quarter = 0 WHERE quarter IS NULL;"))
    conn.execute(text("ALTER TABLE macro_timeseries_observations ALTER COLUMN quarter SET NOT NULL;"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_policy_scenarios (
            id BIGSERIAL PRIMARY KEY,
            scenario_key VARCHAR(120) NOT NULL UNIQUE,
            scenario_name TEXT NOT NULL,
            scenario_type VARCHAR(60) NOT NULL DEFAULT 'tax_policy',
            parameters_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            impact_hints JSONB NOT NULL DEFAULT '{}'::jsonb,
            review_status VARCHAR(30) NOT NULL DEFAULT 'approved',
            source_key VARCHAR(120) REFERENCES macro_data_sources(source_key) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_model_training_runs (
            run_id VARCHAR(100) PRIMARY KEY,
            model_key VARCHAR(120) NOT NULL,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            data_fingerprint VARCHAR(64),
            source_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            artifacts_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            review_policy TEXT NOT NULL DEFAULT 'approved_sources_only',
            status VARCHAR(30) NOT NULL DEFAULT 'completed',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS province_scenario_runs (
            id BIGSERIAL PRIMARY KEY,
            province_code VARCHAR(20) NOT NULL,
            event_key VARCHAR(80),
            gdp_delta_pct DOUBLE PRECISION DEFAULT 0.0,
            tax_rate_delta DOUBLE PRECISION DEFAULT 0.0,
            compliance_delta DOUBLE PRECISION DEFAULT 0.0,
            custom_params JSONB DEFAULT '{}'::jsonb,
            scenario_title TEXT,
            narrative_text TEXT,
            projected_revenue_billion DOUBLE PRECISION,
            projected_risk_level VARCHAR(20),
            metrics_json JSONB DEFAULT '{}'::jsonb,
            model_version VARCHAR(80) DEFAULT 'macro_scenario_v1',
            narrative_model VARCHAR(80),
            generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("ALTER TABLE province_scenario_runs DROP CONSTRAINT IF EXISTS province_scenario_runs_province_code_fkey;"))
    conn.execute(text("ALTER TABLE province_scenario_runs ADD COLUMN IF NOT EXISTS boundary_version VARCHAR(80) DEFAULT 'vn_34_2025';"))
    conn.execute(text("ALTER TABLE province_scenario_runs ADD COLUMN IF NOT EXISTS unemployment_delta DOUBLE PRECISION DEFAULT 0.0;"))
    conn.execute(text("ALTER TABLE province_scenario_runs ADD COLUMN IF NOT EXISTS fdi_delta_pct DOUBLE PRECISION DEFAULT 0.0;"))
    conn.execute(text("ALTER TABLE province_scenario_runs ADD COLUMN IF NOT EXISTS projection_years INTEGER DEFAULT 5;"))
    conn.execute(text("ALTER TABLE province_scenario_runs ADD COLUMN IF NOT EXISTS national_impacts JSONB NOT NULL DEFAULT '{}'::jsonb;"))
    conn.execute(text("ALTER TABLE province_scenario_runs ADD COLUMN IF NOT EXISTS province_impacts JSONB NOT NULL DEFAULT '[]'::jsonb;"))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_province_scenario_boundary_ts
        ON province_scenario_runs (boundary_version, generated_at DESC);
    """))


def _run_macro_research_lab_migration(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_event_articles (
            id BIGSERIAL PRIMARY KEY,
            event_key VARCHAR(120),
            title TEXT NOT NULL,
            source_url TEXT,
            source_name TEXT,
            published_at TIMESTAMPTZ,
            province_code VARCHAR(20),
            article_text TEXT,
            extracted_signals JSONB NOT NULL DEFAULT '{}'::jsonb,
            review_status VARCHAR(30) NOT NULL DEFAULT 'pending_review',
            data_fingerprint VARCHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_event_embeddings (
            id BIGSERIAL PRIMARY KEY,
            article_id BIGINT REFERENCES macro_event_articles(id) ON DELETE CASCADE,
            model_key VARCHAR(120) NOT NULL DEFAULT 'text-embedding-tax-macro-v1',
            embedding_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_province_panel (
            id BIGSERIAL PRIMARY KEY,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            province_code VARCHAR(20) NOT NULL,
            year INTEGER NOT NULL,
            quarter INTEGER NOT NULL DEFAULT 0,
            indicator_key VARCHAR(100) NOT NULL,
            value_num DOUBLE PRECISION NOT NULL,
            unit VARCHAR(80),
            source_key VARCHAR(120),
            review_status VARCHAR(30) NOT NULL DEFAULT 'pending_review',
            observed_level VARCHAR(40) NOT NULL DEFAULT 'province_estimate',
            provenance_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            data_fingerprint VARCHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (boundary_version, province_code, year, quarter, indicator_key, source_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_graph_edges (
            id BIGSERIAL PRIMARY KEY,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            source_code VARCHAR(20) NOT NULL,
            target_code VARCHAR(20) NOT NULL,
            edge_type VARCHAR(60) NOT NULL DEFAULT 'economic_similarity',
            weight DOUBLE PRECISION NOT NULL,
            evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            review_status VARCHAR(30) NOT NULL DEFAULT 'approved',
            data_fingerprint VARCHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (boundary_version, source_code, target_code, edge_type)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_forecast_runs (
            run_id VARCHAR(100) PRIMARY KEY,
            model_key VARCHAR(120) NOT NULL,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            province_code VARCHAR(20),
            horizon_quarters INTEGER NOT NULL,
            scenario_params JSONB NOT NULL DEFAULT '{}'::jsonb,
            forecasts JSONB NOT NULL DEFAULT '[]'::jsonb,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            data_fingerprint VARCHAR(64),
            review_policy TEXT NOT NULL DEFAULT 'approved_sources_only',
            status VARCHAR(30) NOT NULL DEFAULT 'completed',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_shock_runs (
            run_id VARCHAR(100) PRIMARY KEY,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            source_province_code VARCHAR(20) NOT NULL,
            shock_type VARCHAR(80) NOT NULL,
            shock_strength_pct DOUBLE PRECISION NOT NULL,
            horizon_quarters INTEGER NOT NULL,
            timeline_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            edge_paths_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            data_fingerprint VARCHAR(64),
            model_version VARCHAR(80) DEFAULT 'spatio-temporal-shock-v1',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_causal_runs (
            run_id VARCHAR(100) PRIMARY KEY,
            boundary_version VARCHAR(80) NOT NULL DEFAULT 'vn_34_2025',
            province_code VARCHAR(20) NOT NULL,
            treatment_key VARCHAR(120) NOT NULL,
            method VARCHAR(80) NOT NULL,
            actual_series JSONB NOT NULL DEFAULT '[]'::jsonb,
            counterfactual_series JSONB NOT NULL DEFAULT '[]'::jsonb,
            treatment_effects JSONB NOT NULL DEFAULT '[]'::jsonb,
            placebo_tests JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            data_fingerprint VARCHAR(64),
            status VARCHAR(30) NOT NULL DEFAULT 'completed',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS macro_model_cards (
            model_key VARCHAR(120) PRIMARY KEY,
            model_version VARCHAR(80) NOT NULL,
            model_family VARCHAR(80) NOT NULL,
            training_data_policy TEXT NOT NULL,
            intended_use TEXT NOT NULL,
            limitations TEXT NOT NULL,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            sources_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            artifact_paths_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    for index_sql in [
        "CREATE INDEX IF NOT EXISTS idx_macro_event_articles_status ON macro_event_articles (review_status, created_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_macro_province_panel_lookup ON macro_province_panel (boundary_version, province_code, indicator_key, year, quarter);",
        "CREATE INDEX IF NOT EXISTS idx_macro_graph_edges_source ON macro_graph_edges (boundary_version, source_code, weight DESC);",
        "CREATE INDEX IF NOT EXISTS idx_macro_forecast_runs_created ON macro_forecast_runs (model_key, created_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_macro_causal_runs_created ON macro_causal_runs (province_code, created_at DESC);",
    ]:
        conn.execute(text(index_sql))
    conn.execute(text("""
        INSERT INTO macro_model_cards (
            model_key, model_version, model_family, training_data_policy,
            intended_use, limitations, metrics_json, sources_json, artifact_paths_json
        ) VALUES
        (
            'macro-ensemble-v2',
            'macro-research-lab-v1',
            'Hybrid macro-fiscal forecasting',
            'Only approved/reviewed macro panel and economic event sources may be used for production training.',
            'Multi-horizon GRDP, tax revenue and fiscal-pressure forecasting with uncertainty intervals.',
            'Province panel currently mixes observed national series with reviewed baseline-anchored province estimates.',
            '{"baseline": "elasticity_lgbm", "interval_target_coverage": "85-95%"}'::jsonb,
            '["World Bank", "IMF DataMapper", "GSO/NSO", "reviewed macro event queue"]'::jsonb,
            '["Backend/data/models/simulation_lgbm.joblib", "Backend/data/models/simulation_config.json"]'::jsonb
        ),
        (
            'macro-shock-graph-v1',
            'macro-research-lab-v1',
            'Spatio-temporal graph shock propagation',
            'Graph edges must carry reviewed evidence or deterministic reproducible derivation metadata.',
            'Estimate spatial diffusion of macro shocks between provinces and administrative units.',
            'Fallback implementation is deterministic diffusion until STGCN/TFT artifacts are trained.',
            '{"graph_contract": "province adjacency + economic similarity + logistics/FDI similarity"}'::jsonb,
            '["province boundary manifest", "macro province panel", "reviewed event articles"]'::jsonb,
            '[]'::jsonb
        ),
        (
            'macro-causal-merger-v1',
            'macro-research-lab-v1',
            'Causal merger and tax-policy evaluation',
            'Causal claims require reviewed observed data, treatment metadata and placebo checks.',
            'Compare actual outcomes with synthetic-control counterfactuals for merger/policy questions.',
            'Current fallback is synthetic-control proxy and event-study style diagnostics.',
            '{"methods": ["synthetic_control_proxy", "event_study", "placebo_tests"]}'::jsonb,
            '["macro province panel", "merger mapping", "approved policy events"]'::jsonb,
            '[]'::jsonb
        )
        ON CONFLICT (model_key) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            model_family = EXCLUDED.model_family,
            training_data_policy = EXCLUDED.training_data_policy,
            intended_use = EXCLUDED.intended_use,
            limitations = EXCLUDED.limitations,
            metrics_json = EXCLUDED.metrics_json,
            sources_json = EXCLUDED.sources_json,
            artifact_paths_json = EXCLUDED.artifact_paths_json,
            updated_at = NOW();
    """))


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
        _run_legal_agent_v2_migration(conn)
        _run_agent_routing_telemetry_migration(conn)
        _run_agent_session_resilience_migration(conn)
        _run_macro_map_state_migration(conn)
        _run_macro_research_lab_migration(conn)

    print("[OK] Completed migration: user profile columns + offshore proxy mapping + numeric tax_code contract + multimodal uploads + legal agent v2 + agent routing telemetry.")


if __name__ == "__main__":
    run_migration()
