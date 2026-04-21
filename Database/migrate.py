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


def run_migration() -> None:
    with engine.begin() as conn:
        _run_safe_user_profile_migration(conn)
        _run_offshore_proxy_migration(conn)
        _run_numeric_tax_code_contract_migration(conn)

    print("[OK] Completed migration: user profile columns + offshore proxy mapping + numeric tax_code contract.")


if __name__ == "__main__":
    run_migration()
