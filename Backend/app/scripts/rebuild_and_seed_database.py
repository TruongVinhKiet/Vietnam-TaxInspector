"""
rebuild_and_seed_database.py
======================================
1. Truncate existing graph tables (companies, invoices, ownership_links).
2. Apply init_db.sql migrations manually if needed.
3. Seed 10k+ instances utilizing generate_graph_mock_data.py's routines.
4. Load vietnam.json into an authoritative boundary table and use ST_Contains 
   to rigorously tag country_inferred and is_within_vietnam.
"""

import os, sys, json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import psycopg2
from psycopg2.extras import Json

# We can import generator directly to trigger the 10000+ data creation
from app.scripts.generate_graph_mock_data import (
    generate_companies,
    generate_normal_invoices,
    generate_circular_invoices,
    generate_hard_negatives,
    DATA_START, DATA_END
)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "TaxInspector")

def rebuild_and_seed():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    conn.autocommit = True
    cur = conn.cursor()

    print("[1] Rebuilding schema dependencies and cleaning tables...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    
    # Run the table additions just in case they were missing
    cur.execute("""
        ALTER TABLE companies ADD COLUMN IF NOT EXISTS country_inferred VARCHAR(100);
        ALTER TABLE companies ADD COLUMN IF NOT EXISTS confidence_country FLOAT DEFAULT 0.0;
        ALTER TABLE companies ADD COLUMN IF NOT EXISTS is_within_vietnam BOOLEAN;
        ALTER TABLE companies ADD COLUMN IF NOT EXISTS geocoding_method VARCHAR(50);
        ALTER TABLE companies ADD COLUMN IF NOT EXISTS geocoded_at TIMESTAMP;
        CREATE INDEX IF NOT EXISTS idx_companies_geom ON companies USING GIST(geom);
    """)

    # TRUNCATE ALL data! User requested completely replacing data!
    print("[1b] Truncating all graph data...")
    cur.execute("TRUNCATE TABLE companies CASCADE;")
    
    print("[2] Seeding 10,000+ companies and 50,000+ invoices...")
    # These functions use `conn` and assume `conn.commit()` is called, 
    # but we are in autocommit, so it works.
    companies = generate_companies(conn)
    generate_normal_invoices(conn, companies)
    generate_circular_invoices(conn, companies)
    generate_hard_negatives(conn, companies)

    print("[3] Loading Official Vietnam GeoJSON MultiPolygon Boundary...")
    json_path = BACKEND_DIR.parent / "Frontend" / "json" / "vietnam.json"
    if not json_path.exists():
        print(f"[ERROR] Cannot find {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    # In vietnam.json, it's a feature collection. We'll extract the geometry of the first feature, or iterate.
    # Often it's `{"type": "FeatureCollection", "features": ...}`
    features = geo_data.get("features", [])
    if not features:
        # Maybe it's just a raw Feature or array? Let's check structure if it fails.
        if "type" in geo_data and geo_data["type"] == "FeatureCollection":
            pass
        elif isinstance(geo_data, list) and geo_data and "geometry" in geo_data[0]:
            features = geo_data
        else:
            print("[WARN] Unknown GeoJSON format, attempting to parse raw...")

    cur.execute("DROP TABLE IF EXISTS _vietnam_boundary;")
    cur.execute("CREATE TABLE _vietnam_boundary (geom geometry(MultiPolygon, 4326));")
    
    for feat in features:
        geom = feat.get("geometry")
        if geom:
            cur.execute(
                "INSERT INTO _vietnam_boundary (geom) VALUES (ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
                (json.dumps(geom),)
            )

    print("[4] Executing ST_Contains mapping over all 10,000+ entities...")
    
    # 4a. Update Vietnam entities
    cur.execute("""
        UPDATE companies
        SET is_within_vietnam = TRUE,
            country_inferred = 'Vietnam',
            confidence_country = 1.0,
            geocoding_method = 'postgis_st_contains',
            geocoded_at = CURRENT_TIMESTAMP
        FROM _vietnam_boundary v
        WHERE ST_Contains(v.geom, companies.geom);
    """)
    vn_count = cur.rowcount
    
    # 4b. Update Foreign / Offshore entities
    cur.execute("""
        UPDATE companies
        SET is_within_vietnam = FALSE,
            country_inferred = CASE 
                WHEN ST_DistanceSphere(geom, ST_SetSRID(ST_MakePoint(103.8198, 1.3521), 4326)) < 100000 THEN 'Singapore'
                WHEN ST_DistanceSphere(geom, ST_SetSRID(ST_MakePoint(-81.2546, 19.3133), 4326)) < 100000 THEN 'Cayman Islands'
                WHEN ST_DistanceSphere(geom, ST_SetSRID(ST_MakePoint(-64.6400, 18.4207), 4326)) < 100000 THEN 'British Virgin Islands'
                ELSE 'Unknown Offshore'
            END,
            confidence_country = 0.95,
            geocoding_method = 'postgis_nearest_offshore',
            geocoded_at = CURRENT_TIMESTAMP
        WHERE is_within_vietnam IS NULL AND geom IS NOT NULL;
    """)
    offshore_count = cur.rowcount

    print("[5] Normalizing offshore attributes for downstream OSINT queries...")
    cur.execute("""
        UPDATE companies
        SET
            industry = 'Offshore Entity',
            province = COALESCE(NULLIF(country_inferred, ''), province, 'Unknown Offshore')
        WHERE is_within_vietnam = FALSE;
    """)
    offshore_industry_tagged = cur.rowcount

    cur.execute("""
        UPDATE companies
        SET
            country_inferred = COALESCE(NULLIF(country_inferred, ''), province, 'Unknown Offshore'),
            province = COALESCE(NULLIF(country_inferred, ''), province, 'Unknown Offshore')
        WHERE is_within_vietnam = FALSE;
    """)

    cur.execute("""
        UPDATE companies
        SET
            country_inferred = 'Vietnam',
            province = COALESCE(NULLIF(province, ''), 'Vietnam')
        WHERE is_within_vietnam = TRUE;
    """)

    # Keep offshore_entities in sync so /api/osint/high-risk-ubo primary path never stays empty.
    cur.execute("TRUNCATE TABLE offshore_entities RESTART IDENTITY CASCADE;")
    cur.execute("""
        INSERT INTO offshore_entities (
            entity_code,
            proxy_tax_code,
            name,
            country,
            jurisdiction_risk_weight,
            risk_score,
            entity_type,
            registration_date,
            status,
            data_source
        )
        SELECT
            CONCAT('OSF-', c.tax_code) AS entity_code,
            c.tax_code AS proxy_tax_code,
            c.name,
            COALESCE(NULLIF(c.country_inferred, ''), NULLIF(c.province, ''), 'Unknown Offshore') AS country,
            CASE
                WHEN LOWER(COALESCE(c.country_inferred, c.province, '')) IN ('cayman islands', 'british virgin islands', 'panama', 'seychelles')
                    THEN 0.9
                WHEN LOWER(COALESCE(c.country_inferred, c.province, '')) IN ('singapore', 'hong kong')
                    THEN 0.7
                ELSE 0.6
            END AS jurisdiction_risk_weight,
            GREATEST(COALESCE(c.risk_score, 50.0), 60.0) AS risk_score,
            'shell_company' AS entity_type,
            c.registration_date,
            CASE WHEN c.is_active IS FALSE THEN 'inactive' ELSE 'active' END AS status,
            'rebuild_and_seed_database' AS data_source
        FROM companies c
        WHERE c.is_within_vietnam = FALSE
          AND c.tax_code ~ '^[0-9]{10}$'
        ON CONFLICT (entity_code) DO UPDATE
        SET
            proxy_tax_code = EXCLUDED.proxy_tax_code,
            name = EXCLUDED.name,
            country = EXCLUDED.country,
            jurisdiction_risk_weight = EXCLUDED.jurisdiction_risk_weight,
            risk_score = EXCLUDED.risk_score,
            entity_type = EXCLUDED.entity_type,
            registration_date = EXCLUDED.registration_date,
            status = EXCLUDED.status,
            data_source = EXCLUDED.data_source;
    """)
    offshore_entities_seeded = cur.rowcount

    # Cleanup temporary table
    cur.execute("DROP TABLE _vietnam_boundary;")

    cur.execute("SELECT COUNT(*) FROM companies;")
    total_companies = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM invoices;")
    total_invoices = cur.fetchone()[0]

    print("=" * 60)
    print(f"DATABASE RESEEDED SUCCESSFULLY")
    print(f"Total Companies: {total_companies}")
    print(f"Total Invoices: {total_invoices}")
    print(f"Geo-MAPPED to Vietnam: {vn_count}")
    print(f"Geo-MAPPED OFFSHORE: {offshore_count}")
    print(f"Offshore industry normalized: {offshore_industry_tagged}")
    print(f"offshore_entities seeded: {offshore_entities_seeded}")
    print("=" * 60)

if __name__ == "__main__":
    rebuild_and_seed()
