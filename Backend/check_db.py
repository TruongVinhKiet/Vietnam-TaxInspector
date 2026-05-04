"""Run GraphRAG schema migration."""
import psycopg2

SQL = """
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
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_entities_status ON kg_entities(status);
CREATE INDEX IF NOT EXISTS idx_kg_entities_authority ON kg_entities(authority_rank DESC);

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
CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type);

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
CREATE INDEX IF NOT EXISTS idx_kg_communities_level ON kg_communities(level);
"""

conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='Kiet2004', dbname='TaxInspector')
cur = conn.cursor()
cur.execute(SQL)
conn.commit()
print("GraphRAG schema created successfully!")

for t in ['kg_entities', 'kg_relations', 'kg_communities']:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    print(f"  {t}: {cur.fetchone()[0]} rows")

conn.close()
