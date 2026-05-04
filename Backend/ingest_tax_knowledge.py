"""
ingest_tax_knowledge.py - Ingestion Core for GraphRAG Knowledge Graph
======================================================================
Nạp corpus pháp luật thuế VN vào knowledge_documents, knowledge_chunks,
kg_entities, kg_relations.
"""
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import psycopg2

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_legal_intelligence import authority_rank, canonical_doc_type

logger = logging.getLogger(__name__)


def _tokenize(text_value: str) -> list[str]:
    return [
        tok
        for tok in re.split(r"[^a-zA-Z0-9_]+", (text_value or "").lower())
        if tok
    ]


def _fallback_embedding(text_value: str, dim: int = 384) -> list[float]:
    vec = [0.0] * dim
    toks = _tokenize(text_value)
    if not toks:
        return vec
    for tok in toks:
        digest = hashlib.sha256(tok.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [round(v / norm, 8) for v in vec] if norm else vec


def _embed_for_ingestion(text_value: str) -> list[float]:
    try:
        from ml_engine.tax_agent_embeddings import get_embedding_engine

        embedding = get_embedding_engine().embed_passage(text_value)
        return [float(v) for v in embedding.vector]
    except Exception as exc:
        logger.debug("Embedding engine unavailable, using deterministic fallback: %s", exc)
        return _fallback_embedding(text_value)

def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, user='postgres',
        password='Kiet2004', dbname='TaxInspector'
    )

class TaxKnowledgeIngestor:
    def __init__(self):
        self.conn = get_conn()
        self.cur = self.conn.cursor()
        self._pending_rels = []

    def ingest(self, doc):
        """
        doc = {
            'key': 'LUAT_GTGT_2008', 'title': '...', 'type': 'law',
            'authority_rank': 90, 'content': '...', 'effective_from': '2009-01-01',
            'relations': [{'target': 'ND_209_2013', 'type': 'implements'}]
        }
        """
        print(f"  Ingesting: {doc['title'][:80]}")

        # 1. Document
        doc_type = canonical_doc_type(doc.get('type', 'legal_document'))
        doc_rank = int(doc.get('authority_rank') or authority_rank(doc_type))

        self.cur.execute("""
            INSERT INTO knowledge_documents (document_key, title, doc_type, status)
            VALUES (%s, %s, %s, 'active')
            ON CONFLICT (document_key) DO UPDATE SET title=EXCLUDED.title
            RETURNING id
        """, (doc['key'], doc['title'], doc_type))
        doc_id = self.cur.fetchone()[0]

        # 2. Version
        self.cur.execute("""
            INSERT INTO knowledge_document_versions (document_id, version_tag, raw_text)
            VALUES (%s, 'v1', %s)
            ON CONFLICT (document_id, version_tag) DO NOTHING
            RETURNING id
        """, (doc_id, doc['content']))
        r = self.cur.fetchone()
        if r:
            ver_id = r[0]
        else:
            self.cur.execute(
                "SELECT id FROM knowledge_document_versions WHERE document_id=%s AND version_tag='v1'",
                (doc_id,))
            ver_id = self.cur.fetchone()[0]

        # 3. Chunks (split by Điều)
        parts = re.split(r'(?=Điều \d+)', doc['content'])
        chunk_ids = []
        for i, txt in enumerate(parts):
            txt = txt.strip()
            if len(txt) < 15:
                continue
            ckey = f"{doc['key']}_CH{i}"
            hm = re.search(r'Điều \d+\.?\s*([^\n]+)', txt)
            heading = hm.group(0).strip() if hm else f"Đoạn {i}"
            self.cur.execute("""
                INSERT INTO knowledge_chunks (version_id, chunk_key, chunk_index, heading, chunk_text)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_key) DO UPDATE SET chunk_text=EXCLUDED.chunk_text
                RETURNING id
            """, (ver_id, ckey, i, heading[:300], txt))
            chunk_ids.append(self.cur.fetchone()[0])

        # 4. KG Entity. Production uses the shared embedding engine; fallback is deterministic.
        emb = _embed_for_ingestion(f"{doc['title']}\n{doc['content'][:1200]}")
        eff = doc.get('effective_from')
        self.cur.execute("""
            INSERT INTO kg_entities
                (entity_key, entity_type, display_name, description,
                 authority_rank, effective_from, status, chunk_ids, embedding_json)
            VALUES (%s,%s,%s,%s,%s,%s,'active',%s,%s)
            ON CONFLICT (entity_key) DO UPDATE
                SET chunk_ids=EXCLUDED.chunk_ids, embedding_json=EXCLUDED.embedding_json
            RETURNING id
        """, (doc['key'], doc_type, doc['title'],
              doc['content'][:500], doc_rank,
              eff, chunk_ids, json.dumps(emb)))
        eid = self.cur.fetchone()[0]

        # 5. Relations (deferred — target may not exist yet)
        for rel in doc.get('relations', []):
            self._pending_rels.append((doc['key'], rel['target'], rel['type'],
                                       rel.get('weight', 1.0)))

        self.conn.commit()
        return eid, chunk_ids

    def finalize_relations(self):
        """Create all edges after all entities are ingested."""
        created = 0
        for src_key, tgt_key, rtype, weight in self._pending_rels:
            self.cur.execute("SELECT id FROM kg_entities WHERE entity_key=%s", (src_key,))
            sr = self.cur.fetchone()
            self.cur.execute("SELECT id FROM kg_entities WHERE entity_key=%s", (tgt_key,))
            tr = self.cur.fetchone()
            if sr and tr:
                self.cur.execute("""
                    INSERT INTO kg_relations (source_entity_id, target_entity_id, relation_type, weight)
                    VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING
                """, (sr[0], tr[0], rtype, weight))
                created += 1
        self.conn.commit()
        print(f"  Created {created} relations")

    def close(self):
        self.cur.close()
        self.conn.close()
