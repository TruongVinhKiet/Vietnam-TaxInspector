from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService
from ml_engine.tax_agent_legal_intelligence import LegalKnowledgeExtractor, authority_rank, canonical_doc_type


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tokenize(text_value: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", (text_value or "").lower()) if tok]


def _embed(text_value: str, dim: int = 96) -> list[float]:
    vec = [0.0] * dim
    toks = _tokenize(text_value)
    if not toks:
        return vec
    for tok in toks:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _embed_for_ingestion(text_value: str) -> tuple[list[float], str]:
    """Use the production embedding engine, with hash-TFIDF as deterministic fallback."""
    try:
        from ml_engine.tax_agent_embeddings import get_embedding_engine

        engine = get_embedding_engine()
        result = engine.embed_passage(text_value)
        return [float(v) for v in result.vector], str(result.model_tier)
    except Exception:
        emb = _embed(text_value)
        return emb, "hash-tfidf-v1"


def _split_chunks(text_value: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in (text_value or "").split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
            current = ""
    if current:
        chunks.append(current)
    if not chunks and text_value.strip():
        chunks.append(text_value.strip()[:max_chars])
    return chunks


def ingest_document(
    *,
    document_key: str,
    title: str,
    doc_type: str,
    authority: str,
    source_uri: str,
    version_tag: str,
    content: str,
    language_code: str = "vi",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with SessionLocal() as db:
        registry = ModelRegistryService(db)
        
        merged_metadata = {"ingested_by": "ingest_tax_knowledge"}
        if metadata:
            merged_metadata.update(metadata)
            
        row = db.execute(
            text("SELECT id FROM knowledge_documents WHERE document_key = :document_key"),
            {"document_key": document_key},
        ).fetchone()
        if row:
            document_id = int(row[0])
            db.execute(
                text(
                    """
                    UPDATE knowledge_documents
                    SET title = :title,
                        doc_type = :doc_type,
                        authority = :authority,
                        source_uri = :source_uri,
                        language_code = :language_code,
                        metadata_json = CAST(:metadata_json AS jsonb),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :document_id
                    """
                ),
                {
                    "document_id": document_id,
                    "title": title,
                    "doc_type": doc_type,
                    "authority": authority or None,
                    "source_uri": source_uri or None,
                    "language_code": language_code,
                    "metadata_json": json.dumps(merged_metadata),
                },
            )
        else:
            inserted = db.execute(
                text(
                    """
                    INSERT INTO knowledge_documents (document_key, title, doc_type, authority, language_code, source_uri, metadata_json)
                    VALUES (:document_key, :title, :doc_type, :authority, :language_code, :source_uri, CAST(:metadata_json AS jsonb))
                    RETURNING id
                    """
                ),
                {
                    "document_key": document_key,
                    "title": title,
                    "doc_type": doc_type,
                    "authority": authority or None,
                    "language_code": language_code,
                    "source_uri": source_uri or None,
                    "metadata_json": json.dumps(merged_metadata),
                },
            ).fetchone()
            document_id = int(inserted[0])

        version_row = db.execute(
            text(
                "SELECT id FROM knowledge_document_versions WHERE document_id = :document_id AND version_tag = :version_tag"
            ),
            {"document_id": document_id, "version_tag": version_tag},
        ).fetchone()
        content_hash = _sha({"content": content})
        if version_row:
            version_id = int(version_row[0])
            db.execute(
                text(
                    """
                    UPDATE knowledge_document_versions
                    SET content_hash = :content_hash, raw_text = :raw_text, parsed_json = CAST(:parsed_json AS jsonb)
                    WHERE id = :version_id
                    """
                ),
                {
                    "version_id": version_id,
                    "content_hash": content_hash,
                    "raw_text": content,
                    "parsed_json": json.dumps({"token_count_estimate": len(_tokenize(content))}),
                },
            )
            db.execute(text("DELETE FROM knowledge_chunk_embeddings WHERE chunk_id IN (SELECT id FROM knowledge_chunks WHERE version_id = :version_id)"), {"version_id": version_id})
            db.execute(text("DELETE FROM knowledge_citations WHERE chunk_id IN (SELECT id FROM knowledge_chunks WHERE version_id = :version_id)"), {"version_id": version_id})
            db.execute(text("DELETE FROM knowledge_chunks WHERE version_id = :version_id"), {"version_id": version_id})
        else:
            inserted = db.execute(
                text(
                    """
                    INSERT INTO knowledge_document_versions (document_id, version_tag, content_hash, raw_text, parsed_json)
                    VALUES (:document_id, :version_tag, :content_hash, :raw_text, CAST(:parsed_json AS jsonb))
                    RETURNING id
                    """
                ),
                {
                    "document_id": document_id,
                    "version_tag": version_tag,
                    "content_hash": content_hash,
                    "raw_text": content,
                    "parsed_json": json.dumps({"token_count_estimate": len(_tokenize(content))}),
                },
            ).fetchone()
            version_id = int(inserted[0])

        chunks = _split_chunks(content)
        chunk_payloads: list[dict[str, Any]] = []
        citation_inserted = 0
        for idx, chunk_text in enumerate(chunks):
            chunk_key = f"{document_key}:{version_tag}:{idx}"
            chunk_row = db.execute(
                text(
                    """
                    INSERT INTO knowledge_chunks (version_id, chunk_key, chunk_index, heading, chunk_text, token_count, metadata_json)
                    VALUES (:version_id, :chunk_key, :chunk_index, :heading, :chunk_text, :token_count, CAST(:metadata_json AS jsonb))
                    RETURNING id
                    """
                ),
                {
                    "version_id": version_id,
                    "chunk_key": chunk_key,
                    "chunk_index": idx,
                    "heading": f"Section {idx + 1}",
                    "chunk_text": chunk_text,
                    "token_count": len(_tokenize(chunk_text)),
                    "metadata_json": json.dumps({"document_key": document_key, "version_tag": version_tag}),
                },
            ).fetchone()
            chunk_id = int(chunk_row[0])
            chunk_payloads.append({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "heading": f"Section {idx + 1}",
                "text": chunk_text,
            })
            emb, embedding_model = _embed_for_ingestion(chunk_text)
            embedding_params = {
                "chunk_id": chunk_id,
                "embedding_model": embedding_model,
                "embedding_dim": len(emb),
                "embedding_json": json.dumps(emb),
                "embedding_vector": _vector_literal(emb) if len(emb) == 384 else None,
                "embedding_hash_vector": _vector_literal(emb) if len(emb) == 96 else None,
                "content_hash": _sha({"chunk_text": chunk_text}),
            }
            nested = db.begin_nested()
            try:
                db.execute(
                    text(
                        """
                        INSERT INTO knowledge_chunk_embeddings
                        (chunk_id, embedding_model, embedding_dim, embedding_json,
                         embedding_vector, embedding_hash_vector, embedding_source, content_hash, indexed_at)
                        VALUES
                        (:chunk_id, :embedding_model, :embedding_dim, CAST(:embedding_json AS jsonb),
                         CAST(:embedding_vector AS vector(384)), CAST(:embedding_hash_vector AS vector(96)),
                         'ingestion', :content_hash, CURRENT_TIMESTAMP)
                        """
                    ),
                    embedding_params,
                )
                nested.commit()
            except Exception:
                nested.rollback()
                db.execute(
                    text(
                        """
                        INSERT INTO knowledge_chunk_embeddings
                        (chunk_id, embedding_model, embedding_dim, embedding_json, embedding_source, content_hash)
                        VALUES
                        (:chunk_id, :embedding_model, :embedding_dim, CAST(:embedding_json AS jsonb),
                         'ingestion', :content_hash)
                        """
                    ),
                    embedding_params,
                )
            if "điều" in chunk_text.lower() or "article" in chunk_text.lower():
                citation_key = f"{chunk_key}:cite"
                db.execute(
                    text(
                        """
                        INSERT INTO knowledge_citations (chunk_id, citation_key, legal_reference, citation_text, confidence)
                        VALUES (:chunk_id, :citation_key, :legal_reference, :citation_text, :confidence)
                        """
                    ),
                    {
                        "chunk_id": chunk_id,
                        "citation_key": citation_key,
                        "legal_reference": title,
                        "citation_text": chunk_text[:280],
                        "confidence": 0.72,
                    },
                )
                citation_inserted += 1

        kg_result = _upsert_legal_kg(
            db,
            document_key=document_key,
            title=title,
            doc_type=doc_type,
            authority=authority,
            source_uri=source_uri,
            content=content,
            chunk_payloads=chunk_payloads,
        )

        db.commit()
        dataset_version = f"knowledge-{document_key}-{version_tag}"
        registry.register_dataset_version(
            dataset_key="knowledge_corpus",
            dataset_version=dataset_version,
            entity_type="document",
            row_count=len(chunks),
            source_tables=["knowledge_documents", "knowledge_document_versions", "knowledge_chunks", "knowledge_chunk_embeddings"],
            filters={"document_key": document_key, "version_tag": version_tag},
            data_hash=content_hash,
            created_by="ingest_tax_knowledge",
        )
        return {
            "document_key": document_key,
            "version_tag": version_tag,
            "chunks": len(chunks),
            "citations": citation_inserted,
            "kg_entities": kg_result.get("entities", 0),
            "kg_relations": kg_result.get("relations", 0),
            "embedding_model": embedding_model if chunks else "none",
        }


def _upsert_legal_kg(
    db,
    *,
    document_key: str,
    title: str,
    doc_type: str,
    authority: str,
    source_uri: str,
    content: str,
    chunk_payloads: list[dict[str, Any]],
) -> dict[str, int]:
    """Extract and persist legal KG entities/relations for GraphRAG v2."""
    extractor = LegalKnowledgeExtractor()
    extraction = extractor.extract_document(
        document_key=document_key,
        title=title,
        doc_type=doc_type,
        authority=authority,
        content=content,
        chunks=chunk_payloads,
    )
    entity_id_by_key: dict[str, int] = {}
    for entity in extraction.entities:
        emb, embedding_model = _embed_for_ingestion(
            f"{entity.display_name}\n{entity.description}\n{entity.entity_type}"
        )
        row = db.execute(
            text(
                """
                INSERT INTO kg_entities
                    (entity_key, entity_type, display_name, description,
                     authority_rank, effective_from, effective_to, status,
                     chunk_ids, attributes_json, embedding_json)
                VALUES
                    (:entity_key, :entity_type, :display_name, :description,
                     :authority_rank, :effective_from, :effective_to, :status,
                     :chunk_ids, CAST(:attributes_json AS jsonb), CAST(:embedding_json AS jsonb))
                ON CONFLICT (entity_key) DO UPDATE SET
                    entity_type = EXCLUDED.entity_type,
                    display_name = EXCLUDED.display_name,
                    description = EXCLUDED.description,
                    authority_rank = EXCLUDED.authority_rank,
                    effective_from = COALESCE(EXCLUDED.effective_from, kg_entities.effective_from),
                    effective_to = COALESCE(EXCLUDED.effective_to, kg_entities.effective_to),
                    status = EXCLUDED.status,
                    chunk_ids = EXCLUDED.chunk_ids,
                    attributes_json = kg_entities.attributes_json || EXCLUDED.attributes_json,
                    embedding_json = EXCLUDED.embedding_json
                RETURNING id
                """
            ),
            {
                "entity_key": entity.entity_key,
                "entity_type": canonical_doc_type(entity.entity_type),
                "display_name": entity.display_name,
                "description": entity.description,
                "authority_rank": entity.authority_rank or authority_rank(entity.entity_type),
                "effective_from": entity.effective_from,
                "effective_to": entity.effective_to,
                "status": entity.status,
                "chunk_ids": entity.chunk_ids,
                "attributes_json": json.dumps({
                    **(entity.attributes or {}),
                    "embedding_model": embedding_model,
                    "source_uri": source_uri or None,
                    "kg_extractor_version": "legal-kg-v2",
                }, ensure_ascii=False, default=str),
                "embedding_json": json.dumps(emb),
            },
        ).fetchone()
        if row:
            entity_id_by_key[entity.entity_key] = int(row[0])

    relation_count = 0
    for relation in extraction.relations:
        source_id = entity_id_by_key.get(relation.source_key)
        target_id = entity_id_by_key.get(relation.target_key)
        if not source_id or not target_id:
            continue
        db.execute(
            text(
                """
                INSERT INTO kg_relations
                    (source_entity_id, target_entity_id, relation_type, weight,
                     confidence, evidence_text, attributes_json)
                VALUES
                    (:source_id, :target_id, :relation_type, :weight,
                     :confidence, :evidence_text, CAST(:attributes_json AS jsonb))
                ON CONFLICT (source_entity_id, target_entity_id, relation_type)
                DO UPDATE SET
                    weight = GREATEST(kg_relations.weight, EXCLUDED.weight),
                    confidence = GREATEST(kg_relations.confidence, EXCLUDED.confidence),
                    evidence_text = COALESCE(NULLIF(EXCLUDED.evidence_text, ''), kg_relations.evidence_text),
                    attributes_json = kg_relations.attributes_json || EXCLUDED.attributes_json
                """
            ),
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation.relation_type,
                "weight": relation.weight,
                "confidence": relation.confidence,
                "evidence_text": relation.evidence_text[:1000],
                "attributes_json": json.dumps(relation.attributes or {}, ensure_ascii=False, default=str),
            },
        )
        relation_count += 1

    db.execute(
        text(
            """
            INSERT INTO legal_kg_extraction_runs
                (document_key, extractor_version, entities_count, relations_count,
                 citations_count, status, metadata_json)
            VALUES
                (:document_key, 'legal-kg-v2', :entities_count, :relations_count,
                 :citations_count, 'success', CAST(:metadata_json AS jsonb))
            """
        ),
        {
            "document_key": document_key,
            "entities_count": len(extraction.entities),
            "relations_count": relation_count,
            "citations_count": len(extraction.citations),
            "metadata_json": json.dumps({"source_uri": source_uri}, ensure_ascii=False),
        },
    )
    return {"entities": len(extraction.entities), "relations": relation_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-key", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--doc-type", default="tax_law")
    parser.add_argument("--authority", default="")
    parser.add_argument("--source-uri", default="")
    parser.add_argument("--version-tag", default="v1")
    parser.add_argument("--content-file", required=True)
    args = parser.parse_args()

    content = Path(args.content_file).read_text(encoding="utf-8")
    result = ingest_document(
        document_key=args.document_key,
        title=args.title,
        doc_type=args.doc_type,
        authority=args.authority,
        source_uri=args.source_uri,
        version_tag=args.version_tag,
        content=content,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
