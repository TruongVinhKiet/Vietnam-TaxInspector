from __future__ import annotations

from sqlalchemy import text

from app.database import SessionLocal
from app.scripts.ingest_tax_knowledge import ingest_document


def _ensure_knowledge_schema() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id SERIAL PRIMARY KEY,
            document_key VARCHAR(120) NOT NULL UNIQUE,
            title VARCHAR(400) NOT NULL,
            doc_type VARCHAR(80) NOT NULL,
            authority VARCHAR(200),
            language_code VARCHAR(10) NOT NULL DEFAULT 'vi',
            effective_from DATE,
            effective_to DATE,
            status VARCHAR(30) NOT NULL DEFAULT 'active',
            source_uri VARCHAR(500),
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_document_versions (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
            version_tag VARCHAR(80) NOT NULL,
            content_hash VARCHAR(64),
            raw_text TEXT,
            parsed_json JSONB,
            ingestion_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_doc_versions_doc_tag ON knowledge_document_versions (document_id, version_tag)",
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id SERIAL PRIMARY KEY,
            version_id INTEGER NOT NULL REFERENCES knowledge_document_versions(id) ON DELETE CASCADE,
            chunk_key VARCHAR(120) NOT NULL UNIQUE,
            chunk_index INTEGER NOT NULL,
            heading VARCHAR(300),
            chunk_text TEXT NOT NULL,
            token_count INTEGER,
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_citations (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
            citation_key VARCHAR(140) NOT NULL UNIQUE,
            legal_reference VARCHAR(300),
            citation_text TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunk_embeddings (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER NOT NULL UNIQUE REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
            embedding_model VARCHAR(80) NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding_json JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    with SessionLocal() as db:
        for s in statements:
            db.execute(text(s))
        db.commit()


def main() -> None:
    # Minimal corpus to unblock hybrid retrieval evaluation.
    _ensure_knowledge_schema()
    ingest_document(
        document_key="kb_vat_basics",
        title="Tổng quan VAT (cơ bản)",
        doc_type="vat",
        authority="internal_seed",
        source_uri="seed://kb_vat_basics",
        version_tag="v1",
        content=(
            "Khái niệm VAT và đối tượng chịu thuế.\n\n"
            "Kê khai VAT theo tháng hoặc theo quý tùy quy mô doanh nghiệp và quy định hiện hành.\n\n"
            "Hoàn thuế VAT: hồ sơ hoàn cần đối chiếu hóa đơn, chứng từ thanh toán, và kiểm tra dấu hiệu bất thường.\n\n"
            "Điều kiện hóa đơn hợp lệ: thông tin người bán/người mua, ngày lập, giá trị, thuế suất và trạng thái thanh toán."
        ),
    )
    ingest_document(
        document_key="kb_collections_basics",
        title="Quản lý nợ đọng và thu nợ (cơ bản)",
        doc_type="collections",
        authority="internal_seed",
        source_uri="seed://kb_collections_basics",
        version_tag="v1",
        content=(
            "Nợ đọng thuế phát sinh khi người nộp thuế không nộp đúng hạn.\n\n"
            "Biện pháp: nhắc nợ, gọi điện, đối soát, cưỡng chế theo quy định.\n\n"
            "Dự báo nợ đọng cần lịch sử nộp thuế, số tiền đến hạn, và hành vi chậm nộp trước đó."
        ),
    )
    ingest_document(
        document_key="kb_transfer_pricing_basics",
        title="Chuyển giá và giao dịch liên kết (cơ bản)",
        doc_type="transfer_pricing",
        authority="internal_seed",
        source_uri="seed://kb_transfer_pricing_basics",
        version_tag="v1",
        content=(
            "Chuyển giá có thể thể hiện qua giá mua/bán giữa các bên liên kết lệch so với thị trường.\n\n"
            "Cần so sánh theo phương pháp phù hợp, xem xét HS code, thị trường, điều kiện giao hàng, và lợi nhuận.\n\n"
            "Hồ sơ xác định giá giao dịch liên kết cần được lưu và cung cấp khi cơ quan thuế yêu cầu."
        ),
    )
    print("[OK] minimal tax knowledge seeded.")


if __name__ == "__main__":
    main()

