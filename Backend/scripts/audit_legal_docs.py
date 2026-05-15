from app.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
# ALL existing documents
docs = db.execute(text("SELECT id, document_key, title, doc_type FROM knowledge_documents ORDER BY id")).fetchall()
print(f"Total documents: {len(docs)}\n")
for d in docs:
    print(f"  [{d.id}] ({d.doc_type}) {d.document_key} -> {d.title}")

# ALL existing KG entities
entities = db.execute(text("SELECT id, entity_key, entity_type, display_name FROM kg_entities ORDER BY id")).fetchall()
print(f"\nTotal KG entities: {len(entities)}\n")
for e in entities:
    print(f"  [{e.id}] ({e.entity_type}) {e.entity_key} -> {e.display_name}")

# Count chunks
chunks = db.execute(text("SELECT COUNT(*) FROM knowledge_chunks")).scalar()
print(f"\nTotal chunks: {chunks}")
