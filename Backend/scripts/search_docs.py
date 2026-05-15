from app.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
all_docs = db.execute(text("SELECT id, title, doc_type FROM knowledge_documents")).fetchall()
print("Total docs:", len(all_docs))
cv = [d for d in all_docs if 'công văn' in str(d.title).lower() or d.doc_type == 'official_letter']
print("Cong van count:", len(cv))
for d in cv:
    print("-", d.title)
