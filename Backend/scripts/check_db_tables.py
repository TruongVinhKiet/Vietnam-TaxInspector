from app.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
tables = db.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")).fetchall()
print("TABLES:", [t[0] for t in tables])

# Let's also check kg_documents if it exists
if "kg_documents" in [t[0] for t in tables]:
    docs = db.execute(text("SELECT id, document_key, title, doc_type FROM kg_documents LIMIT 50")).fetchall()
    print("\nDOCUMENTS:")
    for doc in docs:
        print(doc)
