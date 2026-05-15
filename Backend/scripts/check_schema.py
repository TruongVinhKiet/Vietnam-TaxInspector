from app.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
schema = db.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'knowledge_documents'")).fetchall()
for c in schema:
    print(c)

schema_chunks = db.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'knowledge_chunks'")).fetchall()
print("\nCHUNKS:")
for c in schema_chunks:
    print(c)
