import sys
sys.path.insert(0, 'e:\\TaxInspector')

from sqlalchemy import text
from Backend.app.database import engine

def run_migration():
    with engine.connect() as conn:
        # Add signature columns to users table
        for col_sql in [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS signature_data TEXT;",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS signature_verified BOOLEAN DEFAULT FALSE;",
        ]:
            conn.execute(text(col_sql))
        conn.commit()
        print("[OK] signature_data and signature_verified columns added to users table.")

if __name__ == "__main__":
    run_migration()
