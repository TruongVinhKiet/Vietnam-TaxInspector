import sys
sys.path.insert(0, 'e:\\TaxInspector')

from sqlalchemy import text
from Backend.app.database import engine

def run_migration():
    with engine.connect() as conn:
        # Add additive user-profile columns (safe to run multiple times)
        for col_sql in [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS signature_data TEXT;",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS signature_verified BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_data TEXT;",
        ]:
            conn.execute(text(col_sql))
        conn.commit()
        print("[OK] signature_data/signature_verified/avatar_data columns verified on users table.")

if __name__ == "__main__":
    run_migration()
