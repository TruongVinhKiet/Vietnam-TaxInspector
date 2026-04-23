import os
import sys
import random
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import psycopg2
from collections import defaultdict

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "TaxInspector")

def seed_ownership_networks():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE ownership_links CASCADE;")

    print("[DEBUG] Generating robust ownership data for 10000 entities...")

    # 1. Get all shell companies (prefix 99)
    cur.execute("SELECT tax_code FROM companies WHERE tax_code LIKE '99%';")
    shell_codes = [r[0] for r in cur.fetchall()]

    # 2. Get high value circular edges to form ownership rings
    cur.execute("""
        SELECT DISTINCT seller_tax_code, buyer_tax_code 
        FROM invoices;
    """)
    invoice_pairs = cur.fetchall()
    # Shuffle and sample because pulling all might be big
    random.shuffle(invoice_pairs)
    invoice_pairs = invoice_pairs[:3000]

    ownerships = []
    seen_links = set()

    def add_link(parent, child):
        if parent == child: return
        key = (parent, child)
        if key in seen_links: return
        seen_links.add(key)
        pct = round(random.uniform(51.0, 100.0), 2)
        rel_type = random.choice(["Owner", "Subsidiary", "RelatedParty"])
        person_id = "UBO_" + str(random.randint(1000, 9999))
        ownerships.append((parent, child, pct, rel_type, person_id, "Proxy Director"))

    # Pattern A: Shell companies share common UBOs (Creating clusters for offshore/shell)
    random.shuffle(shell_codes)
    for i in range(0, len(shell_codes), 5):
        cluster = shell_codes[i:i+5]
        if not cluster: continue
        parent = cluster[0]
        for child in cluster[1:]:
            add_link(parent, child)

    # Pattern C: "Bà con" (Relatives) Shadow Edge Generation
    # We create clusters of "Relatives" where Person A owns Comp A, Person B owns Comp B, but they share Family Registry.
    family_registry = {}
    family_count = 0
    
    for seller, buyer in invoice_pairs:
        if random.random() < 0.2: # 20% of high value trades are related parties
            # randomly assign one as parent or relative
            rel_type = random.choice(["Owner", "Subsidiary", "Relative"])
            if rel_type == "Relative":
                family_id = f"FAM_{family_count}"
                family_count += 1
                person_id_a = f"UBO_FAM_{family_count}_A"
                person_id_b = f"UBO_FAM_{family_count}_B"
                # Since ownership is parent -> child, we define "seller" and "buyer" as owned by Family Registry nodes.
                # However, ownership format is tax_code -> tax_code. A "Relative" edge between Company A and Company B
                # signifies that Owner of A is a relative of Owner of B.
                key = (seller, buyer)
                if key not in seen_links:
                    seen_links.add(key)
                    ownerships.append((seller, buyer, 0.0, "Relative", person_id_b, "Brother/Sister of " + person_id_a))
            else:
                if random.random() < 0.5:
                    add_link(seller, buyer)
                else:
                    add_link(buyer, seller)

    # Note: Ownership chains length ~2-3
    
    cur.executemany("""
        INSERT INTO ownership_links (parent_tax_code, child_tax_code, ownership_percent, relationship_type, person_id, person_name)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, ownerships)

    cur.execute("SELECT count(*) FROM ownership_links;")
    total = cur.fetchone()[0]

    print(f"[SUCCESS] Created {total} ownership links connecting the 10000-node graph.")
    conn.close()

if __name__ == "__main__":
    seed_ownership_networks()
