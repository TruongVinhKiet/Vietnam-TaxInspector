import psycopg2
conn = psycopg2.connect("postgresql://postgres:Kiet2004@localhost:5432/TaxInspector")
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'kg_entities' ORDER BY ordinal_position")
print("kg_entities:", [r[0] for r in cur.fetchall()])
cur.execute("SELECT count(*) FROM kg_entities")
print("rows:", cur.fetchone()[0])
conn.close()
