import psycopg2
conn = psycopg2.connect("postgresql://postgres:Kiet2004@localhost:5432/TaxInspector")
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'kg_relations' ORDER BY ordinal_position")
print("kg_relations:", [r[0] for r in cur.fetchall()])
conn.close()
