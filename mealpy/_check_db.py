"""Check available assignments in DB."""
import sqlite3, os
db_path = os.path.join(os.path.dirname(__file__), 'results', 'assignment_experiments.sqlite')
db = sqlite3.connect(db_path)

cur = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print(f"Tables: {tables}")

for tbl in tables:
    cur2 = db.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tbl}'")
    for r in cur2.fetchall():
        print(f"\n--- {tbl} ---")
        print(r[0])
    # Show sample data
    cur3 = db.execute(f"SELECT * FROM {tbl} LIMIT 5")
    rows = cur3.fetchall()
    print(f"  Sample ({len(rows)} rows):")
    for row in rows:
        print(f"    {row}")

db.close()
