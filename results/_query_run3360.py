import sqlite3
import os
import glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbs = glob.glob(os.path.join(REPO, "**", "*.sqlite"), recursive=True)
dbs = [d for d in dbs if "assignment" in d.lower() or "experiment" in d.lower()]

for db in dbs:
    try:
        c = sqlite3.connect(db)
        c.row_factory = sqlite3.Row
        tables = [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        for t in tables:
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})")]
            if "run_id" not in cols:
                continue
            try:
                rows = c.execute(
                    f"SELECT * FROM {t} WHERE run_id = 3360 LIMIT 20"
                ).fetchall()
            except Exception:
                continue
            if rows:
                print(f"\n=== {db} :: {t} ===")
                for row in rows:
                    print(dict(row))
        c.close()
    except Exception as e:
        print(db, e)

# CSV search
for pat in [
    "**/run3360/**/*.csv",
    "**/*3360*.csv",
    "**/k70/**/*.csv",
]:
    files = glob.glob(os.path.join(REPO, pat), recursive=True)
    for f in files[:15]:
        if "3360" in f or "k70" in f.replace("\\", "/"):
            print("CSV", f)
