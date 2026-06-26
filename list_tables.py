import sqlite3
import glob

dbs = glob.glob('*_ARF.gpkg')
if not dbs:
    dbs = glob.glob('*.gpkg')
    
print('DBS:', dbs)

if dbs:
    for db in dbs:
        try:
            conn = sqlite3.connect(db)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            print(f"Tables in {db}:", tables)
            
            # if we find a table named catalogo, estaciones, etc, print its columns
            for t in tables:
                if 'estacion' in t.lower() or 'catalogo' in t.lower():
                    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t});").fetchall()]
                    print(f"  Columns in {t}:", cols)
        except Exception as e:
            print("Error reading db", e)
