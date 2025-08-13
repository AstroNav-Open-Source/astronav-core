import sqlite3
import os

DB_PATH = r"C:\Users\anast\Documents\star_treckers code\star-treckers\star_catalog.db"

print("Opening DB at:", DB_PATH)
print("Exists?", os.path.exists(DB_PATH))

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in this DB:", c.fetchall())
conn.close()