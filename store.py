"""SQLite store for past sprays (data/sprays.db)."""
import os, sqlite3
from config import DATA
import datasrc as ds

DB = os.path.join(DATA, "sprays.db")
COLS = ["field_name", "lat", "lon", "crop", "pesticide_type", "product", "volatility", "start_time", "end_time", "status",
        "confidence_pct", "wind_mph", "wind_low", "wind_high", "wind_from", "temp_f", "inversion_pct", "sensitive_areas", "outcome", "notes"]


def _con():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS sprays (id INTEGER PRIMARY KEY AUTOINCREMENT, logged_at TEXT, " + ", ".join(c + " TEXT" for c in COLS) + ")")
    return con


def add(rec):
    vals = [ds.now_local().strftime("%Y-%m-%d %H:%M:%S")] + [None if rec.get(c) is None else str(rec.get(c))[:500] for c in COLS]
    with _con() as con:
        cur = con.execute("INSERT INTO sprays (logged_at, %s) VALUES (%s)" % (", ".join(COLS), ", ".join("?" * (len(COLS) + 1))), vals)
        return cur.lastrowid


def recent(n=100):
    with _con() as con:
        return [dict(r) for r in con.execute("SELECT * FROM sprays ORDER BY id DESC LIMIT ?", (int(n),))]


def set_outcome(sid, outcome):
    with _con() as con:
        con.execute("UPDATE sprays SET outcome = ? WHERE id = ?", (str(outcome)[:100], int(sid)))


def delete(sid):
    with _con() as con:
        con.execute("DELETE FROM sprays WHERE id = ?", (int(sid),))
