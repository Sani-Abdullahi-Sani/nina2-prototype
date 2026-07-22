"""
db.py — Question logging for the Admin dashboard.

PROTOTYPE NOTE: Logs every question to a local SQLite file so the Admin tab
has something real to show, live, in the demo.

PRODUCTION RECOMMENDATION: Replace with LangFuse (already in the tech stack
in the pitch deck) — this gives the same visibility (what Nina can't answer)
as a permanent, always-on feature instead of a one-off manual benchmark.
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "nina_log.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            category TEXT,
            confidence REAL,
            tier TEXT,
            sources TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_question(question, category, confidence, tier, sources):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO log (timestamp, question, category, confidence, tier, sources) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            question,
            category,
            confidence,
            tier,
            ", ".join(sources) if sources else "",
        ),
    )
    conn.commit()
    conn.close()


def fetch_all():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM log ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_stats():
    rows = fetch_all()
    total = len(rows)
    if total == 0:
        return {"total": 0, "answered": 0, "hedged": 0, "not_found": 0}
    return {
        "total": total,
        "answered": sum(1 for r in rows if r["tier"] == "answered"),
        "hedged": sum(1 for r in rows if r["tier"] == "hedged"),
        "not_found": sum(1 for r in rows if r["tier"] == "not_found"),
    }
