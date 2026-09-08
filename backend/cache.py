"""SQLite-backed cache for LLM responses, keyed by sha256(prompt)."""
import sqlite3
import json
import os

CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llm_cache.db")


def _conn():
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def get(key: str):
    with _conn() as conn:
        row = conn.execute("SELECT value FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def set(key: str, value):
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()
