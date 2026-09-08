import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "factloom.db")

# Columns added to `facts` beyond the original prototype schema, and their SQL types.
# Kept as an explicit migration list so an existing DB (with the old schema) upgrades
# in place via ALTER TABLE instead of requiring the file to be deleted.
_FACTS_NEW_COLUMNS = {
    "chunk_id": "TEXT",
    "is_reported_value": "INTEGER",           # 1/0/NULL
    "embedding_status": "TEXT DEFAULT 'pending'",
    "embedding_model": "TEXT",
    "embedding_error": "TEXT",
    "norm_metric": "TEXT",                    # canonical metric, e.g. revenue_growth
    "norm_value": "REAL",
    "norm_unit": "TEXT",
    "norm_period": "TEXT",
    "norm_scope": "TEXT",
    "numeric_confidence": "REAL",             # indicative 0-1 confidence, distinct from the low/medium/high label
}

_RELATIONS_NEW_COLUMNS = {
    "confidence": "REAL",
    "reason": "TEXT DEFAULT 'none'",
    "fact_a_evidence": "TEXT",
    "fact_b_evidence": "TEXT",
}


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT,
                num_pages INTEGER,
                status TEXT DEFAULT 'pending',
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                page_no INTEGER,
                entity TEXT,
                metric TEXT,
                value TEXT,
                unit TEXT,
                period TEXT,
                scope TEXT,
                quote TEXT,
                confidence TEXT,
                embedding TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER,
                fact_b_id INTEGER,
                relation_type TEXT,
                explanation TEXT
            );
            """
        )
        _migrate_columns(conn, "facts", _FACTS_NEW_COLUMNS)
        _migrate_columns(conn, "documents", {"status": "TEXT DEFAULT 'pending'", "error": "TEXT"})
        _migrate_columns(conn, "relations", _RELATIONS_NEW_COLUMNS)


def _migrate_columns(conn, table: str, columns: dict[str, str]):
    """Add any missing columns to an existing table. Safe to call every startup."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, coltype in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_document(doc_id, filename, num_pages):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, filename, num_pages, status) VALUES (?,?,?,?)",
            (doc_id, filename, num_pages, "processing"),
        )


def set_document_status(doc_id: str, status: str, error: str | None = None):
    with get_conn() as conn:
        conn.execute("UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id))


def add_fact(fact: dict) -> int:
    """fact is expected to already contain embedding fields (status/model/error/vector-as-json)
    and, optionally, normalized fields (norm_*). Missing keys default to NULL."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO facts
            (document_id, chunk_id, page_no, entity, metric, value, unit, period, scope, quote,
             confidence, is_reported_value, embedding, embedding_status, embedding_model, embedding_error,
             norm_metric, norm_value, norm_unit, norm_period, norm_scope, numeric_confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fact["document_id"], fact.get("chunk_id"), fact["page_no"], fact["entity"], fact["metric"],
                fact["value"], fact.get("unit"), fact.get("period"), fact.get("scope"), fact["quote"],
                fact.get("confidence", "medium"), fact.get("is_reported_value"),
                fact.get("embedding_json"), fact.get("embedding_status", "pending"),
                fact.get("embedding_model"), fact.get("embedding_error"),
                fact.get("norm_metric"), fact.get("norm_value"), fact.get("norm_unit"),
                fact.get("norm_period"), fact.get("norm_scope"), fact.get("numeric_confidence", 0.6),
            ),
        )
        return cur.lastrowid


def get_all_facts(exclude_document_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM facts").fetchall()
    facts = [dict(r) for r in rows]
    if exclude_document_id:
        facts = [f for f in facts if f["document_id"] != exclude_document_id]
    return facts


def get_facts_by_document(document_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM facts WHERE document_id=?", (document_id,)).fetchall()
    return [dict(r) for r in rows]


def add_relation(fact_a_id, fact_b_id, relation_type, explanation, confidence=None,
                  reason="none", fact_a_evidence="", fact_b_evidence=""):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO relations
            (fact_a_id, fact_b_id, relation_type, explanation, confidence, reason, fact_a_evidence, fact_b_evidence)
            VALUES (?,?,?,?,?,?,?,?)""",
            (fact_a_id, fact_b_id, relation_type, explanation, confidence, reason, fact_a_evidence, fact_b_evidence),
        )
        return cur.lastrowid


def get_relations_for_fact(fact_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM relations WHERE fact_a_id=? OR fact_b_id=?", (fact_id, fact_id)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_relations() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM relations").fetchall()
    return [dict(r) for r in rows]


def get_fact(fact_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents").fetchall()
    return [dict(r) for r in rows]


def delete_document_facts(document_id: str):
    with get_conn() as conn:
        fact_ids = [r[0] for r in conn.execute("SELECT id FROM facts WHERE document_id=?", (document_id,)).fetchall()]
        if fact_ids:
            placeholders = ",".join("?" for _ in fact_ids)
            conn.execute(f"DELETE FROM relations WHERE fact_a_id IN ({placeholders}) OR fact_b_id IN ({placeholders})", fact_ids + fact_ids)
            conn.execute("DELETE FROM facts WHERE document_id=?", (document_id,))
