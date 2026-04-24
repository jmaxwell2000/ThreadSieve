from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import KnowledgeItem, Thread


SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
  id TEXT PRIMARY KEY,
  source_app TEXT NOT NULL,
  title TEXT NOT NULL,
  source_uri TEXT,
  local_path TEXT NOT NULL,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS objects (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  tags TEXT NOT NULL,
  confidence REAL NOT NULL,
  status TEXT NOT NULL,
  source_thread_id TEXT NOT NULL,
  local_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_thread_id) REFERENCES threads(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS objects_fts USING fts5(
  object_id UNINDEXED,
  title,
  summary,
  tags,
  content
);
"""


def db_path(workspace: Path) -> Path:
    return workspace / "index.sqlite"


def connect(workspace: Path) -> sqlite3.Connection:
    workspace.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path(workspace))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def index_thread(workspace: Path, thread: Thread, local_path: Path) -> None:
    with connect(workspace) as conn:
        conn.execute(
            """
            INSERT INTO threads (id, source_app, title, source_uri, local_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              source_app=excluded.source_app,
              title=excluded.title,
              source_uri=excluded.source_uri,
              local_path=excluded.local_path,
              created_at=excluded.created_at,
              updated_at=excluded.updated_at
            """,
            (thread.id, thread.source_app, thread.title, thread.source_uri, str(local_path), thread.created_at, thread.updated_at),
        )


def index_object(workspace: Path, item: KnowledgeItem, thread: Thread, local_path: Path) -> None:
    tags = ",".join(item.tags)
    with connect(workspace) as conn:
        conn.execute(
            """
            INSERT INTO objects (id, type, title, summary, tags, confidence, status, source_thread_id, local_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              type=excluded.type,
              title=excluded.title,
              summary=excluded.summary,
              tags=excluded.tags,
              confidence=excluded.confidence,
              status=excluded.status,
              source_thread_id=excluded.source_thread_id,
              local_path=excluded.local_path
            """,
            (
                item.id,
                item.type,
                item.title,
                item.summary,
                tags,
                item.confidence,
                item.status,
                thread.id,
                str(local_path),
                item.created_at,
            ),
        )
        conn.execute("DELETE FROM objects_fts WHERE object_id = ?", (item.id,))
        conn.execute(
            "INSERT INTO objects_fts (object_id, title, summary, tags, content) VALUES (?, ?, ?, ?, ?)",
            (item.id, item.title, item.summary, tags, item.body or ""),
        )


def search(workspace: Path, query: str, limit: int = 10) -> list[dict[str, object]]:
    with connect(workspace) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT objects.id, objects.type, objects.title, objects.summary, objects.tags, objects.local_path,
                       bm25(objects_fts) AS rank
                FROM objects_fts
                JOIN objects ON objects.id = objects_fts.object_id
                WHERE objects_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query}%"
            rows = conn.execute(
                """
                SELECT id, type, title, summary, tags, local_path, 0 AS rank
                FROM objects
                WHERE title LIKE ? OR summary LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like, like, like, limit),
            ).fetchall()
    return [dict(row) for row in rows]


def get_object(workspace: Path, object_id: str) -> dict[str, object] | None:
    with connect(workspace) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
    return dict(row) if row else None


def latest_thread_path(workspace: Path) -> Path | None:
    with connect(workspace) as conn:
        row = conn.execute("SELECT local_path FROM threads ORDER BY rowid DESC LIMIT 1").fetchone()
    return Path(row[0]) if row else None
