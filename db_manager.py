from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from config import DATABASE_PATH


class DatabaseManager:
    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL DEFAULT 'New conversation',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS message_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    page INTEGER,
                    snippet TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def add_document(
        self,
        document_id: str,
        filename: str,
        content_hash: str,
        file_type: str,
        page_count: int | None,
        chunk_count: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO documents
                (document_id, filename, content_hash, file_type, page_count, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (document_id, filename, content_hash, file_type, page_count, chunk_count),
            )

    def get_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._dict(connection.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone())

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY uploaded_at DESC, filename").fetchall()
            return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def clear_documents(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")

    def create_conversation(self, title: str = "New conversation") -> int:
        with self._connect() as connection:
            cursor = connection.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
            return int(cursor.lastrowid)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM conversations ORDER BY updated_at DESC, id DESC").fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._dict(connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone())

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        title = title.strip()[:100] or "New conversation"
        with self._connect() as connection:
            connection.execute("UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?", (title, conversation_id))

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def clear_conversation(self, conversation_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

    def clear_all_conversations(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversations")

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )
            connection.execute("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?", (conversation_id,))
            if role == "user":
                connection.execute(
                    "UPDATE conversations SET title = ? WHERE id = ? AND title = 'New conversation'",
                    (content[:60], conversation_id),
                )
            return int(cursor.lastrowid)

    def get_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def add_message_source(self, message_id: int, filename: str, page: int | None, snippet: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO message_sources (message_id, filename, page, snippet) VALUES (?, ?, ?, ?)",
                (message_id, filename, page, snippet),
            )

    def get_message_sources(self, message_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM message_sources WHERE message_id = ? ORDER BY id", (message_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def count_messages(self, role: str | None = None) -> int:
        with self._connect() as connection:
            if role:
                row = connection.execute("SELECT COUNT(*) AS count FROM messages WHERE role = ?", (role,)).fetchone()
            else:
                row = connection.execute("SELECT COUNT(*) AS count FROM messages").fetchone()
            return int(row["count"])
