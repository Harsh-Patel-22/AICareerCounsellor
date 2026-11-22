import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.nlp import ROOT

DEFAULT_PATH = ROOT / "artifacts" / "sessions.sqlite3"


class Conflict(Exception):
    pass


class Sessions:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(os.environ.get("CAREER_DB_PATH", str(DEFAULT_PATH)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    updated TEXT NOT NULL, content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    request_id TEXT NOT NULL,
                    PRIMARY KEY (session_id, request_id)
                );
            """)

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, state: dict) -> dict:
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE updated < ?",
                               ((datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),))
            connection.execute("INSERT INTO sessions VALUES (?, 0, ?, ?)",
                               (state["id"], self.now(), json.dumps(state, ensure_ascii=False)))
        return state

    def get(self, session_id: str) -> dict | None:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT content FROM sessions WHERE id=? AND updated >= ?",
                                     (session_id, (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat())).fetchone()
        return json.loads(row[0]) if row else None

    def has_request(self, session_id: str, request_id: str) -> bool:
        with closing(self.connect()) as connection:
            return bool(connection.execute("SELECT 1 FROM receipts WHERE session_id=? AND request_id=?",
                                           (session_id, request_id)).fetchone())

    def update(self, state: dict, expected: int, request_id: str | None = None) -> dict:
        with closing(self.connect()) as connection, connection:
            if request_id:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute("SELECT 1 FROM receipts WHERE session_id=? AND request_id=?",
                                      (state["id"], request_id)).fetchone():
                    row = connection.execute("SELECT content FROM sessions WHERE id=?", (state["id"],)).fetchone()
                    return json.loads(row[0])
            state["revision"] = expected + 1
            changed = connection.execute(
                "UPDATE sessions SET revision=?, updated=?, content=? WHERE id=? AND revision=?",
                (state["revision"], self.now(), json.dumps(state, ensure_ascii=False), state["id"], expected),
            ).rowcount
            if changed != 1:
                raise Conflict
            if request_id:
                connection.execute("INSERT INTO receipts VALUES (?, ?)", (state["id"], request_id))
        return state

    def delete(self, session_id: str) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))
