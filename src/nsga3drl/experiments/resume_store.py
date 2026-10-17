from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def task_key(task: dict) -> str:
    return hashlib.sha256(canonical_json(task).encode("utf-8")).hexdigest()


class ResumeStore:
    """Small SQLite ledger with one atomic commit per completed evaluation."""

    def __init__(self, path: str | Path, run_config: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS completed (task_key TEXT PRIMARY KEY, task TEXT NOT NULL, result TEXT NOT NULL)"
        )
        stored = self.connection.execute(
            "SELECT value FROM metadata WHERE key='run_config'"
        ).fetchone()
        config_text = canonical_json(run_config)
        if stored is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('run_config', ?)", (config_text,)
            )
            self.connection.commit()
        elif stored[0] != config_text:
            self.connection.close()
            raise ValueError(
                f"resume database {self.path} belongs to a different run configuration; "
                "choose a new --state-db path or restore the original arguments"
            )

    def completed_keys(self) -> set[str]:
        return {
            row[0]
            for row in self.connection.execute("SELECT task_key FROM completed")
        }

    def save(self, task: dict, result: dict) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO completed(task_key, task, result) VALUES(?, ?, ?)",
            (task_key(task), canonical_json(task), json.dumps(result, separators=(",", ":"))),
        )
        self.connection.commit()

    def records(self) -> list[tuple[dict, dict]]:
        rows = self.connection.execute("SELECT task, result FROM completed").fetchall()
        return [(json.loads(task), json.loads(result)) for task, result in rows]

    def close(self) -> None:
        self.connection.close()

