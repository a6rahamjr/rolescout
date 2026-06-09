"""SQLite-backed alert state and alert matching service."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rolescout.data.contracts import RankedJob, SearchProfile
from rolescout.inference.service import RankingService


@dataclass(frozen=True)
class AlertRecord:
    alert_id: int
    name: str
    profile: SearchProfile
    min_score: float
    active: bool
    created_at: str
    updated_at: str
    last_checked_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "profile": asdict(self.profile),
            "min_score": self.min_score,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_checked_at": self.last_checked_at,
        }


class SQLiteAlertRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    min_score REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_checked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS delivered_jobs (
                    alert_id INTEGER NOT NULL,
                    job_id TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    PRIMARY KEY (alert_id, job_id),
                    FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(connection, "alerts", "updated_at TEXT")
            self._ensure_column(connection, "alerts", "last_checked_at TEXT")
            connection.execute("UPDATE alerts SET updated_at = created_at WHERE updated_at IS NULL")

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        definition: str,
    ) -> None:
        column = definition.split()[0]
        columns = {
            str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AlertRecord:
        return AlertRecord(
            alert_id=int(row["id"]),
            name=str(row["name"]),
            profile=SearchProfile.from_mapping(json.loads(row["profile_json"])),
            min_score=float(row["min_score"]),
            active=bool(row["active"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"] or row["created_at"]),
            last_checked_at=(str(row["last_checked_at"]) if row["last_checked_at"] else None),
        )

    def create(self, name: str, profile: SearchProfile, min_score: float) -> AlertRecord:
        created_at = datetime.now(UTC).isoformat()
        payload = json.dumps(asdict(profile), sort_keys=True)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO alerts (
                    name, profile_json, min_score, active, created_at, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (name, payload, min_score, created_at, created_at),
            )
            alert_id = int(cursor.lastrowid)
        return AlertRecord(
            alert_id,
            name,
            profile,
            min_score,
            True,
            created_at,
            created_at,
            None,
        )

    def list(self) -> list[AlertRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM alerts ORDER BY id").fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, alert_id: int) -> AlertRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._from_row(row) if row else None

    def update(
        self,
        alert_id: int,
        *,
        name: str | None = None,
        profile: SearchProfile | None = None,
        min_score: float | None = None,
        active: bool | None = None,
    ) -> AlertRecord | None:
        current = self.get(alert_id)
        if current is None:
            return None
        updated_at = datetime.now(UTC).isoformat()
        values = {
            "name": name if name is not None else current.name,
            "profile_json": json.dumps(asdict(profile or current.profile), sort_keys=True),
            "min_score": min_score if min_score is not None else current.min_score,
            "active": int(active if active is not None else current.active),
            "updated_at": updated_at,
            "alert_id": alert_id,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE alerts
                SET name = :name,
                    profile_json = :profile_json,
                    min_score = :min_score,
                    active = :active,
                    updated_at = :updated_at
                WHERE id = :alert_id
                """,
                values,
            )
        return self.get(alert_id)

    def delete(self, alert_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        return cursor.rowcount > 0

    def clear_deliveries(self, alert_id: int) -> bool:
        if self.get(alert_id) is None:
            return False
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM delivered_jobs WHERE alert_id = ?", (alert_id,))
        return True

    def record_check(self, alert_id: int) -> None:
        checked_at = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE alerts SET last_checked_at = ? WHERE id = ?",
                (checked_at, alert_id),
            )

    def unseen(self, alert_id: int, jobs: list[RankedJob]) -> list[RankedJob]:
        if not jobs:
            return []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM delivered_jobs WHERE alert_id = ?", (alert_id,)
            ).fetchall()
        delivered = {str(row["job_id"]) for row in rows}
        return [job for job in jobs if job.job.job_id not in delivered]

    def mark_delivered(self, alert_id: int, jobs: list[RankedJob]) -> None:
        delivered_at = datetime.now(UTC).isoformat()
        values = [(alert_id, job.job.job_id, delivered_at) for job in jobs]
        if not values:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO delivered_jobs (alert_id, job_id, delivered_at)
                VALUES (?, ?, ?)
                """,
                values,
            )


class AlertService:
    def __init__(
        self,
        repository: SQLiteAlertRepository,
        ranking_service: RankingService,
    ) -> None:
        self.repository = repository
        self.ranking_service = ranking_service

    async def check(self, alert_id: int, *, limit: int) -> list[RankedJob]:
        alert = self.repository.get(alert_id)
        if alert is None:
            raise KeyError(alert_id)
        if not alert.active:
            return []
        ranked = await self.ranking_service.search(
            alert.profile,
            limit=limit,
            min_score=alert.min_score,
        )
        unseen = self.repository.unseen(alert_id, ranked)
        self.repository.mark_delivered(alert_id, unseen)
        self.repository.record_check(alert_id)
        return unseen

    async def check_all(self, *, limit: int) -> dict[int, list[RankedJob]]:
        results: dict[int, list[RankedJob]] = {}
        for alert in self.repository.list():
            if alert.active:
                results[alert.alert_id] = await self.check(alert.alert_id, limit=limit)
        return results
