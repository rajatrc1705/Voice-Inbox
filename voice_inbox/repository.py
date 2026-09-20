"""SQLite persistence for Voice Inbox domain state."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import Idea, Reminder, ReminderStatus, Task, TaskStatus


def utc_now() -> datetime:
    return datetime.now(UTC)


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class VoiceInboxRepository:
    """Owns database access; callers work with domain objects, not SQL rows."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    details TEXT,
                    due_at TEXT,
                    status TEXT NOT NULL,
                    source_transcript TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ideas (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_transcript TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    trigger_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_transcript TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.commit()

    def create_task(
        self,
        *,
        title: str,
        source_transcript: str,
        details: str | None = None,
        due_at: datetime | None = None,
    ) -> Task:
        now = utc_now()
        task = Task(
            id=uuid4().hex,
            title=title,
            details=details,
            due_at=due_at,
            status=TaskStatus.OPEN,
            source_transcript=source_transcript,
            created_at=now,
            updated_at=now,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO tasks
                    (id, title, details, due_at, status, source_transcript, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.title,
                    task.details,
                    _serialize_datetime(task.due_at),
                    task.status,
                    task.source_transcript,
                    _serialize_datetime(task.created_at),
                    _serialize_datetime(task.updated_at),
                ),
            )
            connection.commit()
        return task

    def create_idea(self, *, text: str, source_transcript: str) -> Idea:
        idea = Idea(
            id=uuid4().hex,
            text=text,
            source_transcript=source_transcript,
            created_at=utc_now(),
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO ideas (id, text, source_transcript, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    idea.id,
                    idea.text,
                    idea.source_transcript,
                    _serialize_datetime(idea.created_at),
                ),
            )
            connection.commit()
        return idea

    def create_reminder(
        self, *, title: str, trigger_at: datetime, source_transcript: str
    ) -> Reminder:
        now = utc_now()
        reminder = Reminder(
            id=uuid4().hex,
            title=title,
            trigger_at=trigger_at,
            status=ReminderStatus.SCHEDULED,
            source_transcript=source_transcript,
            created_at=now,
            updated_at=now,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO reminders
                    (id, title, trigger_at, status, source_transcript, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reminder.id,
                    reminder.title,
                    _serialize_datetime(reminder.trigger_at),
                    reminder.status,
                    reminder.source_transcript,
                    _serialize_datetime(reminder.created_at),
                    _serialize_datetime(reminder.updated_at),
                ),
            )
            connection.commit()
        return reminder

    def list_tasks(self) -> list[Task]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created_at"
            ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def list_ideas(self) -> list[Idea]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM ideas ORDER BY created_at"
            ).fetchall()
        return [self._idea_from_row(row) for row in rows]

    def list_reminders(self) -> list[Reminder]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM reminders ORDER BY trigger_at"
            ).fetchall()
        return [self._reminder_from_row(row) for row in rows]

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            details=row["details"],
            due_at=(
                datetime.fromisoformat(row["due_at"])
                if row["due_at"] is not None
                else None
            ),
            status=TaskStatus(row["status"]),
            source_transcript=row["source_transcript"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _idea_from_row(row: sqlite3.Row) -> Idea:
        return Idea(
            id=row["id"],
            text=row["text"],
            source_transcript=row["source_transcript"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _reminder_from_row(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            title=row["title"],
            trigger_at=datetime.fromisoformat(row["trigger_at"]),
            status=ReminderStatus(row["status"]),
            source_transcript=row["source_transcript"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
