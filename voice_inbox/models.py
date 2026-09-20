"""Domain models stored by Voice Inbox.

These dataclasses describe application state independently of LiveKit, an LLM,
or the HTTP API. Keeping them independent lets those outer layers change without
changing what a task, idea, or reminder means.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TaskStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReminderStatus(StrEnum):
    SCHEDULED = "scheduled"
    FIRED = "fired"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    title: str
    details: str | None
    due_at: datetime | None
    status: TaskStatus
    source_transcript: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Idea:
    id: str
    text: str
    source_transcript: str
    created_at: datetime


@dataclass
class Reminder:
    id: str
    title: str
    trigger_at: datetime
    status: ReminderStatus
    source_transcript: str
    created_at: datetime
    updated_at: datetime
