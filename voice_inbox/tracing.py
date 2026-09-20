import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ToolCallTrace:
    name: str
    arguments: dict[str, object]
    result: str | None
    error: str | None
    started_at: float
    completed_at: float


@dataclass
class TurnTrace:
    session_id: str
    turn_id: str
    user_transcript: str
    started_at: float
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    assistant_response: str | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["total_duration"] = (
            self.completed_at - self.started_at
            if self.completed_at is not None
            else None
        )
        return data


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, trace: TurnTrace) -> None:
        with self.path.open("a", encoding="utf-8") as trace_file:
            json.dump(trace.to_dict(), trace_file)
            trace_file.write("\n")
