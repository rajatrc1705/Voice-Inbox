"""Model-independent validation and execution of tool proposals.

The registry contains application-specific schemas and handlers. The executor
owns the shared outcome and recording contract; it never interprets user intent.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import MISSING, asdict, dataclass, fields, replace
from datetime import UTC, datetime
from time import monotonic
from typing import Literal
from uuid import uuid4

from .repository import MutationBlocked, VoiceInboxRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TitleInput:
    title: str


@dataclass(frozen=True)
class TextInput:
    text: str


@dataclass(frozen=True)
class ReminderInput:
    title: str
    trigger_at: datetime


@dataclass(frozen=True)
class ReminderUpdateInput:
    title: str | None = None
    trigger_at: datetime | None = None


@dataclass(frozen=True)
class ActionContext:
    session_id: str
    recent_item_ids: Mapping[str, str]
    source_transcript: str | None = None
    turn_id: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    action: str
    status: Literal["executed", "blocked", "failed"]
    reason: str
    next_step: Literal["none", "clarify", "correct_proposal"] = "none"
    result: dict[str, object] | None = None
    # A failure after an execution attempt must not promise no side effect.
    effect: Literal["none", "committed", "unknown"] = "none"
    recording_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSpec:
    schema: type
    item_type: str
    creates: bool
    handler: Callable


# Add a schema and handler here to reuse validation, outcomes, and recording.
ACTIONS = {
    "create_task": ActionSpec(TitleInput, "task", True,
        lambda repo, args, ctx, target: repo.create_task(
            title=args.title, source_transcript=ctx.source_transcript)),
    "create_idea": ActionSpec(TextInput, "idea", True,
        lambda repo, args, ctx, target: repo.create_idea(
            text=args.text, source_transcript=ctx.source_transcript)),
    "create_reminder": ActionSpec(ReminderInput, "reminder", True,
        lambda repo, args, ctx, target: repo.create_reminder(
            title=args.title, trigger_at=args.trigger_at,
            source_transcript=ctx.source_transcript)),
    "update_recent_task": ActionSpec(TitleInput, "task", False,
        lambda repo, args, ctx, target: repo.update_task_title(target, args.title)),
    "update_recent_idea": ActionSpec(TextInput, "idea", False,
        lambda repo, args, ctx, target: repo.update_idea_text(target, args.text)),
    "update_recent_reminder": ActionSpec(ReminderUpdateInput, "reminder", False,
        lambda repo, args, ctx, target: repo.update_reminder(
            target, title=args.title, trigger_at=args.trigger_at)),
}


def validate_arguments(schema: type, arguments: dict[str, object]) -> object:
    """Strict validation for the current text/timestamp schemas, without coercion."""
    declared = {field.name: field for field in fields(schema)}
    if not isinstance(arguments, dict) or arguments.keys() - declared.keys():
        raise MutationBlocked("invalid_arguments")
    values = {}
    for name, field in declared.items():
        value = arguments.get(name)
        if value is None:
            if field.default is MISSING:
                raise MutationBlocked("invalid_arguments")
            values[name] = None
            continue
        if not isinstance(value, str):
            raise MutationBlocked("invalid_arguments")
        value = value.strip()
        if not value:
            raise MutationBlocked("empty_content")
        if field.type in (datetime, datetime | None):
            try:
                value = datetime.fromisoformat(value)
            except ValueError as error:
                raise MutationBlocked("invalid_timestamp") from error
            if value.utcoffset() is None:
                raise MutationBlocked("missing_timezone")
        values[name] = value
    if not any(value is not None for value in values.values()):
        raise MutationBlocked("no_changes")
    return schema(**values)


def execute_action(
    action: str,
    arguments: dict[str, object],
    context: ActionContext,
    repository: VoiceInboxRepository,
    *,
    record: Callable[[dict[str, object]], None] | None = None,
) -> ActionOutcome:
    """Execute one attempt; expected blocks occur before any mutation SQL.

    Repository-dependent checks run inside the repository's write transaction.
    Recording is independent of voice turn completion. This is not retry deduplication.
    """
    action_id = uuid4().hex
    started_at = datetime.now(UTC).isoformat()
    started = monotonic()
    execution_started = None
    target = None
    try:
        spec = ACTIONS.get(action)
        if spec is None:
            raise MutationBlocked("unknown_action")
        args = validate_arguments(spec.schema, arguments)
        if spec.creates:
            if not context.source_transcript or not context.source_transcript.strip():
                raise MutationBlocked("context_unavailable")
        else:
            target = context.recent_item_ids.get(spec.item_type)
            if not target:
                raise MutationBlocked("target_not_found")
        execution_started = monotonic()
        item = spec.handler(repository, args, context, target)
        result = asdict(item)
        result = {key: value.isoformat() if isinstance(value, datetime) else value
                  for key, value in result.items()}
        if spec.item_type == "reminder":
            result["notification_delivery_active"] = False
        outcome = ActionOutcome(action_id, action, "executed", "ok",
                                result=result, effect="committed")
    except MutationBlocked as error:
        next_step = "correct_proposal"
        if error.reason in {"target_not_found", "target_inactive", "missing_timezone"}:
            next_step = "clarify"
        elif error.reason in {"no_changes", "context_unavailable"}:
            next_step = "none"
        outcome = ActionOutcome(action_id, action, "blocked", error.reason, next_step)
    except Exception:
        logger.exception("Action %s failed (%s)", action, action_id)
        outcome = ActionOutcome(action_id, action, "failed", "execution_error",
                                effect="unknown" if execution_started is not None else "none")
    finished = monotonic()
    event = {
        "schema_version": 1,
        "event": "action_finished",
        "session_id": context.session_id,
        "turn_id": context.turn_id,
        "tool_call_id": context.tool_call_id,
        "resolved_target_id": target,
        "proposal": {"action": action, "arguments": arguments},
        "started_at": started_at,
        "duration_seconds": finished - started,
        "validation_seconds": (execution_started or finished) - started,
        # Includes transactional target/precondition checks and persistence.
        "repository_seconds": finished - execution_started if execution_started else 0.0,
        "outcome": outcome.to_dict(),
    }
    if record is not None:
        try:
            record(event)
        except Exception:
            # Do not turn a committed action into a retryable execution failure.
            logger.exception("Could not record action %s", action_id)
            outcome = replace(outcome, recording_error="recording_failed")
    return outcome
