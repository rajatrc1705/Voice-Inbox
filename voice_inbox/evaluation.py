import json
from dataclasses import asdict, dataclass

from livekit.agents.voice.run_result import RunEvent

from .repository import VoiceInboxRepository


@dataclass
class ToolCallObservation:
    call_id: str
    name: str
    arguments: dict[str, object]
    result: str | None = None
    succeeded: bool | None = None


@dataclass
class TurnObservation:
    tool_calls: list[ToolCallObservation]
    assistant_response: str
    duration: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EvaluationCheck:
    name: str
    passed: bool
    expected: object
    actual: object

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def observe_run(events: list[RunEvent], duration: float) -> TurnObservation:
    tool_calls: list[ToolCallObservation] = []
    calls_by_id: dict[str, ToolCallObservation] = {}
    assistant_response = ""

    for event in events:
        if event.type == "function_call":
            call = ToolCallObservation(
                call_id=event.item.call_id,
                name=event.item.name,
                arguments=json.loads(event.item.arguments),
            )
            tool_calls.append(call)
            calls_by_id[call.call_id] = call
        elif event.type == "function_call_output":
            call = calls_by_id.get(event.item.call_id)
            if call is not None:
                call.result = event.item.output
                call.succeeded = not event.item.is_error
        elif event.type == "message" and event.item.role == "assistant":
            assistant_response = event.item.text_content or ""

    return TurnObservation(
        tool_calls=tool_calls,
        assistant_response=assistant_response,
        duration=duration,
    )


def grade_turn(
    expected: dict[str, object],
    observation: TurnObservation,
    repository: VoiceInboxRepository,
) -> list[EvaluationCheck]:
    checks = [
        EvaluationCheck(
            name="assistant_response_present",
            passed=bool(observation.assistant_response.strip()),
            expected="a non-empty assistant response",
            actual=observation.assistant_response,
        )
    ]
    expected_calls = expected.get("tool_calls", [])
    assert isinstance(expected_calls, list)

    expected_names = sorted(call["name"] for call in expected_calls)
    actual_names = sorted(call.name for call in observation.tool_calls)
    checks.append(
        EvaluationCheck(
            name="tool_selection",
            passed=actual_names == expected_names,
            expected=expected_names,
            actual=actual_names,
        )
    )

    unmatched_calls = list(observation.tool_calls)
    for index, expected_call in enumerate(expected_calls):
        actual_call = next(
            (call for call in unmatched_calls if call.name == expected_call["name"]),
            None,
        )
        if actual_call is None:
            continue
        unmatched_calls.remove(actual_call)

        exact_arguments = expected_call.get("arguments", {})
        for argument_name, expected_value in exact_arguments.items():
            actual_value = actual_call.arguments.get(argument_name)
            checks.append(
                EvaluationCheck(
                    name=f"tool_{index}_{argument_name}",
                    passed=actual_value == expected_value,
                    expected=expected_value,
                    actual=actual_value,
                )
            )

        contained_arguments = expected_call.get("argument_contains", {})
        for argument_name, required_parts in contained_arguments.items():
            actual_value = str(actual_call.arguments.get(argument_name, "")).lower()
            missing_parts = [
                part for part in required_parts if part.lower() not in actual_value
            ]
            checks.append(
                EvaluationCheck(
                    name=f"tool_{index}_{argument_name}_content",
                    passed=not missing_parts,
                    expected=required_parts,
                    actual=actual_call.arguments.get(argument_name),
                )
            )

        if "succeeds" in expected_call:
            checks.append(
                EvaluationCheck(
                    name=f"tool_{index}_success",
                    passed=actual_call.succeeded == expected_call["succeeds"],
                    expected=expected_call["succeeds"],
                    actual=actual_call.succeeded,
                )
            )

    if "clarification" in expected:
        asked_without_tool = not observation.tool_calls
        checks.append(
            EvaluationCheck(
                name="clarification",
                passed=asked_without_tool == expected["clarification"],
                expected=expected["clarification"],
                actual=asked_without_tool,
            )
        )

    response_lower = observation.assistant_response.lower()
    for required_text in expected.get("response_contains", []):
        checks.append(
            EvaluationCheck(
                name=f"response_contains_{required_text}",
                passed=required_text.lower() in response_lower,
                expected=required_text,
                actual=observation.assistant_response,
            )
        )
    for forbidden_text in expected.get("response_not_contains", []):
        checks.append(
            EvaluationCheck(
                name=f"response_excludes_{forbidden_text}",
                passed=forbidden_text.lower() not in response_lower,
                expected=f"not {forbidden_text}",
                actual=observation.assistant_response,
            )
        )

    state = expected.get("state", {})
    if state:
        reminders = repository.list_reminders()
        actual_state = {
            "tasks": len(repository.list_tasks()),
            "ideas": len(repository.list_ideas()),
            "reminders": len(reminders),
        }
        if "reminder_trigger_at" in state:
            actual_state["reminder_trigger_at"] = (
                reminders[0].trigger_at.isoformat() if len(reminders) == 1 else None
            )
        for item_type, expected_count in state.items():
            checks.append(
                EvaluationCheck(
                    name=f"state_{item_type}",
                    passed=actual_state[item_type] == expected_count,
                    expected=expected_count,
                    actual=actual_state[item_type],
                )
            )

    return checks
