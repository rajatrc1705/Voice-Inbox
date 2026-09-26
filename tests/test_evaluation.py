import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from datetime import datetime
from pathlib import Path

from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.voice.run_result import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)

from evals.run import run_case, summarize_results
from voice_inbox.execution import ActionContext, execute_action
from voice_inbox.evaluation import grade_turn, observe_run
from voice_inbox.repository import VoiceInboxRepository


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository = VoiceInboxRepository(
            Path(self.temporary_directory.name) / "eval.db"
        )
        self.repository.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_observes_tool_call_result_and_assistant_response(self) -> None:
        observation = observe_run(
            [
                FunctionCallEvent(
                    item=FunctionCall(
                        call_id="call-1",
                        name="create_task",
                        arguments='{"title":"Send the invoice"}',
                    )
                ),
                FunctionCallOutputEvent(
                    item=FunctionCallOutput(
                        call_id="call-1",
                        name="create_task",
                        output=json.dumps({"action_id": "attempt-1", "action": "create_task",
                                           "status": "executed", "reason": "ok", "effect": "committed"}),
                        is_error=False,
                    )
                ),
                ChatMessageEvent(
                    item=ChatMessage(
                        role="assistant",
                        content=["I've captured that task."],
                    )
                ),
            ],
            duration=1.25,
        )

        self.assertEqual(observation.tool_calls[0].name, "create_task")
        self.assertEqual(
            observation.tool_calls[0].arguments,
            {"title": "Send the invoice"},
        )
        self.assertTrue(observation.tool_calls[0].succeeded)
        self.assertEqual(observation.assistant_response, "I've captured that task.")
        self.assertEqual(observation.duration, 1.25)

    def test_structured_outcomes_are_graded_separately_from_tool_transport(self) -> None:
        for status, reason, effect in [
            ("executed", "ok", "committed"),
            ("blocked", "target_not_found", "none"),
            ("failed", "execution_error", "unknown"),
        ]:
            with self.subTest(status=status):
                outcome = {"action_id": "attempt-1", "action": "update_recent_task",
                           "status": status, "reason": reason, "effect": effect}
                observation = observe_run([
                    FunctionCallEvent(item=FunctionCall(
                        call_id="call-1", name="update_recent_task", arguments='{"title":"T"}')),
                    FunctionCallOutputEvent(item=FunctionCallOutput(
                        call_id="call-1", name="update_recent_task",
                        output=json.dumps(outcome), is_error=False)),
                    ChatMessageEvent(item=ChatMessage(role="assistant", content=["Result."])),
                ], duration=0.1)
                self.assertEqual(observation.tool_calls[0].succeeded, status == "executed")
                self.assertEqual(observation.tool_calls[0].action_outcome, outcome)
                expected = {"tool_calls": [{"name": "update_recent_task",
                    "succeeds": status == "executed",
                    "outcome": {"status": status, "reason": reason, "effect": effect}}]}
                self.assertTrue(all(check.passed for check in grade_turn(
                    expected, observation, self.repository)))
                expected["tool_calls"][0]["outcome"]["reason"] = "wrong_reason"
                checks = grade_turn(expected, observation, self.repository)
                self.assertFalse(next(check.passed for check in checks
                                      if check.name == "action_0_reason"))

    def test_malformed_mutation_outcomes_never_count_as_success(self) -> None:
        valid = {"action_id": "a", "action": "create_task", "status": "executed",
                 "reason": "ok", "effect": "committed"}
        outputs = ["Created task", "null", "[]", "{}"]
        outputs += [json.dumps({**valid, field: value}) for field, value in [
            ("status", []), ("action_id", ""), ("action", "create_idea"),
            ("reason", None), ("effect", "none"), ("effect", "unknown")]]
        for output in outputs:
            with self.subTest(output=output):
                observation = observe_run([
                    FunctionCallEvent(item=FunctionCall(
                        call_id="c", name="create_task", arguments='{"title":"T"}')),
                    FunctionCallOutputEvent(item=FunctionCallOutput(
                        call_id="c", name="create_task", output=output, is_error=False)),
                ], duration=0.1)
                self.assertFalse(observation.tool_calls[0].succeeded)

    def test_grades_structured_behavior_and_state(self) -> None:
        self.repository.create_task(
            title="Send the invoice",
            source_transcript="I need to send the invoice.",
        )
        observation = observe_run(
            [
                FunctionCallEvent(
                    item=FunctionCall(
                        call_id="call-1",
                        name="create_task",
                        arguments='{"title":"Send the invoice"}',
                    )
                ),
                FunctionCallOutputEvent(
                    item=FunctionCallOutput(
                        call_id="call-1",
                        name="create_task",
                        output=json.dumps({"action_id": "attempt-1", "action": "create_task",
                                           "status": "executed", "reason": "ok", "effect": "committed"}),
                        is_error=False,
                    )
                ),
                ChatMessageEvent(
                    item=ChatMessage(role="assistant", content=["Task captured."])
                ),
            ],
            duration=1.0,
        )
        expected = {
            "tool_calls": [
                {
                    "name": "create_task",
                    "argument_contains": {"title": ["send", "invoice"]},
                    "succeeds": True,
                }
            ],
            "response_contains": ["captured"],
            "state": {"tasks": 1, "ideas": 0, "reminders": 0},
        }

        checks = grade_turn(expected, observation, self.repository)

        self.assertTrue(all(check.passed for check in checks))

    def test_grader_exposes_each_failure(self) -> None:
        observation = observe_run(
            [
                ChatMessageEvent(
                    item=ChatMessage(
                        role="assistant",
                        content=["I recorded that reminder."],
                    )
                )
            ],
            duration=1.0,
        )
        expected = {
            "tool_calls": [],
            "clarification": True,
            "response_contains": ["day", "time"],
            "response_not_contains": ["recorded"],
        }

        checks = grade_turn(expected, observation, self.repository)
        failed_names = {check.name for check in checks if not check.passed}

        self.assertEqual(
            failed_names,
            {
                "response_contains_day",
                "response_contains_time",
                "response_excludes_recorded",
            },
        )

    def test_summarizes_behavior_metrics_separately(self) -> None:
        metrics = summarize_results(
            [
                {
                    "passed": False,
                    "turns": [
                        {
                            "observation": {"duration": 2.0},
                            "checks": [
                                {"name": "tool_selection", "passed": True},
                                {"name": "state_tasks", "passed": False},
                            ],
                        }
                    ],
                }
            ]
        )

        self.assertEqual(metrics["tool_selection"]["rate"], 1.0)
        self.assertEqual(metrics["application_state"]["rate"], 0.0)
        self.assertEqual(metrics["case_success"]["rate"], 0.0)
        self.assertEqual(metrics["average_turn_duration"], 2.0)

    def test_correction_grader_checks_updated_reminder_time(self) -> None:
        reminder = self.repository.create_reminder(
            title="Call Alex",
            trigger_at=datetime.fromisoformat("2026-09-22T11:00:00+02:00"),
            source_transcript="Remind me to call Alex tomorrow at eleven",
        )
        self.repository.update_reminder(
            reminder.id,
            trigger_at=datetime.fromisoformat("2026-09-22T16:00:00+02:00"),
        )
        observation = observe_run(
            [ChatMessageEvent(item=ChatMessage(role="assistant", content=["Updated."]))],
            duration=1.0,
        )

        checks = grade_turn(
            {
                "tool_calls": [],
                "state": {
                    "reminders": 1,
                    "reminder_trigger_at": "2026-09-22T16:00:00+02:00",
                },
            },
            observation,
            self.repository,
        )
        self.assertTrue(all(check.passed for check in checks))

        wrong_time_checks = grade_turn(
            {
                "tool_calls": [],
                "state": {"reminder_trigger_at": "2026-09-22T17:00:00+02:00"},
            },
            observation,
            self.repository,
        )
        self.assertFalse(
            next(
                check.passed
                for check in wrong_time_checks
                if check.name == "state_reminder_trigger_at"
            )
        )



class EvaluationFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_failure_preserves_completed_and_incomplete_action_records(self):
        for failure_mode in ("raised", "event"):
            with self.subTest(failure_mode=failure_mode):
                callbacks = {}
                session = SimpleNamespace(
                    userdata=SimpleNamespace(transcript_ready=SimpleNamespace(set=lambda: None)),
                    start=AsyncMock(), aclose=AsyncMock(),
                    on=lambda name: lambda fn: callbacks.setdefault(name, fn),
                )
                calls = 0

                def build_session(repository, model, **kwargs):
                    async def run(user_input):
                        nonlocal calls
                        calls += 1
                        outcome = execute_action("create_task", {"title": user_input},
                            ActionContext("s", {}, user_input), repository,
                            record=kwargs["action_recorder"])
                        if calls == 2:
                            if failure_mode == "raised":
                                raise RuntimeError("model disconnected")
                            callbacks["error"](SimpleNamespace(error="model disconnected"))
                        return SimpleNamespace(events=[
                            FunctionCallEvent(item=FunctionCall(
                                call_id=str(calls), name="create_task",
                                arguments=json.dumps({"title": user_input}))),
                            FunctionCallOutputEvent(item=FunctionCallOutput(
                                call_id=str(calls), name="create_task",
                                output=json.dumps(outcome.to_dict()), is_error=False)),
                            ChatMessageEvent(item=ChatMessage(role="assistant", content=["Captured."])),
                        ])
                    session.run = run
                    return session

                case = {"id": "failure", "now": "2026-09-22T10:00:00Z", "turns": [
                    {"user": title, "expected": {"tool_calls": [{"name": "create_task",
                     "outcome": {"status": "executed"}}]}} for title in ("First", "Second")]}
                with patch("evals.run.agent.build_session", side_effect=build_session):
                    result = await run_case(case, object())
                self.assertFalse(result["passed"])
                self.assertEqual(result["execution_error"], "model disconnected")
                self.assertEqual(len(result["turns"]), 1)
                self.assertTrue(result["turns"][0]["passed"])
                self.assertEqual(result["turns"][0]["action_events"][0]["outcome"]["result"]["title"], "First")
                self.assertEqual(result["incomplete_action_events"][0]["outcome"]["result"]["title"], "Second")
                session.aclose.assert_awaited_once()
                json.dumps(result)

    async def test_startup_and_cleanup_failures_are_reported_and_session_closed(self):
        for failure_mode in ("startup", "cleanup"):
            with self.subTest(failure_mode=failure_mode):
                session = SimpleNamespace(
                    on=lambda name: lambda fn: fn,
                    start=AsyncMock(side_effect=RuntimeError("startup") if failure_mode == "startup" else None),
                    aclose=AsyncMock(side_effect=RuntimeError("cleanup") if failure_mode == "cleanup" else None),
                )
                with patch("evals.run.agent.build_session", return_value=session):
                    result = await run_case({"id": "failure", "now": "2026-09-22T10:00:00Z", "turns": []}, object())
                self.assertFalse(result["passed"])
                self.assertIn(failure_mode, result["execution_error"])
                self.assertEqual(result["incomplete_action_events"], [])
                session.aclose.assert_awaited_once()

if __name__ == "__main__":
    unittest.main()
