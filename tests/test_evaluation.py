import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.voice.run_result import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)

from evals.run import summarize_results
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
                        output="Created task task-1: Send the invoice",
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
                        output="Created task task-1: Send the invoice",
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


if __name__ == "__main__":
    unittest.main()
