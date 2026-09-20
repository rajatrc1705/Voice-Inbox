import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
from livekit.agents import FunctionToolsExecutedEvent
from livekit.agents.llm import FunctionCall, FunctionCallOutput
from voice_inbox.tracing import TraceWriter


class TracingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.temporary_directory.name) / "traces.jsonl"
        self.writer = TraceWriter(self.trace_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def read_traces(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in self.trace_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_records_successful_tool_call_and_completed_turn(self) -> None:
        state = agent.SessionState(session_id="session-1")
        agent.start_turn(state, "I need to send the invoice", 10.0)
        event = FunctionToolsExecutedEvent(
            function_calls=[
                FunctionCall(
                    call_id="call-1",
                    name="create_task",
                    arguments='{"title":"Send the invoice"}',
                    created_at=11.0,
                )
            ],
            function_call_outputs=[
                FunctionCallOutput(
                    call_id="call-1",
                    name="create_task",
                    output="Created task task-1: Send the invoice",
                    is_error=False,
                    created_at=12.0,
                )
            ],
            created_at=12.0,
        )

        agent.record_tool_calls(state, event)
        with patch.object(agent, "trace_writer", self.writer):
            agent.complete_turn(state, "I've captured that task.", 13.0)

        traces = self.read_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["session_id"], "session-1")
        self.assertEqual(traces[0]["user_transcript"], "I need to send the invoice")
        self.assertEqual(traces[0]["assistant_response"], "I've captured that task.")
        self.assertEqual(traces[0]["total_duration"], 3.0)
        self.assertEqual(
            traces[0]["tool_calls"],
            [
                {
                    "name": "create_task",
                    "arguments": {"title": "Send the invoice"},
                    "result": "Created task task-1: Send the invoice",
                    "error": None,
                    "started_at": 11.0,
                    "completed_at": 12.0,
                }
            ],
        )
        self.assertIsNone(state.active_turn)

    def test_records_failed_tool_call(self) -> None:
        state = agent.SessionState(session_id="session-1")
        agent.start_turn(state, "Remind me sometime", 20.0)
        event = FunctionToolsExecutedEvent(
            function_calls=[
                FunctionCall(
                    call_id="call-2",
                    name="create_reminder",
                    arguments='{"title":"Call Shantanu","trigger_at":"tomorrow"}',
                    created_at=21.0,
                )
            ],
            function_call_outputs=[
                FunctionCallOutput(
                    call_id="call-2",
                    name="create_reminder",
                    output="trigger_at must be an ISO 8601 timestamp.",
                    is_error=True,
                    created_at=22.0,
                )
            ],
        )

        agent.record_tool_calls(state, event)

        tool_call = state.active_turn.tool_calls[0]
        self.assertIsNone(tool_call.result)
        self.assertEqual(
            tool_call.error,
            "trigger_at must be an ISO 8601 timestamp.",
        )

    def test_records_turn_without_tool_calls(self) -> None:
        state = agent.SessionState(session_id="session-1")
        agent.start_turn(state, "Remind me next week", 30.0)

        with patch.object(agent, "trace_writer", self.writer):
            agent.complete_turn(state, "What day and time next week?", 31.0)

        trace = self.read_traces()[0]
        self.assertEqual(trace["tool_calls"], [])
        self.assertEqual(trace["assistant_response"], "What day and time next week?")

    def test_writes_one_json_line_per_completed_turn(self) -> None:
        state = agent.SessionState(session_id="session-1")

        with patch.object(agent, "trace_writer", self.writer):
            agent.start_turn(state, "First turn", 1.0)
            agent.complete_turn(state, "First response", 2.0)
            agent.start_turn(state, "Second turn", 3.0)
            agent.complete_turn(state, "Second response", 4.0)

        self.assertEqual(len(self.read_traces()), 2)


if __name__ == "__main__":
    unittest.main()
