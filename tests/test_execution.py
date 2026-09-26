import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
from voice_inbox.execution import ACTIONS, ActionContext, execute_action
from voice_inbox.repository import VoiceInboxRepository
from voice_inbox.tracing import TraceWriter


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repository = VoiceInboxRepository(Path(self.directory.name) / "test.db")
        self.repository.initialize()
        self.context = ActionContext("session-1", {}, "User supplied content", "turn-1", "call-1")
        self.events = []

    def run_action(self, name, arguments, context=None):
        return execute_action(name, arguments, context or self.context,
                              self.repository, record=self.events.append)

    def snapshot(self):
        with sqlite3.connect(self.repository.database_path) as connection:
            return tuple(tuple(connection.execute(f"SELECT * FROM {table} ORDER BY id"))
                         for table in ("tasks", "ideas", "reminders"))

    def seed(self):
        task = self.repository.create_task(title="Task", source_transcript="source")
        idea = self.repository.create_idea(text="Idea", source_transcript="source")
        reminder = self.repository.create_reminder(
            title="Reminder", trigger_at=datetime.fromisoformat("2026-09-22T11:00:00+02:00"),
            source_transcript="source")
        self.context = replace(self.context, recent_item_ids={
            "task": task.id, "idea": idea.id, "reminder": reminder.id})

    def test_all_six_actions_execute_and_record_structured_results(self):
        cases = [
            ("create_task", {"title": "Task"}, "task"),
            ("create_idea", {"text": "Idea"}, "idea"),
            ("create_reminder", {"title": "Reminder", "trigger_at": "2026-09-22T11:00:00+02:00"}, "reminder"),
            ("update_recent_task", {"title": "Changed task"}, "task"),
            ("update_recent_idea", {"text": "Changed idea"}, "idea"),
            ("update_recent_reminder", {"title": "Changed reminder"}, "reminder"),
        ]
        ids = {}
        for name, arguments, item_type in cases:
            with self.subTest(action=name):
                outcome = self.run_action(name, arguments, replace(self.context, recent_item_ids=dict(ids)))
                self.assertEqual(outcome.status, "executed")
                self.assertEqual(outcome.effect, "committed")
                if name.startswith("create"):
                    ids[item_type] = outcome.result["id"]
                else:
                    self.assertEqual(outcome.result["id"], ids[item_type])
                event = self.events[-1]
                self.assertEqual(event["outcome"], outcome.to_dict())
                self.assertEqual(event["tool_call_id"], "call-1")
                self.assertEqual(event["turn_id"], "turn-1")
                self.assertGreaterEqual(event["duration_seconds"], 0)
                json.dumps(event)
        self.assertEqual(len({e["outcome"]["action_id"] for e in self.events}), 6)
        reopened = VoiceInboxRepository(self.repository.database_path)
        self.assertEqual(reopened.list_tasks()[0].title, "Changed task")
        self.assertEqual(reopened.list_tasks()[0].source_transcript, self.context.source_transcript)
        self.assertEqual(reopened.list_ideas()[0].text, "Changed idea")
        reminder = reopened.list_reminders()[0]
        self.assertEqual(reminder.title, "Changed reminder")
        self.assertEqual(reminder.trigger_at.hour, 11)

    def test_schema_blocks_never_call_repository_or_change_state(self):
        self.seed()
        cases = [
            ("unknown", {}, "unknown_action"),
            ("create_task", {}, "invalid_arguments"),
            ("create_task", {"title": None}, "invalid_arguments"),
            ("create_task", {"title": 123}, "invalid_arguments"),
            ("create_task", {"title": "  "}, "empty_content"),
            ("create_task", {"title": "New", "source_transcript": "spoof"}, "invalid_arguments"),
            ("create_idea", {"text": []}, "invalid_arguments"),
            ("create_idea", {"text": ""}, "empty_content"),
            ("create_reminder", {"title": "R"}, "invalid_arguments"),
            ("create_reminder", {"title": "R", "trigger_at": "later"}, "invalid_timestamp"),
            ("create_reminder", {"title": "R", "trigger_at": "2026-09-25"}, "missing_timezone"),
            ("create_reminder", {"title": "R", "trigger_at": "2026-09-25T11:00:00"}, "missing_timezone"),
            ("update_recent_task", {"title": "New", "target_id": "other"}, "invalid_arguments"),
            ("update_recent_task", {"title": ""}, "empty_content"),
            ("update_recent_idea", {"text": False}, "invalid_arguments"),
            ("update_recent_reminder", {}, "no_changes"),
            ("update_recent_reminder", {"title": None, "trigger_at": None}, "no_changes"),
            ("update_recent_reminder", {"trigger_at": "bad"}, "invalid_timestamp"),
        ]
        before = self.snapshot()
        with patch.object(self.repository, "_connect", side_effect=AssertionError("repository entered")) as connect:
            for name, arguments, reason in cases:
                with self.subTest(action=name, arguments=arguments):
                    outcome = self.run_action(name, arguments)
                    self.assertEqual((outcome.status, outcome.reason, outcome.effect),
                                     ("blocked", reason, "none"))
            connect.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_missing_trusted_context_is_blocked_without_repository_access(self):
        for name, arguments in [("create_task", {"title": "T"}),
                                ("update_recent_task", {"title": "T"})]:
            with self.subTest(action=name):
                with patch.object(self.repository, "_connect") as connect:
                    outcome = self.run_action(name, arguments, replace(self.context, source_transcript=None))
                    self.assertEqual(outcome.status, "blocked")
                    connect.assert_not_called()

    def test_missing_inactive_and_noop_targets_execute_no_mutation_sql(self):
        self.seed()
        cases = [
            ("update_recent_task", {"title": "Task"}, self.context, "no_changes"),
            ("update_recent_idea", {"text": "Idea"}, self.context, "no_changes"),
            ("update_recent_reminder", {"title": "Reminder"}, self.context, "no_changes"),
            # Same instant, different textual offset: still a no-op.
            ("update_recent_reminder", {"trigger_at": "2026-09-22T09:00:00Z"}, self.context, "no_changes"),
        ]
        for kind, args in [("task", {"title": "New"}), ("idea", {"text": "New"}),
                           ("reminder", {"title": "New"})]:
            cases.append(("update_recent_" + kind, args,
                          replace(self.context, recent_item_ids={kind: "deleted"}), "target_not_found"))
        statements = []
        original_connect = self.repository._connect

        def traced_connect():
            connection = original_connect()
            connection.set_trace_callback(statements.append)
            return connection

        before = self.snapshot()
        with patch.object(self.repository, "_connect", side_effect=traced_connect):
            for name, args, context, reason in cases:
                with self.subTest(action=name, args=args, reason=reason):
                    outcome = self.run_action(name, args, context)
                    self.assertEqual((outcome.status, outcome.reason), ("blocked", reason))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(any(sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE"))
                             for sql in statements))
        with sqlite3.connect(self.repository.database_path) as connection:
            connection.execute("UPDATE tasks SET status = 'completed'")
            connection.execute("UPDATE reminders SET status = 'cancelled'")
        before = self.snapshot()
        statements.clear()
        with patch.object(self.repository, "_connect", side_effect=traced_connect):
            for kind in ("task", "reminder"):
                outcome = self.run_action("update_recent_" + kind, {"title": "New"})
                self.assertEqual(outcome.reason, "target_inactive")
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(any(sql.lstrip().upper().startswith("UPDATE") for sql in statements))
        # A blocked transaction released its lock: a subsequent valid write works.
        self.assertEqual(self.run_action("create_task", {"title": "Next"}).status, "executed")

    def test_execution_failure_is_not_a_policy_block(self):
        with patch.object(self.repository, "create_task", side_effect=sqlite3.OperationalError("failure")):
            with self.assertLogs("voice_inbox.execution", level="ERROR"):
                outcome = self.run_action("create_task", {"title": "T"})
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.effect, "unknown")
        self.assertEqual(self.events[-1]["outcome"]["reason"], "execution_error")
        self.assertEqual(self.repository.list_tasks(), [])

    def test_database_failure_after_update_rolls_back_transaction(self):
        self.seed()
        before = self.snapshot()
        with sqlite3.connect(self.repository.database_path) as connection:
            connection.execute("""CREATE TRIGGER fail_update AFTER UPDATE ON tasks
                                  BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
        with self.assertLogs("voice_inbox.execution", level="ERROR"):
            outcome = self.run_action("update_recent_task", {"title": "Changed"})
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(self.snapshot(), before)
        with sqlite3.connect(self.repository.database_path) as connection:
            connection.execute("DROP TRIGGER fail_update")
        self.assertEqual(self.run_action("update_recent_task", {"title": "Changed"}).status,
                         "executed")

    def test_recording_failure_does_not_disguise_committed_action(self):
        def fail_record(event):
            raise OSError("disk full")
        with self.assertLogs("voice_inbox.execution", level="ERROR"):
            outcome = execute_action("create_task", {"title": "T"}, self.context,
                                     self.repository, record=fail_record)
        self.assertEqual(outcome.status, "executed")
        self.assertEqual(outcome.recording_error, "recording_failed")
        self.assertEqual(len(self.repository.list_tasks()), 1)

    def test_jsonl_records_blocks_independently_of_turn_traces(self):
        path = Path(self.directory.name) / "actions.jsonl"
        outcome = execute_action("update_recent_task", {"title": "T"}, self.context,
                                 self.repository, record=TraceWriter(path).write_event)
        event = json.loads(path.read_text())
        self.assertEqual(event["outcome"], outcome.to_dict())
        self.assertEqual(event["event"], "action_finished")


class MutationAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_transcript_is_recorded_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = VoiceInboxRepository(Path(directory) / "test.db")
            repo.initialize()
            events = []
            state = agent.SessionState(repository=repo, action_recorder=events.append)
            context = SimpleNamespace(userdata=state)
            with patch("agent.source_transcript", side_effect=agent.ToolError("unavailable")):
                result = json.loads(await agent.create_task(context, title="T"))
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["reason"], "context_unavailable")
            self.assertEqual(events[0]["outcome"], result)
            self.assertEqual(repo.list_tasks(), [])
            self.assertEqual(state.recent_item_ids, {})

    async def test_all_mutation_adapters_use_executor_and_preserve_context(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = VoiceInboxRepository(Path(directory) / "test.db")
            repo.initialize()
            events = []
            state = agent.SessionState(repository=repo, source_transcript="Test command",
                                       action_recorder=events.append)
            state.transcript_ready.set()
            agent.start_turn(state, state.source_transcript, 1.0)
            turn_id = state.action_turn_id
            # Reproduce the existing early turn completion: action recording survives.
            state.active_turn = None
            context = SimpleNamespace(userdata=state, function_call=SimpleNamespace(call_id="call-1"))
            arguments = [
                ("create_task", {"title": "T"}), ("create_idea", {"text": "I"}),
                ("create_reminder", {"title": "R", "trigger_at": "2026-09-22T11:00:00Z"}),
                ("update_recent_task", {"title": "T2"}), ("update_recent_idea", {"text": "I2"}),
                ("update_recent_reminder", {"title": "R2"}),
            ]
            with patch("agent.execute_action", wraps=execute_action) as execute:
                for name, args in arguments:
                    outcome = json.loads(await getattr(agent, name)(context, **args))
                    self.assertEqual(outcome["status"], "executed")
                    self.assertEqual(outcome, events[-1]["outcome"])
                    self.assertEqual(events[-1]["turn_id"], turn_id)
                self.assertEqual(execute.call_count, len(ACTIONS))
            previous = dict(state.recent_item_ids)
            blocked = json.loads(await agent.create_task(context, title=""))
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(state.recent_item_ids, previous)
