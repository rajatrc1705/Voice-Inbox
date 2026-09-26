import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import agent
from voice_inbox.repository import VoiceInboxRepository


class AgentToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_creation_tools_persist_items_with_runtime_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = VoiceInboxRepository(
                Path(temporary_directory) / "voice-inbox.db"
            )
            repository.initialize()
            state = agent.SessionState(
                repository=repository,
                source_transcript=(
                    "I need to send the invoice, explore prefix caching, "
                    "and remind me tomorrow at eleven to call Alex."
                )
            )
            state.transcript_ready.set()
            context = SimpleNamespace(userdata=state)

            task_result = await agent.create_task(
                context,
                title="Send the invoice",
            )
            idea_result = await agent.create_idea(
                context,
                text="Investigate prefix caching",
            )
            reminder_result = await agent.create_reminder(
                context,
                title="Call Alex",
                trigger_at="2026-09-22T11:00:00+02:00",
            )

            tasks = repository.list_tasks()
            ideas = repository.list_ideas()
            reminders = repository.list_reminders()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].title, "Send the invoice")
        self.assertEqual(tasks[0].source_transcript, state.source_transcript)
        self.assertIn(tasks[0].id, task_result)
        self.assertEqual(ideas[0].text, "Investigate prefix caching")
        self.assertEqual(ideas[0].source_transcript, state.source_transcript)
        self.assertIn(ideas[0].id, idea_result)
        self.assertEqual(reminders[0].title, "Call Alex")
        self.assertEqual(reminders[0].source_transcript, state.source_transcript)
        self.assertEqual(
            reminders[0].trigger_at,
            datetime.fromisoformat("2026-09-22T11:00:00+02:00"),
        )
        self.assertIn(reminders[0].id, reminder_result)

    async def test_follow_up_corrections_reuse_created_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = VoiceInboxRepository(
                Path(temporary_directory) / "voice-inbox.db"
            )
            repository.initialize()
            state = agent.SessionState(
                repository=repository,
                source_transcript="Send an invoice, explore caching, and remind me to call Alex",
            )
            state.transcript_ready.set()
            context = SimpleNamespace(userdata=state)

            await agent.create_task(context, title="Send invoice")
            await agent.create_idea(context, text="Test caching")
            await agent.create_reminder(
                context, title="Call Alex", trigger_at="2026-09-22T11:00:00+02:00"
            )
            state.source_transcript = "Actually make that four in the afternoon"
            await agent.update_recent_task(context, title="Send Jetson invoice")
            await agent.update_recent_idea(context, text="Test prefix caching")
            result = await agent.update_recent_reminder(
                context, trigger_at="2026-09-22T16:00:00+02:00"
            )

            tasks = repository.list_tasks()
            ideas = repository.list_ideas()
            reminders = repository.list_reminders()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].title, "Send Jetson invoice")
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0].text, "Test prefix caching")
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0].title, "Call Alex")
        self.assertEqual(
            reminders[0].trigger_at,
            datetime.fromisoformat("2026-09-22T16:00:00+02:00"),
        )
        self.assertIn(reminders[0].id, result)

    async def test_correction_without_recent_item_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = VoiceInboxRepository(
                Path(temporary_directory) / "voice-inbox.db"
            )
            repository.initialize()
            context = SimpleNamespace(userdata=agent.SessionState(repository=repository))
            outcome = json.loads(await agent.update_recent_task(context, title="Send invoice"))
            self.assertEqual(outcome["status"], "blocked")
            self.assertEqual(outcome["reason"], "target_not_found")
            self.assertEqual(repository.list_tasks(), [])


if __name__ == "__main__":
    unittest.main()
