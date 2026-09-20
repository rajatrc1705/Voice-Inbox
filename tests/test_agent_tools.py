import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
                source_transcript="I need to send the invoice."
            )
            state.transcript_ready.set()
            context = SimpleNamespace(userdata=state)

            with patch.object(agent, "repository", repository):
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
                    title="Call Shantanu",
                    trigger_at="2026-09-21T11:00:00+02:00",
                )

            tasks = repository.list_tasks()
            ideas = repository.list_ideas()
            reminders = repository.list_reminders()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].title, "Send the invoice")
        self.assertEqual(
            tasks[0].source_transcript,
            "I need to send the invoice.",
        )
        self.assertIn(tasks[0].id, task_result)
        self.assertEqual(ideas[0].text, "Investigate prefix caching")
        self.assertIn(ideas[0].id, idea_result)
        self.assertEqual(reminders[0].title, "Call Shantanu")
        self.assertEqual(
            reminders[0].trigger_at,
            datetime.fromisoformat("2026-09-21T11:00:00+02:00"),
        )
        self.assertIn(reminders[0].id, reminder_result)


if __name__ == "__main__":
    unittest.main()
