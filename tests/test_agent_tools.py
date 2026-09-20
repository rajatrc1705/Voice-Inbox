import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
from voice_inbox.repository import VoiceInboxRepository


class AgentToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_task_persists_the_task(self) -> None:
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
                result = await agent.create_task(
                    context,
                    title="Send the invoice",
                )

            tasks = repository.list_tasks()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].title, "Send the invoice")
        self.assertEqual(
            tasks[0].source_transcript,
            "I need to send the invoice.",
        )
        self.assertIn(tasks[0].id, result)


if __name__ == "__main__":
    unittest.main()
