import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from voice_inbox.models import ReminderStatus, TaskStatus
from voice_inbox.repository import VoiceInboxRepository


class VoiceInboxRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "voice-inbox.db"
        self.repository = VoiceInboxRepository(database_path)
        self.repository.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_created_items_survive_a_new_repository_instance(self) -> None:
        due_at = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
        trigger_at = datetime(2026, 9, 22, 11, 0, tzinfo=UTC)

        task = self.repository.create_task(
            title="Send Jetson invoice",
            details="Include the August usage breakdown",
            due_at=due_at,
            source_transcript="I need to send the Jetson invoice tomorrow morning.",
        )
        idea = self.repository.create_idea(
            text="Test tracker performance with different vehicle densities",
            source_transcript="I want to test tracker performance with different vehicle densities.",
        )
        reminder = self.repository.create_reminder(
            title="Call Shantanu",
            trigger_at=trigger_at,
            source_transcript="Remind me to call Shantanu on Monday at 11.",
        )

        reopened = VoiceInboxRepository(self.repository.database_path)
        reopened.initialize()

        self.assertEqual(reopened.list_tasks(), [task])
        self.assertEqual(reopened.list_ideas(), [idea])
        self.assertEqual(reopened.list_reminders(), [reminder])
        self.assertEqual(task.status, TaskStatus.OPEN)
        self.assertEqual(reminder.status, ReminderStatus.SCHEDULED)

if __name__ == "__main__":
    unittest.main()
