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

    def test_corrections_update_existing_items_without_duplicates(self) -> None:
        task = self.repository.create_task(
            title="Send invoice", source_transcript="Send invoice"
        )
        idea = self.repository.create_idea(
            text="Test caching", source_transcript="Test caching"
        )
        reminder = self.repository.create_reminder(
            title="Call Alex",
            trigger_at=datetime(2026, 9, 22, 11, tzinfo=UTC),
            source_transcript="Remind me to call Alex tomorrow at eleven",
        )

        updated_task = self.repository.update_task_title(task.id, "Send Jetson invoice")
        updated_idea = self.repository.update_idea_text(idea.id, "Test prefix caching")
        updated_reminder = self.repository.update_reminder(
            reminder.id, trigger_at=datetime(2026, 9, 22, 16, tzinfo=UTC)
        )

        reopened = VoiceInboxRepository(self.repository.database_path)
        self.assertEqual(reopened.list_tasks(), [updated_task])
        self.assertEqual(reopened.list_ideas(), [updated_idea])
        self.assertEqual(reopened.list_reminders(), [updated_reminder])
        self.assertEqual(updated_task.id, task.id)
        self.assertEqual(updated_idea.id, idea.id)
        self.assertEqual(updated_reminder.id, reminder.id)
        self.assertEqual(updated_reminder.title, "Call Alex")

    def test_update_rejects_unknown_item(self) -> None:
        with self.assertRaises(ValueError):
            self.repository.update_task_title("missing", "New title")
        with self.assertRaises(ValueError):
            self.repository.update_idea_text("missing", "New idea")
        with self.assertRaises(ValueError):
            self.repository.update_reminder(
                "missing", trigger_at=datetime(2026, 9, 22, 16, tzinfo=UTC)
            )

if __name__ == "__main__":
    unittest.main()
