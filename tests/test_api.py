import unittest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api import main
from voice_inbox.repository import VoiceInboxRepository


class ApiTest(unittest.IsolatedAsyncioTestCase):
    def test_health(self) -> None:
        self.assertEqual(main.health(), {"status": "ok"})

    def test_item_endpoints_return_persisted_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = VoiceInboxRepository(
                Path(temporary_directory) / "voice-inbox.db"
            )
            repository.initialize()
            task = repository.create_task(
                title="Send the invoice",
                source_transcript="I need to send the invoice.",
            )
            idea = repository.create_idea(
                text="Investigate prefix caching",
                source_transcript="I want to investigate prefix caching.",
            )
            reminder = repository.create_reminder(
                title="Call Shantanu",
                trigger_at=datetime.fromisoformat("2026-09-21T11:00:00+02:00"),
                source_transcript="Remind me to call Shantanu tomorrow at 11.",
            )

            with patch.object(main, "repository", repository):
                tasks = main.list_tasks()
                ideas = main.list_ideas()
                reminders = main.list_reminders()

        self.assertEqual(tasks, [task])
        self.assertEqual(ideas, [idea])
        self.assertEqual(reminders, [reminder])

    async def test_session_requires_livekit_configuration(self) -> None:
        with patch.multiple(
            main,
            LIVEKIT_URL=None,
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.create_session()

        self.assertEqual(raised.exception.status_code, 503)

    async def test_session_dispatches_agent_and_returns_token(self) -> None:
        token = MagicMock()
        token.with_identity.return_value = token
        token.with_grants.return_value = token
        token.to_jwt.return_value = "browser-token"

        livekit = MagicMock()
        livekit.agent_dispatch.create_dispatch = AsyncMock()
        livekit.aclose = AsyncMock()

        with (
            patch.multiple(
                main,
                LIVEKIT_URL="wss://livekit.example",
                LIVEKIT_API_KEY="key",
                LIVEKIT_API_SECRET="secret",
            ),
            patch.object(main, "AccessToken", return_value=token),
            patch.object(main, "LiveKitAPI", return_value=livekit),
        ):
            response = await main.create_session()

        self.assertEqual(
            response,
            {"livekit_url": "wss://livekit.example", "token": "browser-token"},
        )
        livekit.agent_dispatch.create_dispatch.assert_awaited_once()
        livekit.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
