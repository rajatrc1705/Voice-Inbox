import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api import main
from voice_inbox.repository import VoiceInboxRepository


class ApiTest(unittest.IsolatedAsyncioTestCase):
    def test_health(self) -> None:
        self.assertEqual(main.health(), {"status": "ok"})

    def test_tasks_returns_persisted_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = VoiceInboxRepository(
                Path(temporary_directory) / "voice-inbox.db"
            )
            repository.initialize()
            task = repository.create_task(
                title="Send the invoice",
                source_transcript="I need to send the invoice.",
            )

            with patch.object(main, "repository", repository):
                response = main.list_tasks()

        self.assertEqual(response, [task])

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
