import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
from livekit.agents import ToolError

from voice_inbox.repository import VoiceInboxRepository
from voice_inbox.web import MAX_TEXT_CHARS, extract_page_text, validate_public_url


class WebTextTest(unittest.TestCase):
    def test_extracts_main_text_without_navigation_and_limits_output(self) -> None:
        html = (
            "<html><body><nav>Ignore this navigation</nav><article>"
            "<h1>ML Engineer</h1><p>Python and Kubernetes are required.</p>"
            f"<p>{'Relevant job detail. ' * 900}</p>"
            "</article></body></html>"
        )
        result = extract_page_text(html, "https://example.com/job")
        self.assertIn("Python and Kubernetes", result)
        self.assertNotIn("Ignore this navigation", result)
        self.assertIn("[Page text truncated]", result)
        self.assertLessEqual(len(result), MAX_TEXT_CHARS + 100)

    def test_blocks_private_addresses(self) -> None:
        with self.assertRaises(ValueError):
            validate_public_url("http://127.0.0.1/admin")
        with patch("voice_inbox.web.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("192.168.1.2", 443))
        ]):
            with self.assertRaises(ValueError):
                validate_public_url("https://private.example/resource")


class WebToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_page_tool_reads_only_supplied_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = VoiceInboxRepository(Path(directory) / "eval.db")
            repository.initialize()
            seen = []
            state = agent.SessionState(
                repository=repository,
                page_url="https://example.com/job",
                web_reader=lambda url: seen.append(url) or f"Source: {url}\nKubernetes required",
            )
            result = await agent.read_page(SimpleNamespace(userdata=state))
        self.assertEqual(seen, ["https://example.com/job"])
        self.assertIn("Kubernetes required", result)

    async def test_page_tool_requires_supplied_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = VoiceInboxRepository(Path(directory) / "eval.db")
            repository.initialize()
            state = agent.SessionState(repository=repository)
            with self.assertRaises(ToolError):
                await agent.read_page(SimpleNamespace(userdata=state))
