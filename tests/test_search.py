import io
import json
import unittest
from unittest.mock import patch

from voice_inbox.search import MAX_RESULTS, search_tavily


class TavilySearchTest(unittest.TestCase):
    def test_returns_five_bounded_results(self) -> None:
        response = {
            "results": [
                {
                    "title": f"Result {index}",
                    "url": f"https://example.com/{index}",
                    "content": "Useful snippet " * 100,
                }
                for index in range(7)
            ]
        }
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch(
                "voice_inbox.search.urlopen",
                return_value=io.BytesIO(json.dumps(response).encode()),
            ) as open_url,
        ):
            results = search_tavily("current Python release")

        self.assertEqual(len(results), MAX_RESULTS)
        self.assertEqual(results[0]["url"], "https://example.com/0")
        self.assertLessEqual(len(results[0]["snippet"]), 600)
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["max_results"], 5)
        self.assertFalse(payload["include_answer"])
        self.assertFalse(payload["include_raw_content"])

    def test_requires_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                search_tavily("test")
