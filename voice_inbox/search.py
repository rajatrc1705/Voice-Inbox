import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 600


def search_tavily(query: str) -> list[dict[str, str]]:
    """Return a small ranked list of web results from Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured.")

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    body = json.dumps(
        {
            "query": query,
            "search_depth": "basic",
            "max_results": MAX_RESULTS,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
    ).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "VoiceInbox/0.1",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise ValueError(f"Web search returned HTTP {error.code}.") from error
    except URLError as error:
        raise ValueError(f"Could not search the web: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise ValueError("Web search returned an invalid response.") from error

    results = []
    for item in payload.get("results", [])[:MAX_RESULTS]:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        snippet = str(item.get("content", "")).strip()[:MAX_SNIPPET_CHARS]
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results
