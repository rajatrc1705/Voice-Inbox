import ipaddress
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import trafilatura

MAX_HTML_BYTES = 1_000_000
MAX_TEXT_CHARS = 12_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def validate_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise ValueError("Invalid webpage URL.") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provide a full http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not supported.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError("Could not resolve the webpage host.") from error
    if not addresses or any(
        not ipaddress.ip_address(address[4][0]).is_global for address in addresses
    ):
        raise ValueError("Only public webpages can be opened.")


def extract_page_text(html: str, url: str) -> str:
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        raise ValueError("The webpage has no readable main text.")
    truncated = len(text) > MAX_TEXT_CHARS
    content = text[:MAX_TEXT_CHARS]
    if truncated:
        content += "\n[Page text truncated]"
    return f"Source: {url}\n{content}"


def read_webpage(url: str) -> str:
    """Fetch one public HTML page and return bounded main text for the agent."""
    opener = build_opener(_NoRedirect())
    current_url = url
    for _ in range(4):
        validate_public_url(current_url)
        request = Request(current_url, headers={"User-Agent": "VoiceInbox/0.1"})
        try:
            response = opener.open(request, timeout=8)
        except HTTPError as error:
            if error.code in {301, 302, 303, 307, 308} and error.headers.get("Location"):
                current_url = urljoin(current_url, error.headers["Location"])
                continue
            raise ValueError(f"Webpage returned HTTP {error.code}.") from error
        except URLError as error:
            raise ValueError(f"Could not open webpage: {error.reason}") from error

        with response:
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Only HTML webpages are supported.")
            html_bytes = response.read(MAX_HTML_BYTES + 1)
            if len(html_bytes) > MAX_HTML_BYTES:
                raise ValueError("Webpage HTML is too large to read.")
            charset = response.headers.get_content_charset() or "utf-8"
            html = html_bytes.decode(charset, errors="replace")

        return extract_page_text(html, current_url)
    raise ValueError("Webpage redirected too many times.")
