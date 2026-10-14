"""Downloads article pages and extracts their readable text."""

from typing import Protocol

import httpx
import trafilatura


class ArticleFetcher(Protocol):
    """Anything that can turn a URL into readable article text."""

    def fetch(self, url: str) -> str:
        """Return the article's text, or an empty string if it cannot be fetched."""
        ...


class HttpArticleFetcher:
    """Fetches pages over HTTP and extracts the main text with trafilatura."""

    def __init__(self, http_client: httpx.Client, max_characters: int) -> None:
        self._http_client = http_client
        self._max_characters = max_characters

    def fetch(self, url: str) -> str:
        """Return up to `max_characters` of article text; never raises for network problems."""
        try:
            response = self._http_client.get(url)
            response.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL):
            return ""
        extracted_text = trafilatura.extract(response.text) or ""
        return extracted_text[: self._max_characters]
