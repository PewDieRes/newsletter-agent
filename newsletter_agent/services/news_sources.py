"""News sources the agent can search. Each source returns domain Article objects."""

import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

import feedparser
import httpx

from newsletter_agent.domain.models import Article

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class NewsSource(Protocol):
    """Anything that can search for recent news articles."""

    name: str

    def search(self, query: str, recency_days: int) -> list[Article]:
        """Return recent articles matching the query."""
        ...


class TavilySearchClient(Protocol):
    """The subset of tavily.TavilyClient used by TavilyNewsSource."""

    def search(self, **search_arguments: Any) -> dict[str, Any]:
        """Run a Tavily search and return the raw response."""
        ...


def parse_published_date(raw_date: str | None) -> datetime | None:
    """Parse RFC 2822 or ISO 8601 dates; return None when the input is unusable."""
    if not raw_date:
        return None
    try:
        return parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        return None


def strip_html(raw_text: str) -> str:
    """Remove HTML tags and entities and collapse whitespace."""
    without_tags = HTML_TAG_PATTERN.sub(" ", raw_text)
    return " ".join(html.unescape(without_tags).split())


def strip_source_suffix(title: str, source_name: str) -> str:
    """Remove the trailing ' - Source' that Google News appends to titles."""
    suffix = f" - {source_name}"
    return title[: -len(suffix)] if source_name and title.endswith(suffix) else title


def domain_name(url: str) -> str:
    """Return the host name of a URL without a leading 'www.'."""
    return urlsplit(url).netloc.lower().removeprefix("www.")


class TavilyNewsSource:
    """Web news search through the Tavily API."""

    name = "web_search"

    def __init__(self, search_client: TavilySearchClient, max_results: int) -> None:
        self._search_client = search_client
        self._max_results = max_results

    def search(self, query: str, recency_days: int) -> list[Article]:
        """Search Tavily's news index for articles from the last `recency_days` days."""
        response = self._search_client.search(
            query=query, topic="news", days=recency_days, max_results=self._max_results
        )
        return [
            self._to_article(search_result)
            for search_result in response.get("results", [])
            if search_result.get("url")
        ]

    @staticmethod
    def _to_article(search_result: dict[str, Any]) -> Article:
        url = search_result["url"]
        return Article(
            title=search_result.get("title") or url,
            url=url,
            source_name=domain_name(url),
            published_at=parse_published_date(search_result.get("published_date")),
            snippet=strip_html(search_result.get("content") or ""),
        )


class GoogleNewsRssSource:
    """Headline search through the public Google News RSS feed (no API key)."""

    name = "news_feed"

    def __init__(self, http_client: httpx.Client, max_results: int) -> None:
        self._http_client = http_client
        self._max_results = max_results

    def search(self, query: str, recency_days: int) -> list[Article]:
        """Fetch Google News headlines matching the query from the last `recency_days` days."""
        response = self._http_client.get(
            GOOGLE_NEWS_RSS_URL,
            params={
                "q": f"{query} when:{recency_days}d",
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            },
        )
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        linked_entries = [feed_entry for feed_entry in feed.entries if feed_entry.get("link")]
        return [self._to_article(feed_entry) for feed_entry in linked_entries[: self._max_results]]

    @staticmethod
    def _to_article(feed_entry: Any) -> Article:
        source_name = feed_entry.get("source", {}).get("title", "Google News")
        return Article(
            title=strip_source_suffix(feed_entry.get("title", ""), source_name),
            url=feed_entry.get("link", ""),
            source_name=source_name,
            published_at=parse_published_date(feed_entry.get("published")),
            snippet=strip_html(feed_entry.get("summary", "")),
        )
