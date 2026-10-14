from pathlib import Path
from typing import Any

import httpx
import pytest

from newsletter_agent.services.news_sources import (
    GOOGLE_NEWS_RSS_URL,
    GoogleNewsRssSource,
    TavilyNewsSource,
    parse_published_date,
    strip_html,
    strip_source_suffix,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "google_news_sample.xml"


class StubTavilyClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.received_arguments: list[dict[str, Any]] = []

    def search(self, **search_arguments: Any) -> dict[str, Any]:
        self.received_arguments.append(search_arguments)
        return self.response


def test_tavily_source_requests_recent_news_and_maps_results() -> None:
    search_client = StubTavilyClient(
        {
            "results": [
                {
                    "title": "Agents launch at scale",
                    "url": "https://www.techsite.com/agents",
                    "content": "Agents are <b>everywhere</b>.",
                    "published_date": "Mon, 12 Oct 2026 09:00:00 GMT",
                },
                {"title": "Missing URL is skipped", "url": ""},
            ]
        }
    )
    news_source = TavilyNewsSource(search_client, max_results=5)

    articles = news_source.search("AI agents", recency_days=7)

    assert search_client.received_arguments == [
        {"query": "AI agents", "topic": "news", "days": 7, "max_results": 5}
    ]
    assert len(articles) == 1
    assert articles[0].source_name == "techsite.com"
    assert articles[0].snippet == "Agents are everywhere ."
    assert articles[0].published_at is not None
    assert articles[0].published_at.year == 2026


def test_google_news_source_parses_feed_and_sends_recency_window() -> None:
    received_queries: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        received_queries.append(request.url.params["q"])
        return httpx.Response(200, text=FIXTURE_PATH.read_text(encoding="utf-8"))

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    news_source = GoogleNewsRssSource(http_client, max_results=1)

    articles = news_source.search("AI agents", recency_days=7)

    assert received_queries == ["AI agents when:7d"]
    assert len(articles) == 1
    assert articles[0].title == "OpenAI ships new agent toolkit"
    assert articles[0].source_name == "TechCrunch"
    assert articles[0].url == "https://news.google.com/rss/articles/abc123"
    assert "TechCrunch" in articles[0].snippet
    assert "<" not in articles[0].snippet


def test_google_news_source_raises_on_http_error() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    news_source = GoogleNewsRssSource(http_client, max_results=5)

    with pytest.raises(httpx.HTTPStatusError):
        news_source.search("AI agents", recency_days=7)


def test_google_news_rss_url_is_the_search_endpoint() -> None:
    assert GOOGLE_NEWS_RSS_URL == "https://news.google.com/rss/search"


@pytest.mark.parametrize(
    ("raw_date", "expected_day"),
    [("Mon, 12 Oct 2026 09:00:00 GMT", 12), ("2026-10-13T08:00:00Z", 13)],
)
def test_parse_published_date_accepts_rfc_and_iso_formats(raw_date: str, expected_day: int) -> None:
    parsed_date = parse_published_date(raw_date)

    assert parsed_date is not None
    assert parsed_date.day == expected_day


@pytest.mark.parametrize("raw_date", [None, "", "not a date"])
def test_parse_published_date_returns_none_for_unusable_input(raw_date: str | None) -> None:
    assert parse_published_date(raw_date) is None


def test_strip_html_removes_tags_and_entities() -> None:
    assert strip_html("<a href='x'>Hello</a>&nbsp;&amp; world") == "Hello & world"


def test_strip_source_suffix_only_removes_matching_suffix() -> None:
    assert strip_source_suffix("Big news - TechCrunch", "TechCrunch") == "Big news"
    assert strip_source_suffix("Big news - Wired", "TechCrunch") == "Big news - Wired"


def test_google_news_source_skips_entries_without_a_link() -> None:
    feed_without_first_link = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
<item><title>Linkless story - Example</title><source url="https://example.com">Example</source>
<description>No link here</description></item>
<item><title>Linked story - Example</title><link>https://news.google.com/rss/articles/linked</link>
<source url="https://example.com">Example</source><description>Has a link</description></item>
</channel></rss>"""
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=feed_without_first_link)
        )
    )
    news_source = GoogleNewsRssSource(http_client, max_results=1)

    articles = news_source.search("AI agents", recency_days=7)

    assert [article.url for article in articles] == ["https://news.google.com/rss/articles/linked"]
