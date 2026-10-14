import pytest

from newsletter_agent.domain.models import Article
from newsletter_agent.errors import ResearchFailedError
from newsletter_agent.services.news_aggregator import (
    NewsAggregator,
    deduplicate_articles,
    normalize_title,
    normalize_url,
)
from tests.fakes import FakeNewsSource, make_article


def test_search_all_queries_every_source_with_every_query() -> None:
    web_source = FakeNewsSource("web_search", [make_article(1)])
    feed_source = FakeNewsSource("news_feed", [make_article(2)])
    aggregator = NewsAggregator([web_source, feed_source])

    results = aggregator.search_all(["agents", "agentic AI"], recency_days=7)

    assert web_source.received_searches == [("agents", 7), ("agentic AI", 7)]
    assert feed_source.received_searches == [("agents", 7), ("agentic AI", 7)]
    assert [article.url for article in results.articles] == [
        make_article(1).url,
        make_article(2).url,
    ]
    assert results.failures == []


def test_search_all_survives_one_failing_source() -> None:
    web_source = FakeNewsSource("web_search", error=RuntimeError("quota exceeded"))
    feed_source = FakeNewsSource("news_feed", [make_article(2)])
    aggregator = NewsAggregator([web_source, feed_source])

    results = aggregator.search_all(["agents"], recency_days=7)

    assert [article.url for article in results.articles] == [make_article(2).url]
    assert results.failures[0].source_name == "web_search"
    assert "quota exceeded" in results.failures[0].error_message


def test_search_all_raises_when_every_source_fails() -> None:
    aggregator = NewsAggregator(
        [
            FakeNewsSource("web_search", error=RuntimeError("down")),
            FakeNewsSource("news_feed", error=RuntimeError("down")),
        ]
    )

    with pytest.raises(ResearchFailedError, match="down"):
        aggregator.search_all(["agents"], recency_days=7)


def test_search_all_returns_empty_results_when_nothing_found_without_errors() -> None:
    aggregator = NewsAggregator([FakeNewsSource("web_search"), FakeNewsSource("news_feed")])

    results = aggregator.search_all(["agents"], recency_days=7)

    assert results.articles == []
    assert results.failures == []


def test_deduplicate_removes_same_url_variants() -> None:
    first = make_article(1, url="https://www.example.com/story/")
    duplicate = make_article(2, url="http://example.com/story")

    assert deduplicate_articles([first, duplicate]) == [first]


def test_deduplicate_prefers_publisher_url_over_google_news_link() -> None:
    google_copy = Article(
        title="OpenAI ships new agent toolkit",
        url="https://news.google.com/rss/articles/abc123",
        source_name="TechCrunch",
    )
    publisher_copy = Article(
        title="OpenAI Ships New Agent Toolkit!",
        url="https://techcrunch.com/agents",
        source_name="techcrunch.com",
    )

    assert deduplicate_articles([google_copy, publisher_copy]) == [publisher_copy]


def test_normalizers() -> None:
    assert normalize_title("  OpenAI Ships: New Agent-Toolkit! ") == "openai ships new agenttoolkit"
    assert normalize_url("https://www.Example.com/a/b/?utm=1") == "example.com/a/b"
    assert normalize_url("not a url") == ""
