from pathlib import Path

from newsletter_agent.domain.models import DeliveryReport
from newsletter_agent.services.email_outbox import EmlFileOutbox
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer
from newsletter_agent.tools.content_tools import build_content_tools
from newsletter_agent.tools.delivery_tools import build_send_email_tool
from newsletter_agent.tools.search_tools import (
    NEWS_FEED_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    build_search_tools,
)
from tests.fakes import (
    TEST_SUBSCRIBERS,
    FakeArticleFetcher,
    FakeNewsSource,
    make_article,
    make_draft,
)


def test_search_tools_delegate_to_their_sources() -> None:
    web_source = FakeNewsSource("web_search", [make_article(1)])
    feed_source = FakeNewsSource("news_feed", [make_article(2)])

    web_search, news_feed = build_search_tools(web_source, feed_source)

    assert (web_search.name, news_feed.name) == (WEB_SEARCH_TOOL_NAME, NEWS_FEED_TOOL_NAME)
    assert web_search.invoke({"query": "agents", "recency_days": 3}) == [make_article(1)]
    assert news_feed.invoke({"query": "agents"}) == [make_article(2)]
    assert web_source.received_searches == [("agents", 3)]
    assert feed_source.received_searches == [("agents", 7)]


def test_search_tools_describe_themselves_for_the_model() -> None:
    web_search, news_feed = build_search_tools(
        FakeNewsSource("web_search"), FakeNewsSource("news_feed")
    )

    assert "web" in web_search.description.lower()
    assert "google news" in news_feed.description.lower()
    assert set(web_search.args) == {"query", "recency_days"}


def test_fetch_article_tool_returns_page_text() -> None:
    content_tools = build_content_tools(
        FakeArticleFetcher({"https://example.com/a": "Full text"}), NewsletterRenderer("Weekly")
    )

    assert content_tools.fetch_article.name == "fetch_article"
    assert content_tools.fetch_article.invoke({"url": "https://example.com/a"}) == "Full text"


def test_render_html_tool_renders_draft() -> None:
    content_tools = build_content_tools(FakeArticleFetcher(), NewsletterRenderer("Weekly"))
    draft = make_draft(article_count=2)

    rendered_html = content_tools.render_html.invoke({"draft": draft})

    assert content_tools.render_html.name == "render_html"
    assert draft.items[0].article.url in rendered_html


def test_send_email_tool_returns_delivery_report(tmp_path: Path) -> None:
    send_email = build_send_email_tool(EmlFileOutbox(tmp_path, "News <news@example.com>"))

    delivery_report = send_email.invoke(
        {"draft": make_draft(), "recipients": TEST_SUBSCRIBERS, "batch_name": "run-1"}
    )

    assert send_email.name == "send_email"
    assert isinstance(delivery_report, DeliveryReport)
    assert delivery_report.recipient_count == len(TEST_SUBSCRIBERS)
