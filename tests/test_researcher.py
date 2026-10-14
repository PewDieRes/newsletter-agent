import pytest

from newsletter_agent.agent.nodes.researcher import build_researcher_node
from newsletter_agent.config import Settings
from newsletter_agent.errors import ResearchFailedError
from newsletter_agent.services.news_aggregator import NewsAggregator
from newsletter_agent.tools.search_tools import build_search_tools
from tests.fakes import (
    FakeLanguageModel,
    FakeNewsSource,
    make_article,
    make_plan,
    make_tool_call_message,
    search_both_tools_message,
)


def run_researcher(
    language_model: FakeLanguageModel,
    web_source: FakeNewsSource,
    feed_source: FakeNewsSource,
) -> dict:
    researcher_node = build_researcher_node(
        language_model,
        build_search_tools(web_source, feed_source),
        NewsAggregator([web_source, feed_source]),
        Settings(_env_file=None),
    )
    return researcher_node({"plan": make_plan(), "mode": "auto"})


def test_researcher_executes_model_tool_calls_and_merges_results() -> None:
    web_source = FakeNewsSource("web_search", [make_article(1), make_article(2), make_article(3)])
    feed_source = FakeNewsSource("news_feed", [make_article(3), make_article(4), make_article(5)])
    language_model = FakeLanguageModel(tool_responses=[search_both_tools_message()])

    update = run_researcher(language_model, web_source, feed_source)

    assert [article.url for article in update["candidate_articles"]] == [
        make_article(number).url for number in range(1, 6)
    ]
    assert web_source.received_searches == [("AI agents news", 7)]
    assert feed_source.received_searches == [("AI agents", 7)]
    trace_messages = [entry.message for entry in update["trace_log"]]
    assert "web_search('AI agents news') returned 3 results" in trace_messages
    assert "news_feed('AI agents') returned 3 results" in trace_messages
    second_model_call = language_model.tool_call_messages[1]
    assert second_model_call[-1].content.startswith("1. AI agent story 3")


def test_researcher_falls_back_to_all_sources_when_model_makes_no_calls() -> None:
    web_source = FakeNewsSource("web_search", [make_article(number) for number in range(1, 4)])
    feed_source = FakeNewsSource("news_feed", [make_article(number) for number in range(4, 7)])

    update = run_researcher(FakeLanguageModel(), web_source, feed_source)

    assert len(update["candidate_articles"]) == 6
    assert web_source.received_searches == [("AI agents news", 7), ("agentic AI launch", 7)]


def test_researcher_raises_when_every_source_fails() -> None:
    web_source = FakeNewsSource("web_search", error=RuntimeError("service down"))
    feed_source = FakeNewsSource("news_feed", error=RuntimeError("service down"))
    language_model = FakeLanguageModel(tool_responses=[search_both_tools_message()])

    with pytest.raises(ResearchFailedError, match="service down"):
        run_researcher(language_model, web_source, feed_source)


def test_researcher_widens_window_when_too_few_articles() -> None:
    web_source = FakeNewsSource("web_search", [make_article(1)])
    feed_source = FakeNewsSource("news_feed", [make_article(2)])
    language_model = FakeLanguageModel(tool_responses=[search_both_tools_message()])

    update = run_researcher(language_model, web_source, feed_source)

    assert ("AI agents news", 14) in web_source.received_searches
    assert len(update["candidate_articles"]) == 2
    assert any("widening" in entry.message for entry in update["trace_log"])


def test_researcher_stops_at_tool_call_budget() -> None:
    web_source = FakeNewsSource("web_search", [make_article(number) for number in range(1, 7)])
    feed_source = FakeNewsSource("news_feed")
    endless_calls = [
        make_tool_call_message(("web_search", f"query {number}"), ("web_search", "again"))
        for number in range(10)
    ]
    language_model = FakeLanguageModel(tool_responses=endless_calls)

    run_researcher(language_model, web_source, feed_source)

    assert len(web_source.received_searches) == 4
    assert len(language_model.tool_call_messages) == 2


def test_researcher_reports_unknown_tool_to_model() -> None:
    web_source = FakeNewsSource("web_search", [make_article(number) for number in range(1, 7)])
    language_model = FakeLanguageModel(
        tool_responses=[make_tool_call_message(("made_up_tool", "x"), ("web_search", "agents"))]
    )

    update = run_researcher(language_model, web_source, FakeNewsSource("news_feed"))

    assert any("unknown tool 'made_up_tool'" in entry.message for entry in update["trace_log"])
    assert len(update["candidate_articles"]) == 6


def test_researcher_queries_source_the_model_skipped() -> None:
    web_source = FakeNewsSource("web_search", [make_article(number) for number in range(1, 7)])
    feed_source = FakeNewsSource("news_feed", [make_article(7)])
    language_model = FakeLanguageModel(
        tool_responses=[make_tool_call_message(("web_search", "AI agents news"))]
    )

    update = run_researcher(language_model, web_source, feed_source)

    assert feed_source.received_searches == [("AI agents news", 7)]
    assert any("Model skipped news_feed" in entry.message for entry in update["trace_log"])
    assert len(update["candidate_articles"]) == 7


def test_researcher_survives_one_failing_source_on_tool_path() -> None:
    web_source = FakeNewsSource("web_search", error=RuntimeError("quota exceeded"))
    feed_source = FakeNewsSource("news_feed", [make_article(number) for number in range(1, 7)])
    language_model = FakeLanguageModel(tool_responses=[search_both_tools_message()])

    update = run_researcher(language_model, web_source, feed_source)

    assert len(update["candidate_articles"]) == 6
    assert any(
        "web_search('AI agents news') failed: quota exceeded" in entry.message
        for entry in update["trace_log"]
    )
