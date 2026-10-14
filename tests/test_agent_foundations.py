import inspect

import pytest
from langgraph.errors import GraphInterrupt

from newsletter_agent.agent.node_names import PIPELINE_ORDER, PLANNER
from newsletter_agent.agent.nodes.formatting import (
    format_articles_with_content,
    format_bullets,
    format_candidate_list,
    format_summary_list,
)
from newsletter_agent.agent.nodes.guard import guard_node
from newsletter_agent.errors import NodeExecutionError
from tests.fakes import make_article, make_summary


def test_pipeline_order_lists_every_stage() -> None:
    assert PIPELINE_ORDER == (
        "planner",
        "researcher",
        "selector",
        "article_review",
        "summarizer",
        "writer",
        "critic",
        "draft_review",
        "sender",
    )


def test_guard_node_wraps_failures_with_step_name() -> None:
    def failing_node(state: dict) -> dict:
        raise ValueError("model exploded")

    guarded = guard_node(PLANNER, failing_node)

    with pytest.raises(NodeExecutionError) as raised:
        guarded({})
    assert raised.value.step_name == PLANNER
    assert isinstance(raised.value.original_error, ValueError)


def test_guard_node_lets_graph_interrupts_through() -> None:
    def interrupting_node(state: dict) -> dict:
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        guard_node(PLANNER, interrupting_node)({})


def test_guard_node_preserves_signature_and_result() -> None:
    def node_with_config(state: dict, config: dict) -> dict:
        return {"seen": config["value"]}

    guarded = guard_node(PLANNER, node_with_config)

    assert list(inspect.signature(guarded).parameters) == ["state", "config"]
    assert guarded({}, {"value": 3}) == {"seen": 3}


def test_format_bullets_handles_empty_lists() -> None:
    assert format_bullets(["one", "two"]) == "- one\n- two"
    assert format_bullets([]) == "- none"


def test_format_candidate_list_numbers_articles() -> None:
    formatted = format_candidate_list([make_article(1), make_article(2)])

    assert formatted.startswith("1. AI agent story 1 (Example News, 2026-10-12)")
    assert "2. AI agent story 2" in formatted
    assert format_candidate_list([]) == "No results."


def test_format_articles_with_content_prefers_content_and_truncates() -> None:
    article_with_content = make_article(1).model_copy(update={"content": "x" * 50})

    formatted = format_articles_with_content([article_with_content, make_article(2)], 10)

    assert "Text: xxxxxxxxxx\n" in formatted
    assert formatted.endswith("Text: Snippet ab")


def test_format_summary_list_includes_urls() -> None:
    formatted = format_summary_list([make_summary(1)])

    assert "https://news.example.com/story-1" in formatted
    assert "Why it matters: Story 1 matters." in formatted
