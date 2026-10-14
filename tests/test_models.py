import pytest
from pydantic import ValidationError

from newsletter_agent.domain.models import Article, CritiqueReport, TraceEntry
from newsletter_agent.errors import NewsletterAgentError, NodeExecutionError


def test_article_defaults_are_empty_strings() -> None:
    article = Article(title="Agents ship", url="https://example.com/a", source_name="Example")

    assert article.snippet == ""
    assert article.content == ""
    assert article.published_at is None


@pytest.mark.parametrize("invalid_score", [0, 11])
def test_critique_score_must_be_between_one_and_ten(invalid_score: int) -> None:
    with pytest.raises(ValidationError):
        CritiqueReport(score=invalid_score, issues=[], suggested_fixes=[], passes=False)


def test_trace_entry_timestamp_is_timezone_aware() -> None:
    trace_entry = TraceEntry(step="planner", message="Planned")

    assert trace_entry.timestamp.tzinfo is not None


def test_node_execution_error_names_the_step() -> None:
    original_error = ValueError("bad output")

    node_error = NodeExecutionError("planner", original_error)

    assert isinstance(node_error, NewsletterAgentError)
    assert node_error.step_name == "planner"
    assert node_error.original_error is original_error
    assert "planner" in str(node_error)
    assert "bad output" in str(node_error)
