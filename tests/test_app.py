import base64
import logging
from pathlib import Path

from streamlit.testing.v1 import AppTest

from newsletter_agent.config import DEFAULT_GOAL
from newsletter_agent.domain.models import TraceEntry
from newsletter_agent.ui.components import (
    MODE_LABELS,
    email_preview_source,
    escape_markdown,
    format_trace_line,
)


def test_mode_labels_map_to_agent_modes() -> None:
    assert MODE_LABELS == {"Fully Autonomous": "auto", "Human-in-the-Loop": "hitl"}


def test_format_trace_line_shows_step_and_message() -> None:
    assert format_trace_line(TraceEntry(step="critic", message="Score 9/10")) == (
        "**critic** · Score 9/10"
    )


def test_escape_markdown_keeps_dollar_amounts_literal() -> None:
    assert escape_markdown("raises $40M at $400M") == r"raises \$40M at \$400M"


def test_escape_markdown_escapes_every_markdown_control_character() -> None:
    assert escape_markdown(r"\ $ * _ [ ] ` #") == r"\\ \$ \* \_ \[ \] \` \#"


def test_format_trace_line_escapes_the_message_but_keeps_the_bold_step() -> None:
    trace_entry = TraceEntry(step="researcher", message="web_search('$40M *deal*') returned 3")

    assert format_trace_line(trace_entry) == (
        r"**researcher** · web\_search('\$40M \*deal\*') returned 3"
    )


def test_app_renders_controls_without_running_the_agent() -> None:
    app_test = AppTest.from_file("../app.py", default_timeout=30).run()

    assert not app_test.exception
    assert app_test.sidebar.radio[0].options == ["Fully Autonomous", "Human-in-the-Loop"]
    assert app_test.sidebar.text_area[0].value == DEFAULT_GOAL
    assert app_test.sidebar.button[0].label == "Run agent"


def test_email_preview_source_is_isolated_data_url() -> None:
    html_body = "<html><body>Héllo <script>1</script></body></html>"

    preview_source = email_preview_source(html_body)

    prefix = "data:text/html;charset=utf-8;base64,"
    assert preview_source.startswith(prefix)
    assert base64.b64decode(preview_source[len(prefix) :]).decode("utf-8") == html_body


def test_render_article_review_shows_candidates() -> None:
    def review_script() -> None:
        from newsletter_agent.ui.components import render_article_review

        candidates = [
            {
                "title": f"Article {number} raises $40M",
                "url": f"https://example.com/{number}",
                "source_name": "Example_News",
                "snippet": "A snippet about $400M",
            }
            for number in (1, 2)
        ]
        render_article_review(
            {"selected_urls": ["https://example.com/1"], "candidates": candidates}, "test"
        )

    app_test = AppTest.from_function(review_script, default_timeout=30).run()

    assert not app_test.exception
    assert len(app_test.checkbox) == 2
    assert app_test.checkbox[0].label == r"**Article 1 raises \$40M** · Example\_News"
    assert app_test.caption[0].value.startswith(r"A snippet about \$400M")
    assert app_test.button[0].label == "Approve selection"


def test_render_draft_review_shows_controls() -> None:
    def review_script() -> None:
        from newsletter_agent.ui.components import render_draft_review

        draft = {
            "subject": "Weekly agents",
            "markdown_body": "# Body",
            "html_body": "<html><body>Body</body></html>",
        }
        render_draft_review({"draft": draft, "critique": {"score": 9, "issues": []}}, "test")

    app_test = AppTest.from_function(review_script, default_timeout=30).run()

    assert not app_test.exception
    assert app_test.text_input[0].value == "Weekly agents"
    button_labels = [button.label for button in app_test.button]
    assert button_labels == ["Approve & send", "Request revision", "Reject"]


def test_render_delivery_shows_summary(tmp_path: Path) -> None:
    eml_path = tmp_path / "newsletter.eml"
    eml_path.write_bytes(b"Subject: test\r\n\r\nbody")

    def delivery_script(eml_file: str) -> None:
        from newsletter_agent.domain.models import DeliveryReport
        from newsletter_agent.ui.components import render_delivery
        from tests.fakes import make_critique, make_draft

        draft = make_draft()
        report = DeliveryReport(
            recipient_count=3, subject=draft.subject, outbox_paths=[eml_file], html_path="x.html"
        )
        render_delivery(
            {
                "delivery_report": report,
                "draft": draft,
                "critique_history": [make_critique(6), make_critique(9)],
            },
            "agent@example.com",
        )

    app_test = AppTest.from_function(
        delivery_script, args=(str(eml_path),), default_timeout=30
    ).run()

    assert not app_test.exception
    assert "3 subscribers" in app_test.success[0].value


def test_sidebar_limits_goal_length() -> None:
    app_test = AppTest.from_file("../app.py", default_timeout=30).run()

    assert app_test.sidebar.text_area[0].max_chars == 500


def test_app_quiets_trafilatura_logging() -> None:
    AppTest.from_file("../app.py", default_timeout=30).run()

    assert logging.getLogger("trafilatura").level == logging.ERROR


def test_sidebar_disables_run_when_daily_limit_is_reached() -> None:
    def sidebar_script() -> None:
        from newsletter_agent.ui.components import render_sidebar

        render_sidebar("Goal", runs_remaining=3, keys_configured=True, daily_limit_reached=True)

    app_test = AppTest.from_function(sidebar_script, default_timeout=30).run()

    assert not app_test.exception
    assert app_test.sidebar.button[0].disabled
    captions = [caption.value for caption in app_test.sidebar.caption]
    assert "Daily demo limit reached — please try again tomorrow." in captions


def test_draft_review_limits_feedback_length() -> None:
    def review_script() -> None:
        from newsletter_agent.ui.components import render_draft_review

        draft = {"subject": "Weekly", "markdown_body": "# Body", "html_body": "<p>Body</p>"}
        render_draft_review({"draft": draft, "critique": {"score": 9, "issues": []}}, "test")

    app_test = AppTest.from_function(review_script, default_timeout=30).run()

    assert app_test.text_area(key="test_feedback").max_chars == 1000


def test_unexpected_stream_error_shows_a_generic_message() -> None:
    def stream_script() -> None:
        from collections.abc import Iterator

        import streamlit as st

        import app
        from newsletter_agent.agent.runner import AgentRunResult
        from newsletter_agent.domain.models import TraceEntry

        class FailingStreamRunner:
            def get_result(self, thread_id: str, error_message: str | None) -> AgentRunResult:
                return AgentRunResult("failed", thread_id, error_message=error_message)

        def failing_stream() -> Iterator[TraceEntry]:
            yield TraceEntry(step="planner", message="Planning")
            raise KeyError("internal detail")

        if "result" not in st.session_state:
            st.session_state.error_message = None
            st.session_state.trace_entries = []
            app.execute_stream(FailingStreamRunner(), failing_stream(), "thread")

    app_test = AppTest.from_function(stream_script, default_timeout=30).run()

    assert not app_test.exception
    assert app_test.session_state.error_message == "Unexpected error — please try again."
    assert app_test.session_state.result.status == "failed"
