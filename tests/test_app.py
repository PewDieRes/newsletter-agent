import base64
from pathlib import Path

from streamlit.testing.v1 import AppTest

from newsletter_agent.config import DEFAULT_GOAL
from newsletter_agent.domain.models import TraceEntry
from newsletter_agent.ui.components import MODE_LABELS, email_preview_source, format_trace_line


def test_mode_labels_map_to_agent_modes() -> None:
    assert MODE_LABELS == {"Fully Autonomous": "auto", "Human-in-the-Loop": "hitl"}


def test_format_trace_line_shows_step_and_message() -> None:
    assert format_trace_line(TraceEntry(step="critic", message="Score 9/10")) == (
        "**critic** · Score 9/10"
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
                "title": f"Article {number}",
                "url": f"https://example.com/{number}",
                "source_name": "Example",
                "snippet": "A snippet",
            }
            for number in (1, 2)
        ]
        render_article_review(
            {"selected_urls": ["https://example.com/1"], "candidates": candidates}, "test"
        )

    app_test = AppTest.from_function(review_script, default_timeout=30).run()

    assert not app_test.exception
    assert len(app_test.checkbox) == 2
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
