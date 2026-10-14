from streamlit.testing.v1 import AppTest

from newsletter_agent.config import DEFAULT_GOAL
from newsletter_agent.domain.models import TraceEntry
from newsletter_agent.ui.components import MODE_LABELS, format_trace_line


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
