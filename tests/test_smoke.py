import time

import pytest

from newsletter_agent.agent.runner import run_newsletter_agent
from newsletter_agent.config import DEFAULT_GOAL, get_settings

pytestmark = pytest.mark.smoke


def test_real_autonomous_run_sends_a_newsletter() -> None:
    settings = get_settings()
    if not (settings.google_api_key and settings.tavily_api_key):
        pytest.skip("GOOGLE_API_KEY and TAVILY_API_KEY are required for the smoke test")

    started_at = time.monotonic()
    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto")
    elapsed_seconds = time.monotonic() - started_at

    assert result.status == "completed", result.error_message
    draft = result.final_state["draft"]
    assert settings.min_articles <= len(draft.items) <= settings.max_articles
    assert all(item.article.url.startswith("http") for item in draft.items)
    print(f"\nSubject: {draft.subject}\nElapsed: {elapsed_seconds:.0f}s")
    print("Critique scores:", [report.score for report in result.final_state["critique_history"]])
