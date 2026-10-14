import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.serde import _msgpack as langgraph_msgpack
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener

from newsletter_agent.agent.checkpointing import build_checkpointer
from newsletter_agent.agent.graph import build_newsletter_graph
from newsletter_agent.agent.interrupts import ARTICLE_REVIEW_INTERRUPT, DRAFT_REVIEW_INTERRUPT
from newsletter_agent.agent.node_names import PIPELINE_ORDER
from newsletter_agent.agent.runner import (
    NewsletterAgentRunner,
    resume_newsletter_agent,
    run_newsletter_agent,
)
from newsletter_agent.config import DEFAULT_GOAL
from newsletter_agent.domain.llm_outputs import NewsletterCopy
from newsletter_agent.domain.models import (
    Article,
    ArticleSummary,
    CritiqueReport,
    DeliveryReport,
    NewsletterDraft,
    NewsletterPlan,
    Subscriber,
    TraceEntry,
)
from newsletter_agent.services.email_outbox import EmailSender
from tests.fakes import (
    FakeLanguageModel,
    FakeNewsSource,
    build_test_dependencies,
    build_test_runner,
    default_structured_responses,
    make_test_settings,
    search_both_tools_message,
)


def build_runner(tmp_path: Path, critique_scores=(9,), **dependency_overrides):
    language_model = FakeLanguageModel(
        default_structured_responses(critique_scores), [search_both_tools_message()]
    )
    dependencies = build_test_dependencies(tmp_path, language_model, **dependency_overrides)
    return build_test_runner(dependencies), language_model, dependencies


def first_positions(steps: list[str]) -> list[int]:
    return [steps.index(step_name) for step_name in PIPELINE_ORDER]


def test_autonomous_run_completes_in_one_call(tmp_path: Path) -> None:
    runner, _language_model, dependencies = build_runner(tmp_path)

    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto", runner=runner)

    assert result.status == "completed"
    delivery_report = result.final_state["delivery_report"]
    assert delivery_report.recipient_count == len(dependencies.subscribers)
    assert len(list((tmp_path / "outbox").rglob("*.eml"))) == len(dependencies.subscribers)
    steps = [entry.step for entry in result.final_state["trace_log"]]
    assert first_positions(steps) == sorted(first_positions(steps))
    assert len(result.final_state["draft"].items) == 5


def test_low_critique_score_triggers_one_revision(tmp_path: Path) -> None:
    runner, language_model, _dependencies = build_runner(tmp_path, critique_scores=(6, 9))

    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto", runner=runner)

    assert result.status == "completed"
    assert language_model.call_count(NewsletterCopy) == 2
    assert result.final_state["revision_count"] == 1
    assert [report.score for report in result.final_state["critique_history"]] == [6, 9]


def test_revisions_stop_at_maximum(tmp_path: Path) -> None:
    runner, language_model, _dependencies = build_runner(tmp_path, critique_scores=(5,))

    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto", runner=runner)

    assert result.status == "completed"
    assert language_model.call_count(NewsletterCopy) == 3
    assert result.final_state["revision_count"] == 2


def test_hitl_run_pauses_twice_and_applies_human_decisions(tmp_path: Path) -> None:
    runner, language_model, _dependencies = build_runner(tmp_path)
    thread_id = "hitl-thread"

    result = run_newsletter_agent(DEFAULT_GOAL, mode="hitl", thread_id=thread_id, runner=runner)
    assert result.status == "awaiting_human"
    assert result.pending_interrupt["type"] == ARTICLE_REVIEW_INTERRUPT
    chosen_urls = [candidate["url"] for candidate in result.pending_interrupt["candidates"][:3]]

    result = resume_newsletter_agent(thread_id, {"selected_urls": chosen_urls}, runner=runner)
    assert result.status == "awaiting_human"
    assert result.pending_interrupt["type"] == DRAFT_REVIEW_INTERRUPT
    draft_urls = [item["article"]["url"] for item in result.pending_interrupt["draft"]["items"]]
    assert draft_urls == chosen_urls

    result = resume_newsletter_agent(
        thread_id, {"decision": "revise", "feedback": "Shorter intro"}, runner=runner
    )
    assert result.status == "awaiting_human"
    assert language_model.call_count(NewsletterCopy) == 2
    assert "Shorter intro" in language_model.last_prompt(NewsletterCopy)

    result = resume_newsletter_agent(
        thread_id, {"decision": "approve", "edited_subject": "Edited subject"}, runner=runner
    )
    assert result.status == "completed"
    assert result.final_state["delivery_report"].subject == "Edited subject"
    assert result.final_state["revision_count"] == 0
    assert isinstance(result.final_state["selected_articles"][0], Article)
    assert isinstance(result.final_state["article_summaries"][0], ArticleSummary)


def test_hitl_reject_sends_nothing(tmp_path: Path) -> None:
    runner, _language_model, _dependencies = build_runner(tmp_path)
    thread_id = "reject-thread"
    run_newsletter_agent(DEFAULT_GOAL, mode="hitl", thread_id=thread_id, runner=runner)
    resume_newsletter_agent(thread_id, {"selected_urls": []}, runner=runner)

    result = resume_newsletter_agent(thread_id, {"decision": "reject"}, runner=runner)

    assert result.status == "rejected"
    assert not list((tmp_path / "outbox").rglob("*.eml"))


def test_research_failure_returns_failed_result_naming_the_step(tmp_path: Path) -> None:
    runner, _language_model, _dependencies = build_runner(
        tmp_path,
        web_search_source=FakeNewsSource("web_search", error=RuntimeError("service down")),
        news_feed_source=FakeNewsSource("news_feed", error=RuntimeError("service down")),
    )

    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto", runner=runner)

    assert result.status == "failed"
    assert "researcher" in result.error_message
    assert "service down" in result.error_message


def test_run_timeout_returns_failed_result(tmp_path: Path) -> None:
    runner, _language_model, _dependencies = build_runner(
        tmp_path, settings=make_test_settings(tmp_path, run_timeout_seconds=0.0)
    )

    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto", runner=runner)

    assert result.status == "failed"
    assert "timed out" in result.error_message


def test_stream_run_yields_trace_entries_as_they_happen(tmp_path: Path) -> None:
    runner, _language_model, _dependencies = build_runner(tmp_path)

    streamed_steps = [entry.step for entry in runner.stream_run(DEFAULT_GOAL, "auto", "stream")]

    assert streamed_steps[0] == "planner"
    assert streamed_steps[-1] == "sender"


class SteppingClock:
    """A fake clock that stays still until a test moves it forward."""

    def __init__(self) -> None:
        self.current_time = 0.0

    def __call__(self) -> float:
        return self.current_time


class ClockAdvancingSender:
    """Wraps an EmailSender and jumps the clock far past the deadline while sending."""

    def __init__(self, wrapped_sender: EmailSender, clock: SteppingClock) -> None:
        self._wrapped_sender = wrapped_sender
        self._clock = clock

    def send(
        self, draft: NewsletterDraft, recipients: Sequence[Subscriber], batch_name: str
    ) -> DeliveryReport:
        self._clock.current_time += 10_000.0
        return self._wrapped_sender.send(draft, recipients, batch_name)


def test_run_that_sends_after_the_deadline_is_still_completed(tmp_path: Path) -> None:
    _runner, _language_model, dependencies = build_runner(tmp_path)
    stepping_clock = SteppingClock()
    slow_dependencies = replace(
        dependencies, email_sender=ClockAdvancingSender(dependencies.email_sender, stepping_clock)
    )
    runner = NewsletterAgentRunner(
        build_newsletter_graph(slow_dependencies, build_checkpointer()),
        slow_dependencies.settings,
        clock=stepping_clock,
    )

    result = run_newsletter_agent(DEFAULT_GOAL, mode="auto", runner=runner)

    assert result.status == "completed"
    assert result.error_message is None


@pytest.mark.parametrize("strict_msgpack", [False, True], ids=["default", "strict"])
def test_checkpointed_domain_models_survive_resume_without_serde_warnings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    strict_msgpack: bool,
) -> None:
    # The strict flag is read once at import time, so patch the module value as well.
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true" if strict_msgpack else "false")
    monkeypatch.setattr(langgraph_msgpack, "STRICT_MSGPACK_ENABLED", strict_msgpack)
    serde_events: list[dict[str, Any]] = []
    unregister_listener = register_serde_event_listener(serde_events.append)
    runner, _language_model, _dependencies = build_runner(tmp_path)
    thread_id = f"checkpoint-thread-{strict_msgpack}"

    try:
        with caplog.at_level(logging.WARNING):
            result = run_newsletter_agent(
                DEFAULT_GOAL, mode="hitl", thread_id=thread_id, runner=runner
            )
            chosen_urls = [
                candidate["url"] for candidate in result.pending_interrupt["candidates"][:3]
            ]
            resume_newsletter_agent(thread_id, {"selected_urls": chosen_urls}, runner=runner)
            result = resume_newsletter_agent(thread_id, {"decision": "approve"}, runner=runner)
    finally:
        unregister_listener()

    assert result.status == "completed"
    assert serde_events == []
    assert "Deserializing unregistered type" not in caplog.text
    assert "Blocked deserialization" not in caplog.text
    final_state = result.final_state
    assert isinstance(final_state["plan"], NewsletterPlan)
    assert isinstance(final_state["candidate_articles"][0], Article)
    assert isinstance(final_state["article_summaries"][0], ArticleSummary)
    assert isinstance(final_state["draft"], NewsletterDraft)
    assert isinstance(final_state["critique_history"][0], CritiqueReport)
    assert isinstance(final_state["trace_log"][0], TraceEntry)
    assert isinstance(final_state["delivery_report"], DeliveryReport)
