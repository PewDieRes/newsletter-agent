"""Streamlit entry point for the Newsletter Agent."""

import logging
from collections.abc import Callable, Iterator
from typing import Any

import streamlit as st

from newsletter_agent.agent.interrupts import ARTICLE_REVIEW_INTERRUPT
from newsletter_agent.agent.runner import (
    AgentRunResult,
    NewsletterAgentRunner,
    get_default_runner,
    new_thread_id,
)
from newsletter_agent.config import DEFAULT_GOAL, Settings, get_settings
from newsletter_agent.domain.models import AgentMode, TraceEntry
from newsletter_agent.errors import NewsletterAgentError
from newsletter_agent.ui import components
from newsletter_agent.ui.run_quota import DailyRunQuota

UNEXPECTED_ERROR_MESSAGE = "Unexpected error — please try again."

SESSION_DEFAULT_FACTORIES: dict[str, Callable[[], Any]] = {
    "thread_id": lambda: None,
    "trace_entries": list,
    "result": lambda: None,
    "error_message": lambda: None,
    "runs_used": lambda: 0,
    "review_round": lambda: 0,
}


@st.cache_resource
def get_cached_runner() -> NewsletterAgentRunner:
    """Build the agent once per server process; runs are isolated by thread id."""
    return get_default_runner()


@st.cache_resource
def get_daily_run_quota() -> DailyRunQuota:
    """Share one daily run counter across every session in this server process."""
    return DailyRunQuota(get_settings().max_runs_per_day)


def initialize_session_state() -> None:
    """Create per-browser-session values on first load."""
    for state_key, default_factory in SESSION_DEFAULT_FACTORIES.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_factory()


def execute_stream(
    runner: NewsletterAgentRunner, trace_stream: Iterator[TraceEntry], thread_id: str
) -> None:
    """Show trace entries live while the graph runs, then store the result and rerun."""
    with st.status("Agent working...", expanded=True) as status_box:
        try:
            for trace_entry in trace_stream:
                st.session_state.trace_entries.append(trace_entry)
                status_box.write(components.format_trace_line(trace_entry))
        except NewsletterAgentError as agent_error:
            st.session_state.error_message = str(agent_error)
            status_box.update(label="Agent failed", state="error")
        except Exception:
            logging.getLogger(__name__).exception("Unexpected error while running the agent")
            st.session_state.error_message = UNEXPECTED_ERROR_MESSAGE
            status_box.update(label="Agent failed", state="error")
        else:
            status_box.update(label="Agent step finished", state="complete")
    st.session_state.result = runner.get_result(thread_id, st.session_state.error_message)
    st.rerun()


def start_run(goal: str, mode: AgentMode) -> None:
    """Begin a new run with a fresh thread, if today's demo quota allows it."""
    if not get_daily_run_quota().try_consume():
        st.session_state.error_message = components.DAILY_LIMIT_MESSAGE
        return
    runner = get_cached_runner()
    thread_id = new_thread_id()
    st.session_state.update(
        thread_id=thread_id, trace_entries=[], result=None, error_message=None, review_round=0
    )
    st.session_state.runs_used += 1
    execute_stream(runner, runner.stream_run(goal, mode, thread_id), thread_id)


def continue_run(human_decision: dict[str, Any]) -> None:
    """Resume the paused run with the human's decision."""
    runner = get_cached_runner()
    thread_id = st.session_state.thread_id
    st.session_state.review_round += 1
    execute_stream(runner, runner.stream_resume(thread_id, human_decision), thread_id)


def render_result(result: AgentRunResult, settings: Settings) -> None:
    """Render the review forms or the final outcome for the current run."""
    key_prefix = f"{result.thread_id}_{st.session_state.review_round}"
    if result.status == "awaiting_human" and result.pending_interrupt is not None:
        if result.pending_interrupt["type"] == ARTICLE_REVIEW_INTERRUPT:
            human_decision = components.render_article_review(result.pending_interrupt, key_prefix)
        else:
            human_decision = components.render_draft_review(result.pending_interrupt, key_prefix)
        if human_decision is not None:
            continue_run(dict(human_decision))
    elif result.status == "completed":
        components.render_delivery(result.final_state, settings.sender_address)
    elif result.status == "rejected":
        st.warning("The draft was rejected. Nothing was sent.")


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="Newsletter Agent", page_icon="📰", layout="wide")
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    settings = get_settings()
    initialize_session_state()
    st.title("📰 Newsletter Agent")
    st.caption(
        "An autonomous LangGraph agent that plans, researches, writes, critiques, and sends "
        "a weekly AI agents newsletter."
    )
    sidebar_input = components.render_sidebar(
        DEFAULT_GOAL,
        runs_remaining=settings.runs_per_session - st.session_state.runs_used,
        keys_configured=bool(settings.google_api_key and settings.tavily_api_key),
        daily_limit_reached=get_daily_run_quota().remaining() <= 0,
    )
    if sidebar_input.run_requested:
        start_run(sidebar_input.goal, sidebar_input.mode)
    if st.session_state.trace_entries:
        components.render_trace(st.session_state.trace_entries)
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    if st.session_state.result is None:
        if not st.session_state.trace_entries:
            st.info("Choose a mode in the sidebar and click **Run agent**.")
        return
    render_result(st.session_state.result, settings)


if __name__ == "__main__":
    main()
