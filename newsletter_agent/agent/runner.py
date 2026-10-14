"""Public API of the newsletter agent: run, stream, and resume."""

import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from newsletter_agent.agent.dependencies import build_default_dependencies
from newsletter_agent.agent.graph import build_newsletter_graph
from newsletter_agent.config import DEFAULT_GOAL, Settings, get_settings
from newsletter_agent.domain.models import AgentMode, TraceEntry
from newsletter_agent.errors import NewsletterAgentError, RunTimeoutError

RunStatus = Literal["completed", "awaiting_human", "rejected", "failed"]
GRAPH_RECURSION_LIMIT = 50


@dataclass(frozen=True)
class AgentRunResult:
    """Where a run stands after the graph stops."""

    status: RunStatus
    thread_id: str
    pending_interrupt: dict[str, Any] | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


def new_thread_id() -> str:
    """Return a fresh, file-name-safe identifier for a run."""
    return uuid.uuid4().hex


class NewsletterAgentRunner:
    """Runs the compiled graph and translates its state into results the UI understands."""

    def __init__(self, graph: CompiledStateGraph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    def stream_run(self, goal: str, mode: AgentMode, thread_id: str) -> Iterator[TraceEntry]:
        """Start a run and yield trace entries until it finishes or pauses for a human."""
        initial_state = {
            "goal": goal,
            "mode": mode,
            "revision_count": 0,
            "critique_history": [],
            "trace_log": [],
        }
        yield from self._stream(initial_state, thread_id)

    def stream_resume(self, thread_id: str, human_decision: dict[str, Any]) -> Iterator[TraceEntry]:
        """Resume a paused run with the human's decision."""
        yield from self._stream(Command(resume=human_decision), thread_id)

    def get_result(self, thread_id: str, error_message: str | None = None) -> AgentRunResult:
        """Describe the run's current status from its latest checkpoint."""
        snapshot = self._graph.get_state(self._config(thread_id))
        final_state = dict(snapshot.values)
        if error_message is not None:
            return AgentRunResult("failed", thread_id, None, final_state, error_message)
        if snapshot.interrupts:
            return AgentRunResult(
                "awaiting_human", thread_id, snapshot.interrupts[0].value, final_state
            )
        if final_state.get("review_decision") == "reject":
            return AgentRunResult("rejected", thread_id, None, final_state)
        if final_state.get("delivery_report") is not None:
            return AgentRunResult("completed", thread_id, None, final_state)
        return AgentRunResult(
            "failed", thread_id, None, final_state, "Run ended before the newsletter was sent."
        )

    def _stream(self, graph_input: Any, thread_id: str) -> Iterator[TraceEntry]:
        started_at = time.monotonic()
        for graph_update in self._graph.stream(
            graph_input, self._config(thread_id), stream_mode="updates"
        ):
            for node_update in graph_update.values():
                if isinstance(node_update, dict):
                    yield from node_update.get("trace_log", [])
            if time.monotonic() - started_at > self._settings.run_timeout_seconds:
                raise RunTimeoutError(
                    f"Run timed out after {self._settings.run_timeout_seconds:.0f} seconds"
                )

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}


@lru_cache(maxsize=1)
def get_default_runner() -> NewsletterAgentRunner:
    """Build the production runner once per process."""
    settings = get_settings()
    graph = build_newsletter_graph(build_default_dependencies(settings), InMemorySaver())
    return NewsletterAgentRunner(graph, settings)


def _drain(
    active_runner: NewsletterAgentRunner, trace_stream: Iterator[TraceEntry], thread_id: str
) -> AgentRunResult:
    try:
        deque(trace_stream, maxlen=0)
    except NewsletterAgentError as agent_error:
        return active_runner.get_result(thread_id, error_message=str(agent_error))
    return active_runner.get_result(thread_id)


def run_newsletter_agent(
    goal: str = DEFAULT_GOAL,
    mode: AgentMode = "auto",
    thread_id: str | None = None,
    runner: NewsletterAgentRunner | None = None,
) -> AgentRunResult:
    """Run the whole agent with one call. In 'auto' mode this plans, researches, writes,
    critiques, and sends; in 'hitl' mode it returns 'awaiting_human' at each review step."""
    active_runner = runner or get_default_runner()
    run_thread_id = thread_id or new_thread_id()
    return _drain(active_runner, active_runner.stream_run(goal, mode, run_thread_id), run_thread_id)


def resume_newsletter_agent(
    thread_id: str,
    human_decision: dict[str, Any],
    runner: NewsletterAgentRunner | None = None,
) -> AgentRunResult:
    """Continue a paused human-in-the-loop run with the human's decision."""
    active_runner = runner or get_default_runner()
    return _drain(active_runner, active_runner.stream_resume(thread_id, human_decision), thread_id)
