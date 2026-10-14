"""Researcher node: an LLM-driven tool-calling loop over the news search tools."""

from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

from newsletter_agent.agent.node_names import RESEARCHER
from newsletter_agent.agent.nodes.formatting import format_bullets, format_candidate_list
from newsletter_agent.agent.prompts import RESEARCHER_SYSTEM_PROMPT, RESEARCHER_USER_PROMPT
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import MAX_RECENCY_DAYS, MIN_RECENCY_DAYS, Settings
from newsletter_agent.domain.models import Article, NewsletterPlan, TraceEntry
from newsletter_agent.errors import ResearchFailedError
from newsletter_agent.services.llm_gateway import LanguageModelGateway
from newsletter_agent.services.news_aggregator import (
    NewsAggregator,
    SourceFailure,
    deduplicate_articles,
)


def clamp_recency_days(tool_arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the tool arguments with a numeric recency_days kept in range."""
    recency_days = tool_arguments.get("recency_days")
    if isinstance(recency_days, bool) or not isinstance(recency_days, int | float):
        return dict(tool_arguments)
    clamped_days = max(MIN_RECENCY_DAYS, min(int(recency_days), MAX_RECENCY_DAYS))
    return {**tool_arguments, "recency_days": clamped_days}


class ResearchSession:
    """Collects articles, failures, and trace entries during one research run."""

    def __init__(self, tools_by_name: dict[str, BaseTool]) -> None:
        self._tools_by_name = tools_by_name
        self.article_batches: list[list[Article]] = []
        self.failures: list[SourceFailure] = []
        self.trace_entries: list[TraceEntry] = []
        self.tool_calls_made = 0
        self.called_tool_names: set[str] = set()

    @property
    def articles(self) -> list[Article]:
        """All articles found so far, de-duplicated."""
        return deduplicate_articles(chain.from_iterable(self.article_batches))

    def log(self, message: str) -> None:
        """Add a researcher trace entry."""
        self.trace_entries.append(TraceEntry(step=RESEARCHER, message=message))

    def execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        """Run one tool call requested by the model and return the result for the model."""
        self.tool_calls_made += 1
        tool_name = tool_call["name"]
        tool_call_id = tool_call.get("id") or tool_name
        query = str(tool_call["args"].get("query", ""))
        tool_arguments = clamp_recency_days(tool_call["args"])
        selected_tool = self._tools_by_name.get(tool_name)
        if selected_tool is None:
            self.log(f"Model requested unknown tool '{tool_name}'")
            return ToolMessage(
                content=f"Unknown tool '{tool_name}'.", tool_call_id=tool_call_id, name=tool_name
            )
        self.called_tool_names.add(tool_name)
        try:
            found_articles: list[Article] = selected_tool.invoke(tool_arguments)
        except Exception as tool_error:
            self.failures.append(
                SourceFailure(source_name=tool_name, query=query, error_message=str(tool_error))
            )
            self.log(f"{tool_name}('{query}') failed: {tool_error}")
            return ToolMessage(
                content=f"Tool error: {tool_error}", tool_call_id=tool_call_id, name=tool_name
            )
        self.article_batches.append(found_articles)
        self.log(f"{tool_name}('{query}') returned {len(found_articles)} results")
        return ToolMessage(
            content=format_candidate_list(found_articles), tool_call_id=tool_call_id, name=tool_name
        )


def build_researcher_node(
    language_model: LanguageModelGateway,
    search_tools: Sequence[BaseTool],
    aggregator: NewsAggregator,
    settings: Settings,
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the researcher node."""
    tools_by_name = {search_tool.name: search_tool for search_tool in search_tools}

    def researcher_node(state: NewsletterState) -> dict[str, Any]:
        plan = state["plan"]
        session = ResearchSession(tools_by_name)
        run_tool_calling_loop(
            language_model, search_tools, plan, session, settings.max_research_tool_calls
        )
        query_skipped_sources(session, tools_by_name, plan)
        articles = ensure_articles(session, aggregator, plan)
        articles = widen_if_too_few(articles, session, aggregator, plan, settings)
        if not articles:
            raise ResearchFailedError("No articles were found for any query.")
        session.log(f"Collected {len(articles)} unique candidate articles")
        return {"candidate_articles": articles, "trace_log": session.trace_entries}

    return researcher_node


def query_skipped_sources(
    session: ResearchSession, tools_by_name: dict[str, BaseTool], plan: NewsletterPlan
) -> None:
    """Query each source the model never called, so every source contributes when the model made tool calls."""
    if session.tool_calls_made == 0:
        return
    for tool_name in tools_by_name:
        if tool_name in session.called_tool_names:
            continue
        session.log(
            f"Model skipped {tool_name}; querying it directly with '{plan.search_queries[0]}'"
        )
        session.execute_tool_call(
            {
                "name": tool_name,
                "args": {"query": plan.search_queries[0], "recency_days": plan.recency_days},
                "id": f"direct_{tool_name}",
            }
        )


def run_tool_calling_loop(
    language_model: LanguageModelGateway,
    search_tools: Sequence[BaseTool],
    plan: NewsletterPlan,
    session: ResearchSession,
    max_tool_calls: int,
) -> None:
    """Let the model call search tools until it stops or the budget is spent."""
    messages: list[BaseMessage] = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls)),
        HumanMessage(
            content=RESEARCHER_USER_PROMPT.format(
                topic=plan.topic,
                queries=format_bullets(plan.search_queries),
                recency_days=plan.recency_days,
                criteria=format_bullets(plan.relevance_criteria),
            )
        ),
    ]
    while session.tool_calls_made < max_tool_calls:
        model_reply = language_model.invoke_with_tools(messages, search_tools)
        if not model_reply.tool_calls:
            break
        messages.append(model_reply)
        remaining_budget = max_tool_calls - session.tool_calls_made
        for tool_call in model_reply.tool_calls[:remaining_budget]:
            messages.append(session.execute_tool_call(tool_call))
        skipped_calls = len(model_reply.tool_calls) - remaining_budget
        if skipped_calls > 0:
            session.log(f"Tool budget reached; skipped {skipped_calls} extra tool calls")


def search_all_sources(
    session: ResearchSession,
    aggregator: NewsAggregator,
    queries: Sequence[str],
    recency_days: int,
) -> list[Article]:
    """Run every query on every source directly, recording failures in the session."""
    aggregated_results = aggregator.search_all(queries, recency_days)
    session.failures.extend(aggregated_results.failures)
    for failure in aggregated_results.failures:
        session.log(f"{failure.source_name}('{failure.query}') failed: {failure.error_message}")
    return aggregated_results.articles


def ensure_articles(
    session: ResearchSession, aggregator: NewsAggregator, plan: NewsletterPlan
) -> list[Article]:
    """Return the model's results, or fall back to searching every source directly."""
    articles = session.articles
    if articles:
        return articles
    session.log("Tool calls produced no articles; searching every source with the planned queries")
    return search_all_sources(session, aggregator, plan.search_queries, plan.recency_days)


def widen_if_too_few(
    articles: list[Article],
    session: ResearchSession,
    aggregator: NewsAggregator,
    plan: NewsletterPlan,
    settings: Settings,
) -> list[Article]:
    """Retry once with a wider window when fewer than the minimum articles were found."""
    if len(articles) >= settings.min_articles:
        return articles
    session.log(
        f"Only {len(articles)} articles found; widening search window to "
        f"{settings.widened_recency_days} days"
    )
    try:
        widened_articles = search_all_sources(
            session, aggregator, plan.search_queries, settings.widened_recency_days
        )
    except ResearchFailedError as research_error:
        session.log(f"Widened search failed: {research_error}")
        return articles
    return deduplicate_articles([*articles, *widened_articles])
