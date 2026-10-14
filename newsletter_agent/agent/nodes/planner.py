"""Planner node: turns the goal into a structured research and writing plan."""

from collections.abc import Callable
from datetime import date
from typing import Any

from newsletter_agent.agent.node_names import PLANNER
from newsletter_agent.agent.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import MAX_RECENCY_DAYS, MIN_RECENCY_DAYS, Settings
from newsletter_agent.domain.models import NewsletterPlan, TraceEntry
from newsletter_agent.services.llm_gateway import LanguageModelGateway


def build_planner_node(
    language_model: LanguageModelGateway, settings: Settings
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the planner node."""

    def planner_node(state: NewsletterState) -> dict[str, Any]:
        user_prompt = PLANNER_USER_PROMPT.format(
            goal=state["goal"],
            today=date.today().isoformat(),
            default_recency_days=settings.default_recency_days,
            max_search_queries=settings.max_search_queries,
        )
        raw_plan = language_model.generate_structured(
            PLANNER_SYSTEM_PROMPT, user_prompt, NewsletterPlan
        )
        plan = normalize_plan(raw_plan, settings.max_search_queries)
        trace_message = (
            f"Planned '{plan.topic}' for {plan.audience}; searching the last "
            f"{plan.recency_days} days with: {', '.join(plan.search_queries)}"
        )
        return {
            "plan": plan,
            "revision_count": 0,
            "trace_log": [TraceEntry(step=PLANNER, message=trace_message)],
        }

    return planner_node


def normalize_plan(plan: NewsletterPlan, max_search_queries: int) -> NewsletterPlan:
    """Cap the number of queries, ensure at least one, and clamp the search window."""
    search_queries = [query for query in plan.search_queries if query.strip()]
    return plan.model_copy(
        update={
            "search_queries": search_queries[:max_search_queries] or [plan.topic],
            "recency_days": max(MIN_RECENCY_DAYS, min(plan.recency_days, MAX_RECENCY_DAYS)),
        }
    )
