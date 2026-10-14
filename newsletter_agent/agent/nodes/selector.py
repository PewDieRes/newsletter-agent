"""Selector node: the model ranks candidates and picks the issue's articles."""

from collections.abc import Callable, Sequence
from typing import Any

from newsletter_agent.agent.node_names import SELECTOR
from newsletter_agent.agent.nodes.formatting import format_bullets, format_candidate_list
from newsletter_agent.agent.prompts import SELECTOR_SYSTEM_PROMPT, SELECTOR_USER_PROMPT
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import Settings
from newsletter_agent.domain.llm_outputs import ArticleSelection
from newsletter_agent.domain.models import Article, TraceEntry
from newsletter_agent.services.llm_gateway import LanguageModelGateway


def choose_articles(
    candidates: Sequence[Article], chosen_numbers: Sequence[int], minimum: int, maximum: int
) -> list[Article]:
    """Map 1-based numbers to candidates, drop invalid or repeated numbers, cap, then top up."""
    chosen_indexes: list[int] = []
    for chosen_number in chosen_numbers:
        candidate_index = chosen_number - 1
        if 0 <= candidate_index < len(candidates) and candidate_index not in chosen_indexes:
            chosen_indexes.append(candidate_index)
    chosen_indexes = chosen_indexes[:maximum]
    for candidate_index in range(len(candidates)):
        if len(chosen_indexes) >= minimum:
            break
        if candidate_index not in chosen_indexes:
            chosen_indexes.append(candidate_index)
    return [candidates[candidate_index] for candidate_index in chosen_indexes]


def build_selector_node(
    language_model: LanguageModelGateway, settings: Settings
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the selector node."""

    def selector_node(state: NewsletterState) -> dict[str, Any]:
        candidates = state["candidate_articles"]
        user_prompt = SELECTOR_USER_PROMPT.format(
            criteria=format_bullets(state["plan"].relevance_criteria),
            numbered_candidates=format_candidate_list(candidates),
            min_articles=settings.min_articles,
            max_articles=settings.max_articles,
        )
        selection = language_model.generate_structured(
            SELECTOR_SYSTEM_PROMPT, user_prompt, ArticleSelection
        )
        selected_articles = choose_articles(
            candidates,
            selection.selected_article_numbers,
            settings.min_articles,
            settings.max_articles,
        )
        trace_message = (
            f"Selected {len(selected_articles)} of {len(candidates)} candidates: "
            f"{selection.rationale}"
        )
        return {
            "selected_articles": selected_articles,
            "trace_log": [TraceEntry(step=SELECTOR, message=trace_message)],
        }

    return selector_node
