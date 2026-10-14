"""Article review node: in human-in-the-loop mode, the human confirms the article selection."""

from collections.abc import Callable, Sequence
from typing import Any

from langgraph.types import interrupt

from newsletter_agent.agent.interrupts import (
    ARTICLE_REVIEW_INTERRUPT,
    ArticleReviewPayload,
    ArticleReviewResponse,
)
from newsletter_agent.agent.node_names import ARTICLE_REVIEW
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import Settings
from newsletter_agent.domain.models import Article, TraceEntry


def apply_article_review(
    candidates: Sequence[Article],
    agent_selection: Sequence[Article],
    approved_urls: Sequence[str],
    max_articles: int,
) -> tuple[list[Article], str]:
    """Resolve the human's URLs to articles; keep the agent's choice if none are valid."""
    articles_by_url = {article.url: article for article in candidates}
    approved_articles: list[Article] = []
    for approved_url in approved_urls:
        article = articles_by_url.get(approved_url)
        if article is not None and article not in approved_articles:
            approved_articles.append(article)
    if not approved_articles:
        return list(agent_selection), "Human approved no articles; keeping the agent's selection"
    capped_articles = approved_articles[:max_articles]
    return capped_articles, f"Human approved {len(capped_articles)} articles"


def build_article_review_node(settings: Settings) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the article review node."""

    def article_review_node(state: NewsletterState) -> dict[str, Any]:
        if state["mode"] != "hitl":
            return {
                "trace_log": [
                    TraceEntry(
                        step=ARTICLE_REVIEW,
                        message="Autonomous mode: article selection approved automatically",
                    )
                ]
            }
        candidates = state["candidate_articles"]
        agent_selection = state["selected_articles"]
        payload: ArticleReviewPayload = {
            "type": ARTICLE_REVIEW_INTERRUPT,
            "selected_urls": [article.url for article in agent_selection],
            "candidates": [article.model_dump(mode="json") for article in candidates],
        }
        human_response: ArticleReviewResponse = interrupt(payload)
        approved_articles, trace_message = apply_article_review(
            candidates,
            agent_selection,
            human_response.get("selected_urls", []),
            settings.max_articles,
        )
        return {
            "selected_articles": approved_articles,
            "trace_log": [TraceEntry(step=ARTICLE_REVIEW, message=trace_message)],
        }

    return article_review_node
