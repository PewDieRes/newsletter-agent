"""The shared state passed between graph nodes."""

import operator
from typing import Annotated, TypedDict

from newsletter_agent.domain.models import (
    AgentMode,
    Article,
    ArticleSummary,
    CritiqueReport,
    DeliveryReport,
    NewsletterDraft,
    NewsletterPlan,
    ReviewDecision,
    TraceEntry,
)


class NewsletterState(TypedDict, total=False):
    """Everything the agent knows during one run. List fields with reducers accumulate."""

    goal: str
    mode: AgentMode
    plan: NewsletterPlan
    candidate_articles: list[Article]
    selected_articles: list[Article]
    article_summaries: list[ArticleSummary]
    draft: NewsletterDraft
    critique_history: Annotated[list[CritiqueReport], operator.add]
    revision_count: int
    human_feedback: str | None
    review_decision: ReviewDecision | None
    trace_log: Annotated[list[TraceEntry], operator.add]
    delivery_report: DeliveryReport | None
