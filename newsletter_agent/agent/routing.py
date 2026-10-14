"""Conditional edges of the newsletter graph."""

from collections.abc import Callable

from langgraph.graph import END

from newsletter_agent.agent.node_names import DRAFT_REVIEW, SENDER, WRITER
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import Settings


def build_route_after_critique(settings: Settings) -> Callable[[NewsletterState], str]:
    """Revise while the score is below threshold and automatic revisions remain."""

    def route_after_critique(state: NewsletterState) -> str:
        latest_critique = state["critique_history"][-1]
        needs_revision = latest_critique.score < settings.quality_threshold
        revisions_remaining = state.get("revision_count", 0) < settings.max_revisions
        return WRITER if needs_revision and revisions_remaining else DRAFT_REVIEW

    return route_after_critique


def route_after_draft_review(state: NewsletterState) -> str:
    """Send on approval, rewrite on revision request, stop on rejection."""
    review_decision = state.get("review_decision")
    if review_decision == "revise":
        return WRITER
    if review_decision == "reject":
        return END
    return SENDER
