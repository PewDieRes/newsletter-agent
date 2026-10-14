"""The contract between human-in-the-loop nodes and the front ends that answer them."""

from typing import Any, Literal, NotRequired, TypedDict

from newsletter_agent.domain.models import ReviewDecision

ARTICLE_REVIEW_INTERRUPT = "article_review"
DRAFT_REVIEW_INTERRUPT = "draft_review"
DEFAULT_REVISION_FEEDBACK = "Improve the overall quality of the newsletter based on the critique."


class ArticleReviewPayload(TypedDict):
    """Sent to the human when the agent pauses to confirm article selection."""

    type: Literal["article_review"]
    selected_urls: list[str]
    candidates: list[dict[str, Any]]


class ArticleReviewResponse(TypedDict):
    """The human's final article selection, as URLs in display order."""

    selected_urls: list[str]


class DraftReviewPayload(TypedDict):
    """Sent to the human when the agent pauses before sending."""

    type: Literal["draft_review"]
    draft: dict[str, Any]
    critique: dict[str, Any]


class DraftReviewResponse(TypedDict):
    """The human's decision on the draft, with optional edits or feedback."""

    decision: ReviewDecision
    edited_subject: NotRequired[str]
    edited_markdown: NotRequired[str]
    feedback: NotRequired[str]
