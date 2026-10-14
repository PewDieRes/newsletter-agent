"""Draft review node: in human-in-the-loop mode, the human approves, edits, revises, or rejects."""

from collections.abc import Callable
from typing import Any

from langgraph.types import interrupt

from newsletter_agent.agent.interrupts import (
    DEFAULT_REVISION_FEEDBACK,
    DRAFT_REVIEW_INTERRUPT,
    DraftReviewPayload,
    DraftReviewResponse,
)
from newsletter_agent.agent.node_names import DRAFT_REVIEW
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.domain.models import NewsletterDraft, TraceEntry
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer


def apply_human_edits(
    draft: NewsletterDraft,
    edited_subject: str | None,
    edited_markdown: str | None,
    renderer: NewsletterRenderer,
) -> NewsletterDraft:
    """Apply the human's subject and body edits and re-render the HTML and Markdown to match."""
    subject = (edited_subject or "").strip() or draft.subject
    markdown_changed = bool(edited_markdown and edited_markdown.strip()) and (
        edited_markdown.strip() != draft.markdown_body.strip()
    )
    if markdown_changed:
        return draft.model_copy(
            update={
                "subject": subject,
                "markdown_body": edited_markdown,
                "html_body": renderer.render_html_from_markdown(
                    subject, draft.preheader, edited_markdown
                ),
            }
        )
    if subject != draft.subject:
        retitled_draft = draft.model_copy(update={"subject": subject})
        return retitled_draft.model_copy(
            update={
                "html_body": renderer.render_html(retitled_draft),
                "markdown_body": renderer.render_markdown(retitled_draft),
            }
        )
    return draft


def apply_draft_review(
    draft: NewsletterDraft, human_response: DraftReviewResponse, renderer: NewsletterRenderer
) -> dict[str, Any]:
    """Turn the human's decision into a state update."""
    decision = human_response.get("decision")
    if decision == "reject":
        return {
            "review_decision": "reject",
            "trace_log": [TraceEntry(step=DRAFT_REVIEW, message="Human rejected the draft")],
        }
    if decision == "revise":
        feedback = (human_response.get("feedback") or "").strip() or DEFAULT_REVISION_FEEDBACK
        return {
            "review_decision": "revise",
            "human_feedback": feedback,
            "trace_log": [
                TraceEntry(step=DRAFT_REVIEW, message=f"Human requested a revision: {feedback}")
            ],
        }
    if decision == "approve":
        edited_draft = apply_human_edits(
            draft,
            human_response.get("edited_subject"),
            human_response.get("edited_markdown"),
            renderer,
        )
        edit_note = " with edits" if edited_draft != draft else ""
        return {
            "review_decision": "approve",
            "draft": edited_draft,
            "trace_log": [
                TraceEntry(step=DRAFT_REVIEW, message=f"Human approved the draft{edit_note}")
            ],
        }
    raise ValueError(f"Unknown review decision: {decision!r}")


def build_draft_review_node(
    renderer: NewsletterRenderer,
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the draft review node."""

    def draft_review_node(state: NewsletterState) -> dict[str, Any]:
        if state["mode"] != "hitl":
            return {
                "review_decision": "approve",
                "trace_log": [
                    TraceEntry(
                        step=DRAFT_REVIEW, message="Autonomous mode: draft approved automatically"
                    )
                ],
            }
        draft = state["draft"]
        payload: DraftReviewPayload = {
            "type": DRAFT_REVIEW_INTERRUPT,
            "draft": draft.model_dump(mode="json"),
            "critique": state["critique_history"][-1].model_dump(mode="json"),
        }
        human_response: DraftReviewResponse = interrupt(payload)
        return apply_draft_review(draft, human_response, renderer)

    return draft_review_node
