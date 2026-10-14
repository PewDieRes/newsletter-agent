"""Writer node: drafts (or revises) the newsletter and renders it."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from newsletter_agent.agent.node_names import WRITER
from newsletter_agent.agent.nodes.formatting import format_bullets, format_summary_list
from newsletter_agent.agent.prompts import WRITER_SYSTEM_PROMPT, WRITER_USER_PROMPT
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import Settings
from newsletter_agent.domain.llm_outputs import ArticleCopy, NewsletterCopy
from newsletter_agent.domain.models import (
    ArticleSummary,
    CritiqueReport,
    NewsletterDraft,
    TraceEntry,
)
from newsletter_agent.services.llm_gateway import LanguageModelGateway
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer


def build_revision_guidance(
    previous_draft: NewsletterDraft | None,
    latest_critique: CritiqueReport | None,
    human_feedback: str | None,
) -> str:
    """Explain to the writer whether this is a first draft or a revision, and why."""
    if previous_draft is None:
        return "This is the first draft."
    guidance_sections = ["Revise the previous draft below."]
    if human_feedback:
        guidance_sections.append(
            f"A human editor requested these changes (highest priority):\n{human_feedback}"
        )
    if latest_critique is not None:
        guidance_sections.append(
            f"Reviewer issues:\n{format_bullets(latest_critique.issues)}\n"
            f"Suggested fixes:\n{format_bullets(latest_critique.suggested_fixes)}"
        )
    guidance_sections.append(f"Previous draft:\n{previous_draft.markdown_body}")
    return "\n\n".join(guidance_sections)


def merge_article_copy(
    summaries: Sequence[ArticleSummary], article_copies: Sequence[ArticleCopy]
) -> list[ArticleSummary]:
    """Apply the writer's polished copy to each summary, keeping originals when copy is missing."""
    copies_by_number = {
        article_copy.article_number: article_copy for article_copy in article_copies
    }
    merged_summaries: list[ArticleSummary] = []
    for position, summary in enumerate(summaries, start=1):
        article_copy = copies_by_number.get(position)
        if article_copy is None:
            merged_summaries.append(summary)
        else:
            merged_summaries.append(
                summary.model_copy(
                    update={
                        "summary": article_copy.summary,
                        "why_it_matters": article_copy.why_it_matters,
                    }
                )
            )
    return merged_summaries


def describe_write(
    previous_draft: NewsletterDraft | None,
    human_feedback: str | None,
    revision_count: int,
    subject: str,
) -> str:
    """Return the trace message for this write."""
    if previous_draft is None:
        return f"Wrote first draft: '{subject}'"
    if human_feedback:
        return f"Rewrote draft from human feedback: '{subject}'"
    return f"Revision {revision_count} after critique: '{subject}'"


def build_writer_node(
    language_model: LanguageModelGateway,
    render_html_tool: BaseTool,
    renderer: NewsletterRenderer,
    settings: Settings,
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the writer node."""

    def writer_node(state: NewsletterState) -> dict[str, Any]:
        plan = state["plan"]
        summaries = state["article_summaries"]
        previous_draft = state.get("draft")
        human_feedback = state.get("human_feedback")
        critique_history = state.get("critique_history", [])
        latest_critique = critique_history[-1] if critique_history else None
        user_prompt = WRITER_USER_PROMPT.format(
            audience=plan.audience,
            tone=plan.tone,
            section_outline=", ".join(plan.section_outline),
            numbered_summaries=format_summary_list(summaries),
            revision_guidance=build_revision_guidance(
                previous_draft, latest_critique, human_feedback
            ),
        )
        newsletter_copy = language_model.generate_structured(
            WRITER_SYSTEM_PROMPT.format(newsletter_name=settings.newsletter_name),
            user_prompt,
            NewsletterCopy,
        )
        draft = NewsletterDraft(
            subject=newsletter_copy.subject,
            preheader=newsletter_copy.preheader,
            intro=newsletter_copy.intro,
            items=merge_article_copy(summaries, newsletter_copy.items),
            outro=newsletter_copy.outro,
        )
        draft = draft.model_copy(
            update={
                "html_body": render_html_tool.invoke({"draft": draft}),
                "markdown_body": renderer.render_markdown(draft),
            }
        )
        is_critique_revision = previous_draft is not None and not human_feedback
        revision_count = state.get("revision_count", 0) + int(is_critique_revision)
        trace_message = describe_write(
            previous_draft, human_feedback, revision_count, draft.subject
        )
        return {
            "draft": draft,
            "revision_count": revision_count,
            "human_feedback": None,
            "review_decision": None,
            "trace_log": [TraceEntry(step=WRITER, message=trace_message)],
        }

    return writer_node
