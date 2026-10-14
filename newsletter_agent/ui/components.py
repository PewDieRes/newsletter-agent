"""Streamlit rendering functions. Presentation only: no agent logic lives here."""

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from newsletter_agent.agent.interrupts import ArticleReviewResponse, DraftReviewResponse
from newsletter_agent.domain.models import AgentMode, TraceEntry

MODE_LABELS: dict[str, AgentMode] = {"Fully Autonomous": "auto", "Human-in-the-Loop": "hitl"}
PREVIEW_HEIGHT_PIXELS = 720
MARKDOWN_CONTROL_CHARACTERS = frozenset("\\$*_[]`#")


@dataclass(frozen=True)
class SidebarInput:
    """What the user chose in the sidebar."""

    goal: str
    mode: AgentMode
    run_requested: bool


def render_sidebar(default_goal: str, runs_remaining: int, keys_configured: bool) -> SidebarInput:
    """Render the mode toggle, goal box, and run button."""
    with st.sidebar:
        st.header("Newsletter Agent")
        mode_label = st.radio(
            "Mode",
            list(MODE_LABELS),
            help="Human-in-the-Loop pauses for you to approve articles and the final draft.",
        )
        goal = st.text_area("Goal", value=default_goal, height=130)
        if not keys_configured:
            st.error("GOOGLE_API_KEY and TAVILY_API_KEY must be configured to run the agent.")
        run_requested = st.button(
            "Run agent",
            type="primary",
            disabled=runs_remaining <= 0 or not keys_configured or not goal.strip(),
        )
        st.caption(f"Runs remaining this session: {max(runs_remaining, 0)}")
    return SidebarInput(
        goal=goal.strip(), mode=MODE_LABELS[mode_label], run_requested=run_requested
    )


def email_preview_source(html_body: str) -> str:
    """Encode newsletter HTML as a data: URL so the preview iframe gets an opaque origin."""
    encoded_html = base64.b64encode(html_body.encode("utf-8")).decode("ascii")
    return f"data:text/html;charset=utf-8;base64,{encoded_html}"


def escape_markdown(text: str) -> str:
    """Escape Markdown control characters so web and model text renders literally.

    Streamlit would otherwise turn "$40M at $400M" into LaTeX, or a stray "*" into italics.
    """
    return "".join(
        f"\\{character}" if character in MARKDOWN_CONTROL_CHARACTERS else character
        for character in text
    )


def format_trace_line(trace_entry: TraceEntry) -> str:
    """Format one trace entry as a Markdown line with a bold step label."""
    return f"**{trace_entry.step}** · {escape_markdown(trace_entry.message)}"


def render_trace(trace_entries: Sequence[TraceEntry]) -> None:
    """Show the agent's step-by-step trace."""
    with st.expander("Agent trace: plan → research → write → review → send", expanded=True):
        for trace_entry in trace_entries:
            st.markdown(format_trace_line(trace_entry))


def render_article_review(payload: dict[str, Any], key_prefix: str) -> ArticleReviewResponse | None:
    """Let the human confirm which articles go into the issue."""
    st.subheader("Review step 1 of 2: choose the articles")
    agent_selection = set(payload["selected_urls"])
    ordered_candidates = sorted(
        payload["candidates"], key=lambda candidate: candidate["url"] not in agent_selection
    )
    with st.form(f"{key_prefix}_article_review"):
        kept_urls: list[str] = []
        for position, candidate in enumerate(ordered_candidates, start=1):
            is_kept = st.checkbox(
                f"**{escape_markdown(candidate['title'])}** · "
                f"{escape_markdown(candidate['source_name'])}",
                value=candidate["url"] in agent_selection,
                key=f"{key_prefix}_article_{position}",
            )
            st.caption(f"{escape_markdown(candidate['snippet'][:200])} [link]({candidate['url']})")
            if is_kept:
                kept_urls.append(candidate["url"])
        submitted = st.form_submit_button("Approve selection", type="primary")
    if not submitted:
        return None
    if not kept_urls:
        st.warning("Select at least one article.")
        return None
    return {"selected_urls": kept_urls}


def render_critique(critique: dict[str, Any]) -> None:
    """Show the critic's score and issues."""
    st.metric("Critique score", f"{critique['score']}/10")
    for issue in critique.get("issues", []):
        st.markdown(f"- {escape_markdown(issue)}")


def render_draft_review(payload: dict[str, Any], key_prefix: str) -> DraftReviewResponse | None:
    """Let the human approve (with optional edits), request a revision, or reject."""
    draft = payload["draft"]
    st.subheader("Review step 2 of 2: approve the newsletter")
    render_critique(payload["critique"])
    preview_column, edit_column = st.columns(2)
    with preview_column:
        # A data: URL gives the preview an opaque origin so it can never reach the app,
        # as defense in depth on top of the renderer's escaping/nh3 sanitization.
        st.iframe(email_preview_source(draft["html_body"]), height=PREVIEW_HEIGHT_PIXELS)
    with edit_column:
        edited_subject = st.text_input(
            "Subject", value=draft["subject"], key=f"{key_prefix}_subject"
        )
        edited_markdown = st.text_area(
            "Body (Markdown)", value=draft["markdown_body"], height=380, key=f"{key_prefix}_body"
        )
        feedback = st.text_area("Feedback for a rewrite", key=f"{key_prefix}_feedback")
        approve_column, revise_column, reject_column = st.columns(3)
        approve_clicked = approve_column.button(
            "Approve & send", type="primary", key=f"{key_prefix}_approve"
        )
        revise_clicked = revise_column.button("Request revision", key=f"{key_prefix}_revise")
        reject_clicked = reject_column.button("Reject", key=f"{key_prefix}_reject")
    if approve_clicked:
        return {
            "decision": "approve",
            "edited_subject": edited_subject,
            "edited_markdown": edited_markdown,
        }
    if revise_clicked:
        return {"decision": "revise", "feedback": feedback}
    if reject_clicked:
        return {"decision": "reject"}
    return None


def render_delivery(final_state: dict[str, Any], sender_address: str) -> None:
    """Show the simulated email and download buttons."""
    delivery_report = final_state["delivery_report"]
    draft = final_state["draft"]
    critique_scores = [str(report.score) for report in final_state.get("critique_history", [])]
    st.success(f"Newsletter sent (simulated) to {delivery_report.recipient_count} subscribers.")
    st.markdown(
        f"**From:** {sender_address}  \n"
        f"**To:** {delivery_report.recipient_count} subscribers  \n"
        f"**Subject:** {escape_markdown(draft.subject)}"
    )
    if critique_scores:
        st.caption(f"Critique scores across drafts: {' → '.join(critique_scores)}")
    # A data: URL gives the preview an opaque origin so it can never reach the app,
    # as defense in depth on top of the renderer's escaping/nh3 sanitization.
    st.iframe(email_preview_source(draft.html_body), height=PREVIEW_HEIGHT_PIXELS)
    download_column, eml_column = st.columns(2)
    download_column.download_button(
        "Download newsletter.html", draft.html_body, file_name="newsletter.html", mime="text/html"
    )
    if delivery_report.outbox_paths:
        eml_path = Path(delivery_report.outbox_paths[0])
        if eml_path.exists():
            eml_column.download_button(
                "Download newsletter.eml",
                eml_path.read_bytes(),
                file_name="newsletter.eml",
                mime="message/rfc822",
            )
