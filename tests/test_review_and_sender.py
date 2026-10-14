from pathlib import Path

import pytest

from newsletter_agent.agent.interrupts import DEFAULT_REVISION_FEEDBACK
from newsletter_agent.agent.nodes.draft_review import (
    apply_draft_review,
    apply_human_edits,
    build_draft_review_node,
)
from newsletter_agent.agent.nodes.sender import build_sender_node
from newsletter_agent.services.email_outbox import EmlFileOutbox
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer
from newsletter_agent.tools.delivery_tools import build_send_email_tool
from tests.fakes import TEST_SUBSCRIBERS, make_critique, make_draft

RENDERER = NewsletterRenderer("AI Agents Weekly")


def test_draft_review_auto_mode_approves() -> None:
    update = build_draft_review_node(RENDERER)(
        {"mode": "auto", "draft": make_draft(), "critique_history": [make_critique(9)]}
    )

    assert update["review_decision"] == "approve"
    assert update["trace_log"][0].step == "draft_review"


def test_approve_without_edits_keeps_draft() -> None:
    draft = make_draft()

    update = apply_draft_review(draft, {"decision": "approve"}, RENDERER)

    assert update["review_decision"] == "approve"
    assert update["draft"] == draft


def test_approve_with_markdown_edit_rerenders_html() -> None:
    draft = make_draft()

    update = apply_draft_review(
        draft,
        {
            "decision": "approve",
            "edited_subject": "New subject",
            "edited_markdown": "# Edited body",
        },
        RENDERER,
    )

    edited_draft = update["draft"]
    assert edited_draft.subject == "New subject"
    assert edited_draft.markdown_body == "# Edited body"
    assert "<h1>Edited body</h1>" in edited_draft.html_body
    assert "<title>New subject</title>" in edited_draft.html_body


def test_subject_only_edit_rerenders_structured_html() -> None:
    edited_draft = apply_human_edits(make_draft(), "Only subject", None, RENDERER)

    assert edited_draft.subject == "Only subject"
    assert "<title>Only subject</title>" in edited_draft.html_body
    assert "AI agent story 1" in edited_draft.html_body


def test_blank_subject_edit_keeps_original_subject() -> None:
    edited_draft = apply_human_edits(make_draft(), "   ", None, RENDERER)

    assert edited_draft.subject == "AI agents this week"


def test_revise_passes_feedback_to_writer() -> None:
    update = apply_draft_review(
        make_draft(), {"decision": "revise", "feedback": "Shorter intro"}, RENDERER
    )

    assert update["review_decision"] == "revise"
    assert update["human_feedback"] == "Shorter intro"


def test_blank_revision_feedback_uses_default_instruction() -> None:
    update = apply_draft_review(make_draft(), {"decision": "revise", "feedback": "  "}, RENDERER)

    assert update["human_feedback"] == DEFAULT_REVISION_FEEDBACK


def test_reject_stops_the_run() -> None:
    update = apply_draft_review(make_draft(), {"decision": "reject"}, RENDERER)

    assert update["review_decision"] == "reject"


def test_unknown_decision_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="maybe"):
        apply_draft_review(make_draft(), {"decision": "maybe"}, RENDERER)


def test_sender_uses_thread_id_as_batch_name(tmp_path: Path) -> None:
    send_email = build_send_email_tool(EmlFileOutbox(tmp_path, "News <news@example.com>"))
    sender_node = build_sender_node(send_email, TEST_SUBSCRIBERS)

    update = sender_node({"draft": make_draft()}, {"configurable": {"thread_id": "thread-123"}})

    assert update["delivery_report"].recipient_count == len(TEST_SUBSCRIBERS)
    assert len(list((tmp_path / "thread-123").glob("*.eml"))) == len(TEST_SUBSCRIBERS)
    assert "3 subscribers" in update["trace_log"][0].message
