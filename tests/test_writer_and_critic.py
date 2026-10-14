from langgraph.graph import END

from newsletter_agent.agent.node_names import DRAFT_REVIEW, SENDER, WRITER
from newsletter_agent.agent.nodes.critic import build_critic_node
from newsletter_agent.agent.nodes.writer import (
    build_revision_guidance,
    build_writer_node,
    merge_article_copy,
)
from newsletter_agent.agent.routing import build_route_after_critique, route_after_draft_review
from newsletter_agent.config import Settings
from newsletter_agent.domain.llm_outputs import ArticleCopy, NewsletterCopy
from newsletter_agent.domain.models import CritiqueReport
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer
from newsletter_agent.tools.content_tools import build_content_tools
from tests.fakes import (
    FakeArticleFetcher,
    FakeLanguageModel,
    make_critique,
    make_draft,
    make_newsletter_copy,
    make_plan,
    make_summary,
)

SUMMARIES = [make_summary(number) for number in range(1, 6)]
SETTINGS = Settings(_env_file=None)


def build_writer(language_model: FakeLanguageModel):
    renderer = NewsletterRenderer("AI Agents Weekly")
    content_tools = build_content_tools(FakeArticleFetcher(), renderer)
    return build_writer_node(language_model, content_tools.render_html, renderer, SETTINGS)


def writer_state(**overrides) -> dict:
    return {"plan": make_plan(), "article_summaries": SUMMARIES, **overrides}


def test_writer_first_draft_renders_html_and_markdown() -> None:
    language_model = FakeLanguageModel({NewsletterCopy: [make_newsletter_copy(5)]})

    update = build_writer(language_model)(writer_state())

    draft = update["draft"]
    assert draft.subject == "Agents everywhere: this week's top stories"
    assert draft.items[0].summary == "Polished summary 1."
    assert draft.items[0].article.url == SUMMARIES[0].article.url
    assert SUMMARIES[0].article.url in draft.html_body
    assert draft.markdown_body.startswith("# Agents everywhere")
    assert update["revision_count"] == 0
    assert update["human_feedback"] is None
    assert "This is the first draft." in language_model.last_prompt(NewsletterCopy)


def test_writer_critique_revision_increments_count_and_uses_issues() -> None:
    language_model = FakeLanguageModel({NewsletterCopy: [make_newsletter_copy(5)]})
    state = writer_state(draft=make_draft(), critique_history=[make_critique(5)], revision_count=0)

    update = build_writer(language_model)(state)

    assert update["revision_count"] == 1
    prompt = language_model.last_prompt(NewsletterCopy)
    assert "Intro is too long (score 5)" in prompt
    assert "Previous draft:" in prompt
    assert update["trace_log"][0].message.startswith("Revision 1 after critique")


def test_writer_human_revision_does_not_increment_count() -> None:
    language_model = FakeLanguageModel({NewsletterCopy: [make_newsletter_copy(5)]})
    state = writer_state(
        draft=make_draft(),
        critique_history=[make_critique(9)],
        revision_count=2,
        human_feedback="Make it punchier",
    )

    update = build_writer(language_model)(state)

    assert update["revision_count"] == 2
    assert update["human_feedback"] is None
    assert "Make it punchier" in language_model.last_prompt(NewsletterCopy)


def test_merge_article_copy_keeps_original_when_copy_missing() -> None:
    merged = merge_article_copy(
        SUMMARIES[:2], [ArticleCopy(article_number=2, summary="New two", why_it_matters="Why two")]
    )

    assert merged[0] == SUMMARIES[0]
    assert merged[1].summary == "New two"


def test_revision_guidance_for_first_draft() -> None:
    assert build_revision_guidance(None, None, None) == "This is the first draft."


def test_critic_records_report_and_recomputes_passes() -> None:
    inconsistent_report = CritiqueReport(score=9, issues=[], suggested_fixes=[], passes=False)
    language_model = FakeLanguageModel({CritiqueReport: [inconsistent_report]})
    critic_node = build_critic_node(language_model, SETTINGS)

    update = critic_node({"article_summaries": SUMMARIES, "draft": make_draft()})

    assert update["critique_history"][0].passes is True
    assert update["trace_log"][0].message.startswith("Score 9/10")
    assert "# AI agents this week" in language_model.last_prompt(CritiqueReport)


def test_route_after_critique_revises_low_scores_until_limit() -> None:
    route_after_critique = build_route_after_critique(SETTINGS)

    assert (
        route_after_critique({"critique_history": [make_critique(7)], "revision_count": 0})
        == WRITER
    )
    assert (
        route_after_critique({"critique_history": [make_critique(7)], "revision_count": 1})
        == WRITER
    )
    assert (
        route_after_critique({"critique_history": [make_critique(7)], "revision_count": 2})
        == DRAFT_REVIEW
    )
    assert (
        route_after_critique({"critique_history": [make_critique(8)], "revision_count": 0})
        == DRAFT_REVIEW
    )


def test_route_after_draft_review() -> None:
    assert route_after_draft_review({"review_decision": "approve"}) == SENDER
    assert route_after_draft_review({"review_decision": "revise"}) == WRITER
    assert route_after_draft_review({"review_decision": "reject"}) == END
