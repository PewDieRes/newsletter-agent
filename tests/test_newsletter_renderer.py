from datetime import date

from newsletter_agent.domain.models import Article, ArticleSummary
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer
from tests.fakes import make_draft


def build_renderer() -> NewsletterRenderer:
    return NewsletterRenderer("AI Agents Weekly", issue_date_provider=lambda: date(2026, 10, 14))


def test_render_html_includes_subject_items_links_and_date() -> None:
    draft = make_draft(article_count=3)

    rendered_html = build_renderer().render_html(draft)

    assert "<title>AI agents this week</title>" in rendered_html
    assert "AI Agents Weekly" in rendered_html
    assert "October 14, 2026" in rendered_html
    for item in draft.items:
        assert item.article.url in rendered_html
        assert item.summary in rendered_html
        assert item.why_it_matters in rendered_html


def test_render_html_escapes_untrusted_text() -> None:
    risky_item = ArticleSummary(
        article=Article(
            title="Agents <script>alert(1)</script> & tools",
            url="https://example.com/risky",
            source_name="Example",
        ),
        summary="Uses <b>bold</b> claims",
    )
    draft = make_draft(article_count=0).model_copy(update={"items": [risky_item]})

    rendered_html = build_renderer().render_html(draft)

    assert "<script>alert(1)</script>" not in rendered_html
    assert "&lt;script&gt;" in rendered_html
    assert "&amp; tools" in rendered_html


def test_render_html_omits_empty_why_it_matters() -> None:
    plain_item = ArticleSummary(
        article=Article(title="Plain", url="https://example.com/plain", source_name="Example"),
        summary="Only a summary.",
    )
    draft = make_draft(article_count=0).model_copy(update={"items": [plain_item]})

    assert "Why it matters" not in build_renderer().render_html(draft)


def test_render_markdown_lists_every_item_with_link() -> None:
    draft = make_draft(article_count=2)

    rendered_markdown = build_renderer().render_markdown(draft)

    assert rendered_markdown.startswith("# AI agents this week")
    assert "## 1. [AI agent story 1](https://news.example.com/story-1)" in rendered_markdown
    assert "**Why it matters:** Story 2 matters." in rendered_markdown
    assert rendered_markdown.rstrip().endswith("See you next week.")


def test_render_markdown_escapes_link_brackets() -> None:
    bracket_item = ArticleSummary(
        article=Article(
            title="Agents [beta] launch ✨", url="https://example.com/beta", source_name="Example"
        ),
        summary="Beta.",
    )
    draft = make_draft(article_count=0).model_copy(update={"items": [bracket_item]})

    rendered_markdown = build_renderer().render_markdown(draft)

    assert "[Agents \\[beta\\] launch ✨](https://example.com/beta)" in rendered_markdown


def test_render_html_from_markdown_converts_human_edits() -> None:
    rendered_html = build_renderer().render_html_from_markdown(
        "Edited subject", "Preview", "# Heading\n\nA **bold** edit."
    )

    assert "<title>Edited subject</title>" in rendered_html
    assert "<h1>Heading</h1>" in rendered_html
    assert "<strong>bold</strong>" in rendered_html
