"""Formats domain objects as compact text for prompts and tool results."""

from collections.abc import Sequence

from newsletter_agent.domain.models import Article, ArticleSummary

SNIPPET_PREVIEW_CHARACTERS = 200
FULL_TEXT_LABEL = "Text"
SNIPPET_ONLY_TEXT_LABEL = "Text (headline/snippet only — summarize conservatively, add no facts)"


def format_bullets(lines: Sequence[str]) -> str:
    """Render lines as a Markdown bullet list."""
    return "\n".join(f"- {line}" for line in lines) if lines else "- none"


def format_published_date(article: Article) -> str:
    """Return the article's publication date as YYYY-MM-DD, or 'date unknown'."""
    return article.published_at.date().isoformat() if article.published_at else "date unknown"


def format_candidate_list(articles: Sequence[Article]) -> str:
    """Number articles with title, source, date, and a short snippet."""
    if not articles:
        return "No results."
    return "\n".join(
        f"{position}. {article.title} ({article.source_name}, {format_published_date(article)})\n"
        f"   {article.snippet[:SNIPPET_PREVIEW_CHARACTERS]}"
        for position, article in enumerate(articles, start=1)
    )


def format_articles_with_content(articles: Sequence[Article], max_characters: int) -> str:
    """Number articles with their full text (or snippet), truncated for the prompt.

    Snippet-only articles are labelled so the model does not invent missing details.
    """
    article_blocks = []
    for position, article in enumerate(articles, start=1):
        article_text = (article.content or article.snippet)[:max_characters]
        text_label = FULL_TEXT_LABEL if article.content else SNIPPET_ONLY_TEXT_LABEL
        article_blocks.append(
            f"Article {position}: {article.title}\n"
            f"Source: {article.source_name}, {format_published_date(article)}\n"
            f"{text_label}: {article_text}"
        )
    return "\n\n".join(article_blocks)


def format_summary_list(summaries: Sequence[ArticleSummary]) -> str:
    """Number summaries with title, source, URL, summary, and why it matters."""
    return "\n".join(
        f"{position}. {item.article.title} ({item.article.source_name}) {item.article.url}\n"
        f"   Summary: {item.summary}\n"
        f"   Why it matters: {item.why_it_matters or 'n/a'}"
        for position, item in enumerate(summaries, start=1)
    )
