"""Tools for reading articles and producing newsletter HTML."""

from dataclasses import dataclass

from langchain_core.tools import BaseTool, tool

from newsletter_agent.domain.models import NewsletterDraft
from newsletter_agent.services.article_fetcher import ArticleFetcher
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer


@dataclass(frozen=True)
class ContentTools:
    """The content tools, by role."""

    fetch_article: BaseTool
    render_html: BaseTool


def build_content_tools(
    article_fetcher: ArticleFetcher, renderer: NewsletterRenderer
) -> ContentTools:
    """Create the fetch_article and render_html tools."""

    @tool("fetch_article")
    def fetch_article(url: str) -> str:
        """Download an article and return its readable text, or an empty string on failure."""
        return article_fetcher.fetch(url)

    @tool("render_html")
    def render_html(draft: NewsletterDraft) -> str:
        """Render a newsletter draft as an email-safe HTML document."""
        return renderer.render_html(draft)

    return ContentTools(fetch_article=fetch_article, render_html=render_html)
