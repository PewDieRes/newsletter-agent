"""Turns a NewsletterDraft into email-safe HTML and Markdown."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader

from newsletter_agent.domain.models import NewsletterDraft

TEMPLATES_DIRECTORY = Path(__file__).parent / "templates"
NEWSLETTER_TEMPLATE_NAME = "newsletter.html.j2"
ISSUE_DATE_FORMAT = "%B %d, %Y"


def escape_markdown_link_text(link_text: str) -> str:
    """Escape square brackets so a title cannot break a Markdown link."""
    return link_text.replace("[", "\\[").replace("]", "\\]")


class NewsletterRenderer:
    """Renders newsletters with a single Jinja2 email template."""

    def __init__(
        self,
        newsletter_name: str,
        issue_date_provider: Callable[[], date] = date.today,
    ) -> None:
        self._newsletter_name = newsletter_name
        self._issue_date_provider = issue_date_provider
        self._environment = Environment(
            loader=FileSystemLoader(TEMPLATES_DIRECTORY), autoescape=True
        )

    def render_html(self, draft: NewsletterDraft) -> str:
        """Render the structured draft as a complete HTML email."""
        return self._render(
            subject=draft.subject,
            preheader=draft.preheader,
            intro=draft.intro,
            items=draft.items,
            outro=draft.outro,
            markdown_html=None,
        )

    def render_html_from_markdown(self, subject: str, preheader: str, markdown_text: str) -> str:
        """Render human-edited Markdown inside the same email shell."""
        return self._render(
            subject=subject,
            preheader=preheader,
            intro="",
            items=[],
            outro="",
            markdown_html=markdown.markdown(markdown_text, extensions=["extra"]),
        )

    def render_markdown(self, draft: NewsletterDraft) -> str:
        """Render the structured draft as Markdown (used for plain-text email and editing)."""
        lines = [f"# {draft.subject}", "", draft.intro, ""]
        for position, item in enumerate(draft.items, start=1):
            link_text = escape_markdown_link_text(item.article.title)
            lines.extend(
                [
                    f"## {position}. [{link_text}]({item.article.url})",
                    f"*{item.article.source_name}*",
                    "",
                    item.summary,
                    "",
                ]
            )
            if item.why_it_matters:
                lines.extend([f"**Why it matters:** {item.why_it_matters}", ""])
        lines.append(draft.outro)
        return "\n".join(lines)

    def _render(self, **template_values: object) -> str:
        template = self._environment.get_template(NEWSLETTER_TEMPLATE_NAME)
        return template.render(
            newsletter_name=self._newsletter_name,
            issue_date=self._issue_date_provider().strftime(ISSUE_DATE_FORMAT),
            **template_values,
        )
