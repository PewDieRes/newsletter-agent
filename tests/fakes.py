"""Builders and fakes shared by the test suite."""

from collections.abc import Sequence
from datetime import UTC, datetime

from newsletter_agent.domain.llm_outputs import ArticleCopy, NewsletterCopy, SummaryBatch
from newsletter_agent.domain.models import (
    Article,
    ArticleSummary,
    CritiqueReport,
    NewsletterDraft,
    NewsletterPlan,
    Subscriber,
)

TEST_SUBSCRIBERS = [
    Subscriber(name="Ada Example", email="ada@example.com"),
    Subscriber(name="Grace Example", email="grace@example.com"),
    Subscriber(name="Linus Example", email="linus@example.com"),
]


def make_article(number: int, source_name: str = "Example News", url: str | None = None) -> Article:
    return Article(
        title=f"AI agent story {number}",
        url=url or f"https://news.example.com/story-{number}",
        source_name=source_name,
        published_at=datetime(2026, 10, 12, 9, 0, tzinfo=UTC),
        snippet=f"Snippet about agent story {number}.",
    )


def make_plan() -> NewsletterPlan:
    return NewsletterPlan(
        topic="AI agents",
        audience="Engineers and product leaders",
        tone="Clear and upbeat",
        search_queries=["AI agents news", "agentic AI launch"],
        recency_days=7,
        relevance_criteria=["About AI agents", "Published this week"],
        section_outline=["Intro", "Top stories", "Outro"],
    )


def make_summary(number: int) -> ArticleSummary:
    return ArticleSummary(
        article=make_article(number),
        summary=f"Summary of story {number}.",
        why_it_matters=f"Story {number} matters.",
    )


def make_draft(article_count: int = 5) -> NewsletterDraft:
    return NewsletterDraft(
        subject="AI agents this week",
        preheader="The biggest agent news in five minutes",
        intro="Welcome to this week's issue.",
        items=[make_summary(number) for number in range(1, article_count + 1)],
        outro="See you next week.",
        html_body="<html><body>Draft</body></html>",
        markdown_body="# AI agents this week",
    )


def make_article_copies(count: int) -> list[ArticleCopy]:
    return [
        ArticleCopy(
            article_number=number,
            summary=f"Polished summary {number}.",
            why_it_matters=f"Polished reason {number}.",
        )
        for number in range(1, count + 1)
    ]


def make_summary_batch(count: int = 7) -> SummaryBatch:
    return SummaryBatch(items=make_article_copies(count))


def make_newsletter_copy(count: int = 7) -> NewsletterCopy:
    return NewsletterCopy(
        subject="Agents everywhere: this week's top stories",
        preheader="Five launches you should know about",
        intro="Agents had a big week.",
        items=make_article_copies(count),
        outro="Thanks for reading.",
    )


def make_critique(score: int) -> CritiqueReport:
    return CritiqueReport(
        score=score,
        issues=[] if score >= 8 else [f"Intro is too long (score {score})"],
        suggested_fixes=[] if score >= 8 else ["Cut the intro to two sentences"],
        passes=score >= 8,
    )


class FakeNewsSource:
    """A NewsSource that returns canned articles or raises a canned error."""

    def __init__(
        self,
        name: str,
        articles: Sequence[Article] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self._articles = list(articles or [])
        self._error = error
        self.received_searches: list[tuple[str, int]] = []

    def search(self, query: str, recency_days: int) -> list[Article]:
        self.received_searches.append((query, recency_days))
        if self._error is not None:
            raise self._error
        return list(self._articles)
