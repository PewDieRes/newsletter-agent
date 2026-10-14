"""Builders and fakes shared by the test suite."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

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


class FakeArticleFetcher:
    """An ArticleFetcher that returns canned page text by URL."""

    def __init__(self, contents: dict[str, str] | None = None) -> None:
        self._contents = dict(contents or {})
        self.fetched_urls: list[str] = []

    def fetch(self, url: str) -> str:
        self.fetched_urls.append(url)
        return self._contents.get(url, "")


class FakeLanguageModel:
    """A LanguageModelGateway with scripted replies.

    Structured responses are queued per schema; the last queued response repeats.
    Tool responses are consumed in order; when exhausted, a reply without tool calls is returned.
    """

    def __init__(
        self,
        structured_responses: dict[type, list[Any]] | None = None,
        tool_responses: list[AIMessage] | None = None,
    ) -> None:
        self._structured_responses = {
            output_schema: list(responses)
            for output_schema, responses in (structured_responses or {}).items()
        }
        self._tool_responses = list(tool_responses or [])
        self.structured_calls: list[tuple[type, str]] = []
        self.tool_call_messages: list[list[BaseMessage]] = []

    def generate_structured(self, system_prompt: str, user_prompt: str, output_schema: type) -> Any:
        self.structured_calls.append((output_schema, user_prompt))
        queued_responses = self._structured_responses.get(output_schema)
        if not queued_responses:
            raise AssertionError(f"No fake response queued for {output_schema.__name__}")
        if len(queued_responses) > 1:
            return queued_responses.pop(0)
        return queued_responses[0]

    def invoke_with_tools(self, messages: Sequence[BaseMessage], tools: Sequence[Any]) -> AIMessage:
        self.tool_call_messages.append(list(messages))
        if not self._tool_responses:
            return AIMessage(content="Research complete.")
        return self._tool_responses.pop(0)

    def call_count(self, output_schema: type) -> int:
        return sum(
            1 for called_schema, _prompt in self.structured_calls if called_schema is output_schema
        )

    def last_prompt(self, output_schema: type) -> str:
        matching_prompts = [
            user_prompt
            for called_schema, user_prompt in self.structured_calls
            if called_schema is output_schema
        ]
        return matching_prompts[-1]


def make_tool_call_message(*tool_calls: tuple[str, str]) -> AIMessage:
    """Build a model reply that calls tools; each tuple is (tool_name, query)."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": {"query": query, "recency_days": 7},
                "id": f"call_{position}",
            }
            for position, (tool_name, query) in enumerate(tool_calls, start=1)
        ],
    )


def search_both_tools_message() -> AIMessage:
    return make_tool_call_message(("web_search", "AI agents news"), ("news_feed", "AI agents"))
