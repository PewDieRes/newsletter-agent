"""Domain objects shared by every layer of the newsletter agent."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

AgentMode = Literal["auto", "hitl"]
ReviewDecision = Literal["approve", "revise", "reject"]


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class Article(BaseModel):
    """A news article discovered during research."""

    title: str
    url: str
    source_name: str
    published_at: datetime | None = None
    snippet: str = ""
    content: str = ""


class NewsletterPlan(BaseModel):
    """The agent's plan for researching and writing one newsletter issue."""

    topic: str = Field(description="The newsletter subject in a few words")
    audience: str = Field(description="Who the newsletter is written for")
    tone: str = Field(description="The writing style to use")
    search_queries: list[str] = Field(description="Distinct news search queries")
    recency_days: int = Field(description="How many days back to search")
    relevance_criteria: list[str] = Field(description="Rules for including an article")
    section_outline: list[str] = Field(description="Ordered newsletter sections")


class ArticleSummary(BaseModel):
    """An article paired with its newsletter-ready summary."""

    article: Article
    summary: str
    why_it_matters: str = ""


class NewsletterDraft(BaseModel):
    """A complete newsletter issue, in structured, HTML, and Markdown form."""

    subject: str
    preheader: str
    intro: str
    items: list[ArticleSummary]
    outro: str
    html_body: str = ""
    markdown_body: str = ""


class CritiqueReport(BaseModel):
    """The critic's evaluation of a draft."""

    score: int = Field(ge=1, le=10, description="Overall quality from 1 to 10")
    issues: list[str] = Field(description="Concrete problems found in the draft")
    suggested_fixes: list[str] = Field(description="Changes that would resolve the issues")
    passes: bool = Field(description="True when the draft is ready to send")


class TraceEntry(BaseModel):
    """One human-readable line in the agent's execution trace."""

    step: str
    message: str
    timestamp: datetime = Field(default_factory=utc_now)


class Subscriber(BaseModel):
    """A newsletter recipient."""

    name: str
    email: str


class DeliveryReport(BaseModel):
    """What the (simulated) send produced."""

    recipient_count: int
    subject: str
    outbox_paths: list[str]
    html_path: str
