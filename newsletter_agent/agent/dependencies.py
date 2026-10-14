"""The agent's dependency container and its production wiring."""

from dataclasses import dataclass

import httpx
from tavily import TavilyClient

from newsletter_agent.config import Settings
from newsletter_agent.domain.models import Subscriber
from newsletter_agent.services.article_fetcher import ArticleFetcher, HttpArticleFetcher
from newsletter_agent.services.email_outbox import EmailSender, EmlFileOutbox
from newsletter_agent.services.llm_factory import build_chat_model
from newsletter_agent.services.llm_gateway import ChatModelGateway, LanguageModelGateway
from newsletter_agent.services.news_sources import (
    GoogleNewsRssSource,
    NewsSource,
    TavilyNewsSource,
)
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer
from newsletter_agent.services.subscribers import load_subscribers

USER_AGENT = "Mozilla/5.0 (compatible; NewsletterAgent/1.0)"


@dataclass(frozen=True)
class AgentDependencies:
    """Everything the graph's nodes depend on, injected from one place."""

    language_model: LanguageModelGateway
    web_search_source: NewsSource
    news_feed_source: NewsSource
    article_fetcher: ArticleFetcher
    renderer: NewsletterRenderer
    email_sender: EmailSender
    subscribers: list[Subscriber]
    settings: Settings


def build_http_client(timeout_seconds: float) -> httpx.Client:
    """Create the shared HTTP client used for RSS and article fetching."""
    return httpx.Client(
        timeout=timeout_seconds, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )


def build_default_dependencies(settings: Settings) -> AgentDependencies:
    """Wire the real Gemini, Tavily, Google News, HTTP, and outbox implementations."""
    http_client = build_http_client(settings.fetch_timeout_seconds)
    return AgentDependencies(
        language_model=ChatModelGateway(build_chat_model(settings)),
        web_search_source=TavilyNewsSource(
            TavilyClient(api_key=settings.tavily_api_key), settings.results_per_search
        ),
        news_feed_source=GoogleNewsRssSource(http_client, settings.results_per_search),
        article_fetcher=HttpArticleFetcher(http_client, settings.max_article_characters),
        renderer=NewsletterRenderer(settings.newsletter_name),
        email_sender=EmlFileOutbox(settings.outbox_directory, settings.sender_address),
        subscribers=load_subscribers(settings.subscribers_path),
        settings=settings,
    )
