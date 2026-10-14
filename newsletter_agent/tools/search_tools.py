"""Search tools the research loop exposes to the language model."""

from langchain_core.tools import BaseTool, tool

from newsletter_agent.config import DEFAULT_RECENCY_DAYS
from newsletter_agent.domain.models import Article
from newsletter_agent.services.news_sources import NewsSource

WEB_SEARCH_TOOL_NAME = "web_search"
NEWS_FEED_TOOL_NAME = "news_feed"


def build_search_tools(
    web_search_source: NewsSource, news_feed_source: NewsSource
) -> list[BaseTool]:
    """Create the web_search and news_feed tools bound to their news sources."""

    @tool(WEB_SEARCH_TOOL_NAME)
    def web_search(query: str, recency_days: int = DEFAULT_RECENCY_DAYS) -> list[Article]:
        """Search the web for recent news articles. Returns titles, sources, dates, and rich snippets.

        Args:
            query: A news search query, for example "AI agent framework launch".
            recency_days: Only include articles published within this many days.
        """
        return web_search_source.search(query, recency_days)

    @tool(NEWS_FEED_TOOL_NAME)
    def news_feed(query: str, recency_days: int = DEFAULT_RECENCY_DAYS) -> list[Article]:
        """Search the Google News headline feed for recent articles about a query.

        Args:
            query: A news search query, for example "agentic AI funding".
            recency_days: Only include articles published within this many days.
        """
        return news_feed_source.search(query, recency_days)

    return [web_search, news_feed]
