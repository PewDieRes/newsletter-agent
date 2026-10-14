"""Summarizer node: fetches each selected article and summarizes them in one model call."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.tools import BaseTool

from newsletter_agent.agent.node_names import SUMMARIZER
from newsletter_agent.agent.nodes.formatting import format_articles_with_content
from newsletter_agent.agent.prompts import SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT
from newsletter_agent.agent.state import NewsletterState
from newsletter_agent.config import Settings
from newsletter_agent.domain.llm_outputs import ArticleCopy, SummaryBatch
from newsletter_agent.domain.models import Article, ArticleSummary, TraceEntry
from newsletter_agent.services.llm_gateway import LanguageModelGateway


def fetch_article_contents(
    articles: Sequence[Article], fetch_article_tool: BaseTool
) -> list[Article]:
    """Fetch every article's text concurrently; failures leave content empty."""
    if not articles:
        return []

    def fetch_content(article: Article) -> str:
        return str(fetch_article_tool.invoke({"url": article.url}))

    with ThreadPoolExecutor(max_workers=len(articles)) as executor:
        contents = list(executor.map(fetch_content, articles))
    return [
        article.model_copy(update={"content": content})
        for article, content in zip(articles, contents, strict=True)
    ]


def build_summaries(
    articles: Sequence[Article], article_copies: Sequence[ArticleCopy]
) -> tuple[list[ArticleSummary], int]:
    """Pair each article with its summary; fall back to the snippet when one is missing."""
    copies_by_number = {
        article_copy.article_number: article_copy for article_copy in article_copies
    }
    summaries: list[ArticleSummary] = []
    missing_count = 0
    for position, article in enumerate(articles, start=1):
        light_article = article.model_copy(update={"content": ""})
        article_copy = copies_by_number.get(position)
        if article_copy is None:
            missing_count += 1
            summaries.append(
                ArticleSummary(article=light_article, summary=article.snippet or article.title)
            )
        else:
            summaries.append(
                ArticleSummary(
                    article=light_article,
                    summary=article_copy.summary,
                    why_it_matters=article_copy.why_it_matters,
                )
            )
    return summaries, missing_count


def build_summarizer_node(
    language_model: LanguageModelGateway, fetch_article_tool: BaseTool, settings: Settings
) -> Callable[[NewsletterState], dict[str, Any]]:
    """Create the summarizer node."""

    def summarizer_node(state: NewsletterState) -> dict[str, Any]:
        articles = fetch_article_contents(state["selected_articles"], fetch_article_tool)
        fetched_count = sum(1 for article in articles if article.content)
        user_prompt = SUMMARIZER_USER_PROMPT.format(
            audience=state["plan"].audience,
            numbered_articles=format_articles_with_content(
                articles, settings.max_article_characters
            ),
        )
        summary_batch = language_model.generate_structured(
            SUMMARIZER_SYSTEM_PROMPT, user_prompt, SummaryBatch
        )
        summaries, missing_count = build_summaries(articles, summary_batch.items)
        summary_message = f"Summarized {len(summaries)} articles"
        if missing_count:
            summary_message += f"; {missing_count} fell back to search snippets"
        return {
            "article_summaries": summaries,
            "trace_log": [
                TraceEntry(
                    step=SUMMARIZER,
                    message=(
                        f"Fetched full text for {fetched_count}/{len(articles)} articles "
                        "(others use search snippets)"
                    ),
                ),
                TraceEntry(step=SUMMARIZER, message=summary_message),
            ],
        }

    return summarizer_node
