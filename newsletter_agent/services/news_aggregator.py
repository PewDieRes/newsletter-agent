"""Combines several news sources into one de-duplicated list of articles."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from urllib.parse import parse_qsl, urlencode, urlsplit

from newsletter_agent.domain.models import Article
from newsletter_agent.errors import ResearchFailedError
from newsletter_agent.services.news_sources import NewsSource

GOOGLE_NEWS_HOST = "news.google.com"
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9 ]")
TRACKING_QUERY_PARAMETERS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref"})
TRACKING_QUERY_PREFIX = "utm_"


@dataclass(frozen=True)
class SourceFailure:
    """A single search that raised an error."""

    source_name: str
    query: str
    error_message: str


@dataclass(frozen=True)
class AggregatedResults:
    """Merged search results plus any searches that failed."""

    articles: list[Article]
    failures: list[SourceFailure] = field(default_factory=list)


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for duplicate detection."""
    return " ".join(NON_ALPHANUMERIC_PATTERN.sub("", title.lower()).split())


def _is_tracking_parameter(parameter_name: str) -> bool:
    return parameter_name.startswith(TRACKING_QUERY_PREFIX) or (
        parameter_name in TRACKING_QUERY_PARAMETERS
    )


def normalize_url(url: str) -> str:
    """Reduce a URL to host + path + identifying query parameters.

    Scheme, 'www.', trailing slashes, and tracking parameters (utm_*, fbclid, etc.)
    are ignored, so only query parameters that identify the article are kept.
    """
    url_parts = urlsplit(url.strip())
    if not url_parts.netloc:
        return ""
    host = url_parts.netloc.lower().removeprefix("www.")
    key = f"{host}{url_parts.path.rstrip('/')}"
    identifying_pairs = sorted(
        (parameter_name, parameter_value)
        for parameter_name, parameter_value in parse_qsl(url_parts.query, keep_blank_values=True)
        if not _is_tracking_parameter(parameter_name)
    )
    if identifying_pairs:
        key += "?" + urlencode(identifying_pairs)
    return key


def _is_google_news_link(url: str) -> bool:
    return urlsplit(url).netloc.lower() == GOOGLE_NEWS_HOST


def deduplicate_articles(articles: Iterable[Article]) -> list[Article]:
    """Keep the first copy of each story, preferring a publisher URL over a Google News link."""
    unique_articles: list[Article] = []
    positions_by_key: dict[str, int] = {}
    for article in articles:
        duplicate_keys = [
            duplicate_key
            for duplicate_key in (normalize_url(article.url), normalize_title(article.title))
            if duplicate_key
        ]
        existing_position = next(
            (
                positions_by_key[duplicate_key]
                for duplicate_key in duplicate_keys
                if duplicate_key in positions_by_key
            ),
            None,
        )
        if existing_position is None:
            positions_by_key.update(
                {duplicate_key: len(unique_articles) for duplicate_key in duplicate_keys}
            )
            unique_articles.append(article)
        elif _is_google_news_link(unique_articles[existing_position].url) and not (
            _is_google_news_link(article.url)
        ):
            unique_articles[existing_position] = article
            positions_by_key.update(
                {duplicate_key: existing_position for duplicate_key in duplicate_keys}
            )
    return unique_articles


class NewsAggregator:
    """Runs every query against every source and merges the results."""

    def __init__(self, news_sources: Sequence[NewsSource]) -> None:
        self._news_sources = list(news_sources)

    def search_all(self, queries: Sequence[str], recency_days: int) -> AggregatedResults:
        """Search all sources; raise ResearchFailedError only if nothing was found and errors occurred."""
        article_batches: list[list[Article]] = []
        failures: list[SourceFailure] = []
        for news_source in self._news_sources:
            for query in queries:
                try:
                    article_batches.append(news_source.search(query, recency_days))
                except Exception as search_error:
                    failures.append(
                        SourceFailure(
                            source_name=news_source.name,
                            query=query,
                            error_message=str(search_error),
                        )
                    )
        merged_articles = deduplicate_articles(chain.from_iterable(article_batches))
        if not merged_articles and failures:
            failure_details = "; ".join(
                f"{failure.source_name}: {failure.error_message}" for failure in failures
            )
            raise ResearchFailedError(f"Every news search failed: {failure_details}")
        return AggregatedResults(articles=merged_articles, failures=failures)
