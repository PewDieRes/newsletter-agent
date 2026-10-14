import httpx

from newsletter_agent.services.article_fetcher import HttpArticleFetcher

ARTICLE_PARAGRAPH = (
    "Autonomous AI agents are moving from research demos into production systems. "
    "This week several companies announced agent platforms that plan tasks, call tools, "
    "and check their own work before reporting back to users. "
)
ARTICLE_HTML = f"""
<html><head><title>Agents arrive</title></head>
<body>
  <nav>Home | News | About</nav>
  <article>
    <h1>Agents arrive in production</h1>
    <p>{ARTICLE_PARAGRAPH * 3}</p>
    <p>{ARTICLE_PARAGRAPH * 3}</p>
    <p>{ARTICLE_PARAGRAPH * 3}</p>
  </article>
  <footer>Copyright 2026</footer>
</body></html>
"""


def build_fetcher(handle_request, max_characters: int = 3000) -> HttpArticleFetcher:
    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    return HttpArticleFetcher(http_client, max_characters=max_characters)


def test_fetch_extracts_article_text() -> None:
    fetcher = build_fetcher(lambda request: httpx.Response(200, text=ARTICLE_HTML))

    article_text = fetcher.fetch("https://example.com/agents")

    assert "Autonomous AI agents" in article_text
    assert "Copyright 2026" not in article_text


def test_fetch_truncates_to_max_characters() -> None:
    fetcher = build_fetcher(
        lambda request: httpx.Response(200, text=ARTICLE_HTML), max_characters=50
    )

    assert len(fetcher.fetch("https://example.com/agents")) <= 50


def test_fetch_returns_empty_string_on_http_error() -> None:
    fetcher = build_fetcher(lambda request: httpx.Response(404))

    assert fetcher.fetch("https://example.com/missing") == ""


def test_fetch_returns_empty_string_on_timeout() -> None:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    fetcher = build_fetcher(raise_timeout)

    assert fetcher.fetch("https://example.com/slow") == ""
