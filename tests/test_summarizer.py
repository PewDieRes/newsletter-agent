from newsletter_agent.agent.nodes.summarizer import build_summaries, build_summarizer_node
from newsletter_agent.config import Settings
from newsletter_agent.domain.llm_outputs import ArticleCopy, SummaryBatch
from newsletter_agent.services.newsletter_renderer import NewsletterRenderer
from newsletter_agent.tools.content_tools import build_content_tools
from tests.fakes import FakeArticleFetcher, FakeLanguageModel, make_article, make_plan

ARTICLES = [make_article(1), make_article(2)]


def test_summarizer_fetches_text_and_summarizes_every_article() -> None:
    fetcher = FakeArticleFetcher({ARTICLES[0].url: "Full article text about agents."})
    content_tools = build_content_tools(fetcher, NewsletterRenderer("Weekly"))
    batch = SummaryBatch(
        items=[
            ArticleCopy(article_number=1, summary="First summary.", why_it_matters="First why."),
            ArticleCopy(article_number=2, summary="Second summary.", why_it_matters="Second why."),
        ]
    )
    language_model = FakeLanguageModel({SummaryBatch: [batch]})
    summarizer_node = build_summarizer_node(
        language_model, content_tools.fetch_article, Settings(_env_file=None)
    )

    update = summarizer_node({"plan": make_plan(), "selected_articles": ARTICLES})

    assert sorted(fetcher.fetched_urls) == sorted(article.url for article in ARTICLES)
    prompt = language_model.last_prompt(SummaryBatch)
    assert "Text: Full article text about agents." in prompt
    assert "Text: Snippet about agent story 2." in prompt
    assert [item.summary for item in update["article_summaries"]] == [
        "First summary.",
        "Second summary.",
    ]
    assert all(item.article.content == "" for item in update["article_summaries"])
    assert update["trace_log"][0].message.startswith("Fetched full text for 1/2 articles")


def test_build_summaries_falls_back_to_snippet_for_missing_items() -> None:
    summaries, missing_count = build_summaries(
        ARTICLES, [ArticleCopy(article_number=2, summary="Only two.", why_it_matters="Why.")]
    )

    assert missing_count == 1
    assert summaries[0].summary == "Snippet about agent story 1."
    assert summaries[0].why_it_matters == ""
    assert summaries[1].summary == "Only two."
