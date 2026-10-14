from newsletter_agent.agent.nodes.article_review import (
    apply_article_review,
    build_article_review_node,
)
from newsletter_agent.agent.nodes.selector import build_selector_node, choose_articles
from newsletter_agent.config import Settings
from newsletter_agent.domain.llm_outputs import ArticleSelection
from tests.fakes import FakeLanguageModel, make_article, make_plan

CANDIDATES = [make_article(number) for number in range(1, 9)]


def test_choose_articles_keeps_model_order() -> None:
    chosen = choose_articles(CANDIDATES, [2, 1, 5, 4, 3], minimum=5, maximum=7)

    assert chosen == [CANDIDATES[1], CANDIDATES[0], CANDIDATES[4], CANDIDATES[3], CANDIDATES[2]]


def test_choose_articles_drops_invalid_and_tops_up() -> None:
    chosen = choose_articles(CANDIDATES, [3, 3, 99, 0, -1, 1], minimum=5, maximum=7)

    assert chosen == [CANDIDATES[2], CANDIDATES[0], CANDIDATES[1], CANDIDATES[3], CANDIDATES[4]]


def test_choose_articles_caps_at_maximum() -> None:
    chosen = choose_articles(CANDIDATES, list(range(1, 9)), minimum=5, maximum=7)

    assert len(chosen) == 7


def test_choose_articles_returns_all_when_fewer_candidates_than_minimum() -> None:
    assert choose_articles(CANDIDATES[:3], [], minimum=5, maximum=7) == CANDIDATES[:3]


def test_selector_node_uses_model_selection() -> None:
    language_model = FakeLanguageModel(
        {
            ArticleSelection: [
                ArticleSelection(selected_article_numbers=[8, 7, 6, 5, 4], rationale="Freshest")
            ]
        }
    )
    selector_node = build_selector_node(language_model, Settings(_env_file=None))

    update = selector_node({"plan": make_plan(), "candidate_articles": CANDIDATES})

    assert update["selected_articles"][0] == CANDIDATES[7]
    assert update["trace_log"][0].message == "Selected 5 of 8 candidates: Freshest"
    assert "8. AI agent story 8" in language_model.last_prompt(ArticleSelection)


def test_article_review_is_a_logged_no_op_in_auto_mode() -> None:
    article_review_node = build_article_review_node(Settings(_env_file=None))

    update = article_review_node(
        {"mode": "auto", "candidate_articles": CANDIDATES, "selected_articles": CANDIDATES[:5]}
    )

    assert "selected_articles" not in update
    assert update["trace_log"][0].step == "article_review"


def test_apply_article_review_uses_human_order_and_ignores_unknown_urls() -> None:
    approved, message = apply_article_review(
        CANDIDATES,
        CANDIDATES[:5],
        [CANDIDATES[6].url, "https://unknown.example.com", CANDIDATES[0].url, CANDIDATES[0].url],
        max_articles=7,
    )

    assert approved == [CANDIDATES[6], CANDIDATES[0]]
    assert message == "Human approved 2 articles"


def test_apply_article_review_keeps_agent_selection_when_human_approves_none() -> None:
    approved, message = apply_article_review(CANDIDATES, CANDIDATES[:5], [], max_articles=7)

    assert approved == CANDIDATES[:5]
    assert "keeping the agent's selection" in message


def test_apply_article_review_caps_at_max_articles() -> None:
    approved, _message = apply_article_review(
        CANDIDATES, CANDIDATES[:5], [article.url for article in CANDIDATES], max_articles=7
    )

    assert len(approved) == 7
