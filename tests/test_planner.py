from newsletter_agent.agent.nodes.planner import build_planner_node
from newsletter_agent.config import DEFAULT_GOAL, Settings
from newsletter_agent.domain.models import NewsletterPlan
from tests.fakes import FakeLanguageModel, make_plan


def run_planner(plan: NewsletterPlan) -> tuple[dict, FakeLanguageModel]:
    language_model = FakeLanguageModel({NewsletterPlan: [plan]})
    planner_node = build_planner_node(language_model, Settings(_env_file=None))
    return planner_node({"goal": DEFAULT_GOAL, "mode": "auto"}), language_model


def test_planner_returns_plan_and_trace() -> None:
    update, language_model = run_planner(make_plan())

    assert update["plan"].topic == "AI agents"
    assert update["revision_count"] == 0
    assert update["trace_log"][0].step == "planner"
    assert "AI agents news" in update["trace_log"][0].message
    assert DEFAULT_GOAL in language_model.last_prompt(NewsletterPlan)


def test_planner_caps_queries_and_clamps_recency() -> None:
    oversized_plan = make_plan().model_copy(
        update={"search_queries": [f"query {number}" for number in range(10)], "recency_days": 90}
    )

    update, _language_model = run_planner(oversized_plan)

    assert len(update["plan"].search_queries) == 4
    assert update["plan"].recency_days == 30


def test_planner_falls_back_to_topic_when_no_queries() -> None:
    empty_plan = make_plan().model_copy(update={"search_queries": [], "recency_days": 0})

    update, _language_model = run_planner(empty_plan)

    assert update["plan"].search_queries == ["AI agents"]
    assert update["plan"].recency_days == 1
