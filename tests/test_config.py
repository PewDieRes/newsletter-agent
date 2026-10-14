from newsletter_agent.config import DEFAULT_GOAL, PROJECT_ROOT, Settings


def test_settings_defaults_match_spec() -> None:
    settings = Settings(_env_file=None)

    assert settings.min_articles == 5
    assert settings.max_articles == 7
    assert settings.quality_threshold == 8
    assert settings.max_revisions == 2
    assert settings.max_research_tool_calls == 4
    assert settings.default_recency_days == 7
    assert settings.widened_recency_days == 14
    assert settings.fetch_timeout_seconds == 8.0
    assert settings.runs_per_session == 3
    assert settings.run_timeout_seconds == 120.0
    assert settings.subscribers_path == PROJECT_ROOT / "data" / "subscribers.json"


def test_settings_read_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("MAX_REVISIONS", "1")

    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-test-model"
    assert settings.max_revisions == 1


def test_default_goal_is_the_assignment_goal() -> None:
    assert DEFAULT_GOAL == (
        "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
    )
