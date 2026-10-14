"""Application settings loaded from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RECENCY_DAYS = 7
DEFAULT_GOAL = "Create a weekly newsletter on latest AI agent news and send it to our subscribers."


class Settings(BaseSettings):
    """Every tunable value for the agent. Secrets come from the environment only."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    google_api_key: str = ""
    tavily_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    # Used when the primary model errors (e.g. 503 overload); empty disables the fallback.
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    # Google recommends the default temperature of 1.0 for Gemini 3 models.
    llm_temperature: float = 1.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    min_articles: int = 5
    max_articles: int = 7
    quality_threshold: int = 8
    max_revisions: int = 2
    max_research_tool_calls: int = 4
    max_search_queries: int = 4
    default_recency_days: int = DEFAULT_RECENCY_DAYS
    widened_recency_days: int = 14
    results_per_search: int = 8
    fetch_timeout_seconds: float = 8.0
    max_article_characters: int = 3000

    runs_per_session: int = 3
    run_timeout_seconds: float = 120.0

    newsletter_name: str = "AI Agents Weekly"
    sender_address: str = "AI Agents Weekly <newsletter@example.com>"
    subscribers_path: Path = PROJECT_ROOT / "data" / "subscribers.json"
    outbox_directory: Path = PROJECT_ROOT / "outbox"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
