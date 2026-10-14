"""Builds the configured chat models. The only module that knows the LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from newsletter_agent.config import Settings


def _build_gemini_model(
    settings: Settings, model_name: str, max_retries: int, temperature: float | None
) -> BaseChatModel:
    """Create a Gemini chat model with retries and a request timeout.

    A temperature of None leaves the model's own default in place.
    """
    optional_arguments: dict[str, float] = (
        {} if temperature is None else {"temperature": temperature}
    )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=max_retries,
        **optional_arguments,
    )


def _fallback_model_name(settings: Settings) -> str | None:
    """Return the fallback model name, or None when disabled or same as the primary."""
    fallback_name = settings.gemini_fallback_model
    if not fallback_name or fallback_name == settings.gemini_model:
        return None
    return fallback_name


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the primary Gemini chat model.

    When a fallback model exists, the primary gets fewer retries so an outage hands
    over to the fallback quickly instead of exhausting the run's time budget.
    """
    max_retries = (
        settings.llm_primary_max_retries_with_fallback
        if _fallback_model_name(settings) is not None
        else settings.llm_max_retries
    )
    return _build_gemini_model(
        settings, settings.gemini_model, max_retries, settings.llm_temperature
    )


def build_fallback_chat_models(settings: Settings) -> list[BaseChatModel]:
    """Create the fallback chat models; empty when disabled or same as the primary.

    The fallback keeps the model's default temperature, which it would otherwise ignore.
    """
    fallback_name = _fallback_model_name(settings)
    if fallback_name is None:
        return []
    return [_build_gemini_model(settings, fallback_name, settings.llm_max_retries, None)]
