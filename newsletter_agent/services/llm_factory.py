"""Builds the configured chat models. The only module that knows the LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from newsletter_agent.config import Settings


def _build_gemini_model(settings: Settings, model_name: str) -> BaseChatModel:
    """Create a Gemini chat model with retries and a request timeout."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the primary Gemini chat model."""
    return _build_gemini_model(settings, settings.gemini_model)


def build_fallback_chat_models(settings: Settings) -> list[BaseChatModel]:
    """Create the fallback chat models; empty when disabled or same as the primary."""
    fallback_name = settings.gemini_fallback_model
    if not fallback_name or fallback_name == settings.gemini_model:
        return []
    return [_build_gemini_model(settings, fallback_name)]
