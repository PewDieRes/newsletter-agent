"""Builds the configured chat model. The only module that knows the LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from newsletter_agent.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the Gemini chat model with retries and a request timeout."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
