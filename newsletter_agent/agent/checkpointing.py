"""Builds the checkpointer that stores graph state between steps and human reviews."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from newsletter_agent.domain.models import (
    Article,
    ArticleSummary,
    CritiqueReport,
    DeliveryReport,
    NewsletterDraft,
    NewsletterPlan,
    Subscriber,
    TraceEntry,
)

CHECKPOINTED_MODEL_TYPES = (
    Article,
    NewsletterPlan,
    ArticleSummary,
    NewsletterDraft,
    CritiqueReport,
    TraceEntry,
    DeliveryReport,
    Subscriber,
)


def build_checkpointer() -> InMemorySaver:
    """Create an in-memory checkpointer that may restore exactly our domain models.

    Registering the models explicitly keeps resumes working when LangGraph blocks
    unregistered types from checkpoints, and silences its deprecation warning today.
    """
    serializer = JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINTED_MODEL_TYPES))
    return InMemorySaver(serde=serializer)
