"""Loads the mock subscriber list."""

import json
from pathlib import Path

from newsletter_agent.domain.models import Subscriber


def load_subscribers(subscribers_path: Path) -> list[Subscriber]:
    """Read subscribers from a JSON array of {name, email} objects."""
    raw_subscribers = json.loads(subscribers_path.read_text(encoding="utf-8"))
    return [Subscriber.model_validate(raw_subscriber) for raw_subscriber in raw_subscribers]
