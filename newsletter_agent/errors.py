"""Exceptions raised by the newsletter agent."""


class NewsletterAgentError(Exception):
    """Base class for every expected agent failure."""


class StructuredOutputError(NewsletterAgentError):
    """The language model did not return output matching the requested schema."""


class ResearchFailedError(NewsletterAgentError):
    """No usable articles could be found."""


class RunTimeoutError(NewsletterAgentError):
    """A run took longer than the configured time limit."""


class NodeExecutionError(NewsletterAgentError):
    """A graph node failed; records which step failed and why."""

    def __init__(self, step_name: str, original_error: Exception) -> None:
        super().__init__(f"Step '{step_name}' failed: {original_error}")
        self.step_name = step_name
        self.original_error = original_error
