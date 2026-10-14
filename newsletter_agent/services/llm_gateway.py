"""A small, provider-agnostic interface over LangChain chat models."""

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from newsletter_agent.errors import StructuredOutputError

SchemaT = TypeVar("SchemaT", bound=BaseModel)

STRUCTURED_OUTPUT_REMINDER = (
    "Your previous reply could not be parsed. Respond again with output that matches "
    "the requested schema exactly, filling in every required field."
)


class LanguageModelGateway(Protocol):
    """What the agent's nodes need from a language model."""

    def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[SchemaT]
    ) -> SchemaT:
        """Return the model's answer parsed into `output_schema`."""
        ...

    def invoke_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool]
    ) -> AIMessage:
        """Send the conversation with tools bound and return the model's reply."""
        ...


class ChatModelGateway:
    """LanguageModelGateway backed by LangChain chat models.

    The primary model is tried first; if it fails with a provider error (e.g. an
    overload), each fallback model is tried in order.
    """

    def __init__(
        self, chat_model: BaseChatModel, fallback_chat_models: Sequence[BaseChatModel] = ()
    ) -> None:
        self._chat_models = [chat_model, *fallback_chat_models]

    def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[SchemaT]
    ) -> SchemaT:
        """Ask for structured output, falling back to the next model on provider errors.

        Parse failures are retried once with a reminder and then raise
        StructuredOutputError without trying other models. If every model raises a
        provider error, the last one is re-raised.
        """
        first_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        last_error: Exception | None = None
        for chat_model in self._chat_models:
            try:
                return self._generate_structured_with(chat_model, first_messages, output_schema)
            except StructuredOutputError:
                raise
            except Exception as model_error:
                last_error = model_error
        assert last_error is not None
        raise last_error

    @staticmethod
    def _generate_structured_with(
        chat_model: BaseChatModel,
        first_messages: list[BaseMessage],
        output_schema: type[SchemaT],
    ) -> SchemaT:
        """Run the two-attempt parse-retry logic against a single model."""
        structured_model = chat_model.with_structured_output(output_schema)
        retry_messages = [*first_messages, HumanMessage(content=STRUCTURED_OUTPUT_REMINDER)]
        failure_reason = "no attempt made"
        for attempt_messages in (first_messages, retry_messages):
            try:
                parsed_output = structured_model.invoke(attempt_messages)
            except (OutputParserException, ValidationError) as parse_error:
                failure_reason = str(parse_error)
                continue
            if isinstance(parsed_output, output_schema):
                return parsed_output
            failure_reason = f"model returned {type(parsed_output).__name__}"
        raise StructuredOutputError(
            f"{output_schema.__name__} could not be generated: {failure_reason}"
        )

    def invoke_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool]
    ) -> AIMessage:
        """Bind the tools and return the reply; on any error try the next model."""
        last_error: Exception | None = None
        for chat_model in self._chat_models:
            try:
                tool_model: Any = chat_model.bind_tools(list(tools))
                return cast(AIMessage, tool_model.invoke(list(messages)))
            except Exception as model_error:
                last_error = model_error
        assert last_error is not None
        raise last_error
