"""A small, provider-agnostic interface over LangChain chat models."""

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from newsletter_agent.errors import StructuredOutputError

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ResultT = TypeVar("ResultT")

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
    overload), each fallback model is tried in order. After such a failure the primary
    is skipped for a cooldown period, so an outage costs one failed call, not one per call.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        fallback_chat_models: Sequence[BaseChatModel] = (),
        fallback_cooldown_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary_chat_model = chat_model
        self._fallback_chat_models = list(fallback_chat_models)
        self._fallback_cooldown_seconds = fallback_cooldown_seconds
        self._clock = clock
        self._primary_unavailable_until = float("-inf")

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
        return self._call_with_failover(
            lambda chat_model: self._generate_structured_with(
                chat_model, first_messages, output_schema
            )
        )

    def invoke_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool]
    ) -> AIMessage:
        """Bind the tools and return the reply; on a provider error try the next model."""

        def invoke_model(chat_model: BaseChatModel) -> AIMessage:
            tool_model: Any = chat_model.bind_tools(list(tools))
            return cast(AIMessage, tool_model.invoke(list(messages)))

        return self._call_with_failover(invoke_model)

    def _models_to_try(self) -> list[BaseChatModel]:
        """Return the models in order, skipping the primary while it is cooling down."""
        primary_cooling_down = self._clock() < self._primary_unavailable_until
        if self._fallback_chat_models and primary_cooling_down:
            return list(self._fallback_chat_models)
        return [self._primary_chat_model, *self._fallback_chat_models]

    def _call_with_failover(self, call_model: Callable[[BaseChatModel], ResultT]) -> ResultT:
        """Run `call_model` on each model in turn until one succeeds.

        StructuredOutputError propagates immediately; any other error moves on to the
        next model, and a primary failure starts the cooldown. The last error is
        re-raised when every model fails.
        """
        last_error: Exception | None = None
        for chat_model in self._models_to_try():
            try:
                return call_model(chat_model)
            except StructuredOutputError:
                raise
            except Exception as model_error:
                if chat_model is self._primary_chat_model:
                    self._primary_unavailable_until = (
                        self._clock() + self._fallback_cooldown_seconds
                    )
                last_error = model_error
        if last_error is None:
            raise RuntimeError("No chat model was available to call.")
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
