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
    """LanguageModelGateway backed by any LangChain chat model."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model

    def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[SchemaT]
    ) -> SchemaT:
        """Ask for structured output; retry once with a reminder if parsing fails."""
        structured_model = self._chat_model.with_structured_output(output_schema)
        first_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
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
        """Bind the tools and return the model's reply, which may contain tool calls."""
        tool_model: Any = self._chat_model.bind_tools(list(tools))
        return cast(AIMessage, tool_model.invoke(list(messages)))
