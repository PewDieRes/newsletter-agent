from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from newsletter_agent.config import Settings
from newsletter_agent.domain.models import CritiqueReport
from newsletter_agent.errors import StructuredOutputError
from newsletter_agent.services.llm_factory import build_chat_model, build_fallback_chat_models
from newsletter_agent.services.llm_gateway import STRUCTURED_OUTPUT_REMINDER, ChatModelGateway
from tests.fakes import make_critique


class StubRunnable:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.received_messages: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.received_messages.append(list(messages))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    """A controllable replacement for time.monotonic."""

    def __init__(self) -> None:
        self.current_time = 1000.0

    def __call__(self) -> float:
        return self.current_time


class StubChatModel:
    def __init__(self, structured_responses: list[Any], tool_responses: list[Any] | None = None):
        self.structured_runnable = StubRunnable(structured_responses)
        self.tool_runnable = StubRunnable(tool_responses or [])
        self.bound_tools: list[Any] = []

    def with_structured_output(self, output_schema: type) -> StubRunnable:
        return self.structured_runnable

    def bind_tools(self, tools: list[Any]) -> StubRunnable:
        self.bound_tools = tools
        return self.tool_runnable


def test_generate_structured_returns_parsed_model() -> None:
    chat_model = StubChatModel([make_critique(9)])
    gateway = ChatModelGateway(chat_model)

    critique = gateway.generate_structured("system", "user", CritiqueReport)

    assert critique.score == 9
    sent_messages = chat_model.structured_runnable.received_messages[0]
    assert [message.content for message in sent_messages] == ["system", "user"]


def test_generate_structured_retries_once_with_reminder() -> None:
    chat_model = StubChatModel([None, make_critique(7)])
    gateway = ChatModelGateway(chat_model)

    critique = gateway.generate_structured("system", "user", CritiqueReport)

    assert critique.score == 7
    retry_messages = chat_model.structured_runnable.received_messages[1]
    assert isinstance(retry_messages[-1], HumanMessage)
    assert retry_messages[-1].content == STRUCTURED_OUTPUT_REMINDER


def test_generate_structured_raises_after_second_failure() -> None:
    chat_model = StubChatModel([OutputParserException("bad json"), None])
    gateway = ChatModelGateway(chat_model)

    with pytest.raises(StructuredOutputError, match="CritiqueReport"):
        gateway.generate_structured("system", "user", CritiqueReport)


def test_invoke_with_tools_binds_tools_and_returns_message() -> None:
    tool_reply = AIMessage(content="done")
    chat_model = StubChatModel([], tool_responses=[tool_reply])
    gateway = ChatModelGateway(chat_model)
    sentinel_tool = object()

    reply = gateway.invoke_with_tools([HumanMessage(content="go")], [sentinel_tool])

    assert reply is tool_reply
    assert chat_model.bound_tools == [sentinel_tool]


def test_build_chat_model_uses_configured_gemini_model() -> None:
    settings = Settings(_env_file=None, google_api_key="test-key", gemini_model="gemini-test")

    chat_model = build_chat_model(settings)

    assert isinstance(chat_model, ChatGoogleGenerativeAI)
    assert chat_model.model.endswith("gemini-test")


def test_generate_structured_falls_back_when_primary_model_errors() -> None:
    primary_model = StubChatModel([RuntimeError("503 UNAVAILABLE")])
    fallback_model = StubChatModel([make_critique(9)])
    gateway = ChatModelGateway(primary_model, [fallback_model])

    critique = gateway.generate_structured("system", "user", CritiqueReport)

    assert critique.score == 9
    assert len(primary_model.structured_runnable.received_messages) == 1
    assert len(fallback_model.structured_runnable.received_messages) == 1


def test_generate_structured_does_not_fall_back_on_parse_errors() -> None:
    primary_model = StubChatModel([None, None])
    fallback_model = StubChatModel([make_critique(9)])
    gateway = ChatModelGateway(primary_model, [fallback_model])

    with pytest.raises(StructuredOutputError):
        gateway.generate_structured("system", "user", CritiqueReport)

    assert fallback_model.structured_runnable.received_messages == []


def test_generate_structured_reraises_last_error_when_all_models_fail() -> None:
    primary_model = StubChatModel([RuntimeError("first")])
    fallback_model = StubChatModel([RuntimeError("second")])
    gateway = ChatModelGateway(primary_model, [fallback_model])

    with pytest.raises(RuntimeError, match="second"):
        gateway.generate_structured("system", "user", CritiqueReport)

    assert len(primary_model.structured_runnable.received_messages) == 1


def test_invoke_with_tools_falls_back_when_primary_model_errors() -> None:
    fallback_reply = AIMessage(content="from fallback")
    primary_model = StubChatModel([], tool_responses=[RuntimeError("503 UNAVAILABLE")])
    fallback_model = StubChatModel([], tool_responses=[fallback_reply])
    gateway = ChatModelGateway(primary_model, [fallback_model])

    reply = gateway.invoke_with_tools([HumanMessage(content="go")], [])

    assert reply is fallback_reply
    assert len(primary_model.tool_runnable.received_messages) == 1


def test_build_fallback_chat_models() -> None:
    primary_name = "gemini-a"
    with_fallback = build_fallback_chat_models(
        Settings(
            _env_file=None,
            google_api_key="test-key",
            gemini_model=primary_name,
            gemini_fallback_model="gemini-b",
        )
    )
    disabled = build_fallback_chat_models(
        Settings(
            _env_file=None,
            google_api_key="test-key",
            gemini_model=primary_name,
            gemini_fallback_model="",
        )
    )
    same_as_primary = build_fallback_chat_models(
        Settings(
            _env_file=None,
            google_api_key="test-key",
            gemini_model=primary_name,
            gemini_fallback_model=primary_name,
        )
    )

    assert len(with_fallback) == 1
    assert isinstance(with_fallback[0], ChatGoogleGenerativeAI)
    assert with_fallback[0].model.endswith("gemini-b")
    assert disabled == []
    assert same_as_primary == []


def test_primary_is_skipped_during_cooldown_after_a_provider_error() -> None:
    primary_model = StubChatModel([RuntimeError("503 UNAVAILABLE")])
    fallback_model = StubChatModel([make_critique(9), make_critique(8)])
    gateway = ChatModelGateway(
        primary_model, [fallback_model], fallback_cooldown_seconds=300.0, clock=FakeClock()
    )

    first_critique = gateway.generate_structured("system", "user", CritiqueReport)
    second_critique = gateway.generate_structured("system", "user", CritiqueReport)

    assert [first_critique.score, second_critique.score] == [9, 8]
    assert len(primary_model.structured_runnable.received_messages) == 1
    assert len(fallback_model.structured_runnable.received_messages) == 2


def test_primary_is_retried_after_the_cooldown_expires() -> None:
    fake_clock = FakeClock()
    primary_model = StubChatModel([RuntimeError("503 UNAVAILABLE"), make_critique(10)])
    fallback_model = StubChatModel([make_critique(9)])
    gateway = ChatModelGateway(
        primary_model, [fallback_model], fallback_cooldown_seconds=300.0, clock=fake_clock
    )

    gateway.generate_structured("system", "user", CritiqueReport)
    fake_clock.current_time += 301.0
    critique = gateway.generate_structured("system", "user", CritiqueReport)

    assert critique.score == 10
    assert len(primary_model.structured_runnable.received_messages) == 2
    assert len(fallback_model.structured_runnable.received_messages) == 1


def test_single_model_gateway_reraises_its_provider_error_on_every_call() -> None:
    chat_model = StubChatModel([RuntimeError("first outage"), RuntimeError("second outage")])
    gateway = ChatModelGateway(chat_model, clock=FakeClock())

    with pytest.raises(RuntimeError, match="first outage"):
        gateway.generate_structured("system", "user", CritiqueReport)
    with pytest.raises(RuntimeError, match="second outage"):
        gateway.generate_structured("system", "user", CritiqueReport)


def test_build_chat_model_uses_a_single_attempt_when_a_fallback_exists() -> None:
    settings = Settings(
        _env_file=None,
        google_api_key="test-key",
        gemini_model="gemini-a",
        gemini_fallback_model="gemini-b",
        llm_max_retries=2,
        llm_primary_max_retries_with_fallback=0,
    )

    primary_model = build_chat_model(settings)
    fallback_model = build_fallback_chat_models(settings)[0]

    assert primary_model.max_retries == 0
    assert fallback_model.max_retries == 2
    assert "temperature" in primary_model.model_fields_set
    assert "temperature" not in fallback_model.model_fields_set


def test_build_chat_model_keeps_retries_when_no_fallback_exists() -> None:
    settings = Settings(
        _env_file=None,
        google_api_key="test-key",
        gemini_model="gemini-a",
        gemini_fallback_model="",
        llm_max_retries=2,
    )

    primary_model = build_chat_model(settings)

    assert primary_model.max_retries == 2
