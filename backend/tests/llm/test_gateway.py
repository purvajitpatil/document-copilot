from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import openai
import pytest
from pydantic import BaseModel

from app.config import settings
from app.llm import gateway
from app.llm.gateway import (
    StructuredOutputError,
    extract_json_object,
    structured_completion,
)


class Thing(BaseModel):
    name: str
    count: int


def _text_response(content) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _parsed_response(parsed) -> MagicMock:
    response = _text_response(None)
    response.choices[0].message.parsed = parsed
    return response


def _bad_request() -> openai.BadRequestError:
    request = httpx.Request("POST", "http://gateway/v1/chat/completions")
    return openai.BadRequestError(
        "response_format unsupported",
        response=httpx.Response(400, request=request),
        body=None,
    )


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
]


@pytest.fixture(autouse=True)
def _full_degradation_chain(monkeypatch):
    monkeypatch.setattr(settings, "llm_structured_mode", "json_schema")


def test_extract_json_handles_fences_think_blocks_and_chatter() -> None:
    raw = '<think>hmm {not json}</think>\nSure!\n```json\n{"name":"a","count":2}\n```\nDone.'
    assert extract_json_object(raw) == '{"name":"a","count":2}'


def test_extract_json_raises_without_object() -> None:
    with pytest.raises(StructuredOutputError):
        extract_json_object("no json here")


def test_json_schema_mode_used_first() -> None:
    client = MagicMock()
    client.chat.completions.parse.return_value = _parsed_response(Thing(name="x", count=1))

    result = structured_completion(client, model="m", messages=MESSAGES, schema=Thing)

    assert result == Thing(name="x", count=1)
    client.chat.completions.create.assert_not_called()


def test_degrades_to_json_object_when_schema_mode_rejected() -> None:
    client = MagicMock()
    client.chat.completions.parse.side_effect = _bad_request()
    client.chat.completions.create.return_value = _text_response('{"name":"y","count":3}')

    result = structured_completion(client, model="m", messages=MESSAGES, schema=Thing)

    assert result == Thing(name="y", count=3)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    # schema instructions are appended to the system prompt in non-strict modes
    assert "JSON Schema" in kwargs["messages"][0]["content"]
    assert MESSAGES[0]["content"] == "sys"  # caller's messages are not mutated


def test_degrades_to_prompt_only_when_json_object_also_rejected() -> None:
    client = MagicMock()
    client.chat.completions.parse.side_effect = _bad_request()
    client.chat.completions.create.side_effect = [
        _bad_request(),
        _text_response('```json\n{"name":"z","count":4}\n```'),
    ]

    result = structured_completion(client, model="m", messages=MESSAGES, schema=Thing)

    assert result == Thing(name="z", count=4)
    assert "response_format" not in client.chat.completions.create.call_args.kwargs


def test_unparseable_reply_in_every_mode_raises() -> None:
    client = MagicMock()
    client.chat.completions.parse.return_value = _parsed_response(None)
    client.chat.completions.create.return_value = _text_response("sorry, no")

    with pytest.raises(StructuredOutputError):
        structured_completion(client, model="m", messages=MESSAGES, schema=Thing)


def test_transport_errors_are_not_degraded() -> None:
    client = MagicMock()
    client.chat.completions.parse.side_effect = RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        structured_completion(client, model="m", messages=MESSAGES, schema=Thing)

    client.chat.completions.create.assert_not_called()


def test_prompt_mode_skips_strict_attempts(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_structured_mode", "prompt")
    client = MagicMock()
    client.chat.completions.create.return_value = _text_response('{"name":"p","count":5}')

    result = structured_completion(client, model="m", messages=MESSAGES, schema=Thing)

    assert result.count == 5
    client.chat.completions.parse.assert_not_called()


def test_check_gateway_reports_ok_and_unreachable(monkeypatch) -> None:
    fake = MagicMock()
    fake.with_options.return_value.models.list.return_value = MagicMock(data=[1, 2, 3])
    monkeypatch.setattr(gateway, "get_llm_client", lambda: fake)
    ok = gateway.check_gateway(use_cache=False)
    assert ok["status"] == "ok" and ok["models_visible"] == 3

    fake.with_options.return_value.models.list.side_effect = httpx.ConnectError("down")
    down = gateway.check_gateway(use_cache=False)
    assert down == {"status": "unreachable", "error": "ConnectError"}


def test_embedding_kwargs_respects_setting(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_send_dimensions", True)
    assert gateway.embedding_kwargs() == {"dimensions": settings.openai_embedding_dimensions}
    monkeypatch.setattr(settings, "embedding_send_dimensions", False)
    assert gateway.embedding_kwargs() == {}


def test_agent_output_type_switches_on_mode(monkeypatch) -> None:
    from pydantic_ai import PromptedOutput

    from app.assistant.outputs import GroundedAnswer

    monkeypatch.setattr(settings, "llm_output_mode", "tool")
    assert gateway.agent_output_type() is GroundedAnswer
    monkeypatch.setattr(settings, "llm_output_mode", "prompted")
    assert isinstance(gateway.agent_output_type(), PromptedOutput)
