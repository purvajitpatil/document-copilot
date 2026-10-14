"""Single entry point for every LLM and embedding call.

All model traffic goes through an OpenAI-compatible endpoint (``LLM_BASE_URL``),
which is typically a self-hosted OmniRoute gateway (``http://omniroute:20128/v1``)
that fans requests out to free providers with automatic fallback. The same code
works against OpenAI, OpenRouter, Ollama, or any other compatible server.

Free and heterogeneous models vary in what they support, so this module also
provides graceful degradation for structured output instead of hard failures.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any, TypeVar

import openai
import structlog
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError
from pydantic_ai import PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.assistant.outputs import GroundedAnswer
from app.config import settings

log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_MODE_ORDER: dict[str, tuple[str, ...]] = {
    "json_schema": ("json_schema", "json_object", "prompt"),
    "json_object": ("json_object", "prompt"),
    "prompt": ("prompt",),
}


class StructuredOutputError(RuntimeError):
    """The model answered, but not in a form we could parse into the schema."""


# --------------------------------------------------------------------------- clients


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """Synchronous client for chat completions (judge, keyword extraction, titles)."""
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    """Client for /v1/embeddings. Defaults to OpenAI; override via EMBEDDING_*."""
    return OpenAI(
        api_key=settings.resolved_embedding_api_key,
        base_url=settings.embedding_base_url,
        timeout=settings.openai_embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )


def build_chat_model() -> OpenAIChatModel:
    """PydanticAI chat model that talks to the configured gateway."""
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return OpenAIChatModel(
        settings.llm_chat_model,
        provider=OpenAIProvider(openai_client=client),
    )


def agent_output_type() -> Any:
    """Output spec for the document agent, per LLM_OUTPUT_MODE."""
    if settings.llm_output_mode == "prompted":
        return PromptedOutput(GroundedAnswer)
    return GroundedAnswer


def embedding_kwargs() -> dict[str, Any]:
    if settings.embedding_send_dimensions:
        return {"dimensions": settings.openai_embedding_dimensions}
    return {}


# ------------------------------------------------------------------ structured output

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> str:
    """Pull the JSON object out of a chat reply (fences, <think> blocks, chatter)."""
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise StructuredOutputError("No JSON object found in model reply.")
    return cleaned[start : end + 1]


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with ONLY a single JSON object (no prose, no code fences) that "
        "conforms to this JSON Schema:\n"
        + json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )


def _with_schema_instruction(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
) -> list[dict[str, str]]:
    instruction = _schema_instruction(schema)
    patched = [dict(m) for m in messages]
    if patched and patched[0]["role"] == "system":
        patched[0]["content"] = f"{patched[0]['content']}\n\n{instruction}"
    else:
        patched.insert(0, {"role": "system", "content": instruction})
    return patched


def _parse_text_reply(response: Any, schema: type[T]) -> T:
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputError("Model returned empty content.")
    try:
        return schema.model_validate_json(extract_json_object(content))
    except ValidationError as exc:
        raise StructuredOutputError(f"Reply did not match schema: {exc}") from exc


def _attempt(
    client: OpenAI,
    mode: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: type[T],
    temperature: float,
) -> T:
    if mode == "json_schema":
        response = client.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=messages,  # type: ignore[arg-type]
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise StructuredOutputError("Model returned no parsed structured output.")
        return parsed

    patched = _with_schema_instruction(messages, schema)
    kwargs: dict[str, Any] = {}
    if mode == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=patched,  # type: ignore[arg-type]
        **kwargs,
    )
    return _parse_text_reply(response, schema)


_DEGRADABLE = (
    StructuredOutputError,
    ValidationError,
    json.JSONDecodeError,
    openai.BadRequestError,
    openai.UnprocessableEntityError,
    openai.LengthFinishReasonError,
    openai.ContentFilterFinishReasonError,
)


def structured_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: type[T],
    temperature: float = 0.0,
) -> T:
    """Chat completion parsed into ``schema`` with automatic mode degradation.

    Tries strict JSON-schema output first, then JSON-object mode, then plain
    prompting + tolerant parsing. Transport errors (timeouts, 5xx, auth, rate limits)
    are NOT degraded - they propagate so callers can fail closed or fall back.
    """
    last_error: Exception | None = None
    for mode in _MODE_ORDER[settings.llm_structured_mode]:
        try:
            return _attempt(
                client,
                mode,
                model=model,
                messages=messages,
                schema=schema,
                temperature=temperature,
            )
        except _DEGRADABLE as exc:
            last_error = exc
            log.warning(
                "llm.structured_output_degraded",
                mode=mode,
                model=model,
                schema=schema.__name__,
                error=str(exc)[:300],
            )
    assert last_error is not None
    raise StructuredOutputError(
        f"No structured-output mode succeeded for {schema.__name__}: {last_error}"
    ) from last_error


# ------------------------------------------------------------------------- health

_HEALTH_TTL_SECONDS = 30.0
_health_cache: tuple[float, dict[str, Any]] | None = None


def check_gateway(*, use_cache: bool = True) -> dict[str, Any]:
    """Cheap reachability probe for the LLM gateway (GET /v1/models)."""
    global _health_cache
    now = time.monotonic()
    if use_cache and _health_cache and now - _health_cache[0] < _HEALTH_TTL_SECONDS:
        return _health_cache[1]

    started = time.monotonic()
    try:
        models = get_llm_client().with_options(timeout=3.0, max_retries=0).models.list()
        count = len(getattr(models, "data", []) or [])
        result: dict[str, Any] = {
            "status": "ok",
            "models_visible": count,
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        result = {"status": "unreachable", "error": type(exc).__name__}
    _health_cache = (now, result)
    return result
