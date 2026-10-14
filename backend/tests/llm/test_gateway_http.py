"""Wire-level check against a fake OpenAI-compatible gateway (stands in for OmniRoute).

Proves the real OpenAI SDK path works end to end: base URL + bearer auth, model
listing, embeddings, and structured-output degradation when the upstream model
rejects strict JSON-schema mode with a 400.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.ai.titles import suggest_thread_title
from app.config import settings
from app.llm import gateway
from app.retrieval.embeddings import embed_query


class _State:
    requests: list[dict] = []
    reject_json_schema = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # silence
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        _State.requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "auto", "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "nope"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        _State.requests.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        if self.path == "/v1/embeddings":
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = [
                {"object": "embedding", "index": i, "embedding": [0.01] * body.get("dimensions", 1536)}
                for i, _ in enumerate(inputs)
            ]
            self._send(200, {"object": "list", "data": data, "model": body["model"],
                             "usage": {"prompt_tokens": 1, "total_tokens": 1}})
            return
        if self.path == "/v1/chat/completions":
            response_format = body.get("response_format") or {}
            if response_format.get("type") == "json_schema" and _State.reject_json_schema:
                self._send(400, {"error": {"message": "response_format json_schema unsupported",
                                           "type": "invalid_request_error"}})
                return
            content = '```json\n{"title": "Apple Revenue Mix."}\n```'
            self._send(200, {
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
            return
        self._send(404, {"error": {"message": "nope"}})


@pytest.fixture
def fake_gateway(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    _State.requests = []
    _State.reject_json_schema = True

    monkeypatch.setattr(settings, "llm_base_url", base)
    monkeypatch.setattr(settings, "llm_api_key", "omni-key")
    monkeypatch.setattr(settings, "embedding_base_url", base)
    monkeypatch.setattr(settings, "embedding_api_key", "omni-embed-key")
    monkeypatch.setattr(settings, "llm_max_retries", 0)
    monkeypatch.setattr(settings, "llm_structured_mode", "json_schema")
    gateway.get_llm_client.cache_clear()
    gateway.get_embedding_client.cache_clear()
    gateway._health_cache = None
    yield _State
    server.shutdown()
    gateway.get_llm_client.cache_clear()
    gateway.get_embedding_client.cache_clear()
    gateway._health_cache = None


def test_title_generation_degrades_through_real_http(fake_gateway) -> None:
    title = asyncio.run(suggest_thread_title("How did Apple's revenue mix change?"))

    assert title == "Apple Revenue Mix"
    chat_calls = [r for r in fake_gateway.requests if r["path"] == "/v1/chat/completions"]
    assert len(chat_calls) == 2  # json_schema rejected, json_object accepted
    assert chat_calls[0]["body"]["response_format"]["type"] == "json_schema"
    assert chat_calls[1]["body"]["response_format"] == {"type": "json_object"}
    assert all(r["auth"] == "Bearer omni-key" for r in chat_calls)


def test_embeddings_use_embedding_credentials_and_dimensions(fake_gateway) -> None:
    vector = embed_query("revenue mix")

    assert len(vector) == settings.openai_embedding_dimensions
    call = next(r for r in fake_gateway.requests if r["path"] == "/v1/embeddings")
    assert call["auth"] == "Bearer omni-embed-key"
    assert call["body"]["dimensions"] == settings.openai_embedding_dimensions


def test_gateway_health_probe(fake_gateway) -> None:
    result = gateway.check_gateway(use_cache=False)
    assert result["status"] == "ok"
    assert result["models_visible"] == 1
