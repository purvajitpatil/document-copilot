from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import CurrentUser, get_access_token, get_current_user
from app.config import Settings, settings
from app.main import app

USER = CurrentUser(id=uuid.uuid4(), email="analyst@example.com")


@pytest.fixture
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_is_dependency_free(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_responses_carry_request_id_and_security_headers(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-12345"})
    assert response.headers["x-request-id"] == "trace-abc-12345"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"


def test_malformed_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id!\t"})
    assert response.headers["x-request-id"] != "bad id!\t"
    assert len(response.headers["x-request-id"]) == 32


def test_ready_reports_database_up(client: TestClient) -> None:
    with (
        patch("app.main._ping_database"),
        patch("app.main.check_gateway", return_value={"status": "ok"}),
    ):
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "llm": {"status": "ok"}}


def test_ready_returns_503_when_database_down_but_still_reports_llm(client: TestClient) -> None:
    with (
        patch("app.main._ping_database", side_effect=RuntimeError("no route")),
        patch("app.main.check_gateway", return_value={"status": "unreachable"}),
    ):
        response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["database"] == "unreachable"
    assert body["status"] == "unavailable"
    assert body["llm"] == {"status": "unreachable"}


def test_unhandled_error_returns_json_500_without_internals() -> None:
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/__boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail sk-12345")

    app.include_router(router)
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.get("/__boom")
    finally:
        app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") != "/__boom"]

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["requestId"] == response.headers["x-request-id"]
    assert "secret" not in response.text


def test_chat_stream_is_rate_limited_per_user(monkeypatch) -> None:
    from app.ratelimit import reset_chat_limiter

    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_chat_per_minute", 2)
    reset_chat_limiter()
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_access_token] = lambda: "token"
    body = {
        "threadId": str(uuid.uuid4()),
        "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
    }
    thread = MagicMock(title="New chat")

    async def empty_turn(**_kwargs):
        if False:  # pragma: no cover - makes this an async generator
            yield ""

    try:
        with (
            patch("app.api.chat.ensure_user", AsyncMock()),
            patch("app.api.chat.require_thread_access", AsyncMock(return_value=thread)),
            patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
            patch("app.api.chat.run_turn", side_effect=empty_turn),
            TestClient(app) as test_client,
        ):
            codes = [test_client.post("/chat/stream", json=body).status_code for _ in range(3)]
            limited = test_client.post("/chat/stream", json=body)
    finally:
        app.dependency_overrides.clear()
        reset_chat_limiter()

    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_oversized_message_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_message_chars", 10)
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_access_token] = lambda: "token"
    body = {
        "threadId": str(uuid.uuid4()),
        "messages": [{"role": "user", "parts": [{"type": "text", "text": "x" * 11}]}],
    }
    try:
        with (
            patch("app.api.chat.ensure_user", AsyncMock()),
            patch("app.api.chat.require_thread_access", AsyncMock(return_value=MagicMock(title="t"))),
            TestClient(app) as test_client,
        ):
            response = test_client.post("/chat/stream", json=body)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 413


_BASE = {
    "supabase_url": "https://x.supabase.co",
    "supabase_anon_key": "a",
    "supabase_service_role_key": "b",
    "database_url": "postgres://u:p@h:5432/d",
    "llm_api_key": "k",
}


@pytest.fixture
def clean_settings_env(monkeypatch) -> None:
    """Make Settings(...) depend only on explicit kwargs, not the suite's test env."""
    for name in ("EMBEDDING_API_KEY", "OPENAI_API_KEY", "APP_ENV", "ALLOWED_ORIGINS"):
        monkeypatch.delenv(name, raising=False)


def test_settings_require_an_embedding_key(clean_settings_env) -> None:
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        Settings(_env_file=None, **_BASE)


def test_settings_accept_openai_key_as_embedding_fallback(clean_settings_env) -> None:
    cfg = Settings(_env_file=None, openai_api_key="sk-o", **_BASE)
    assert cfg.resolved_embedding_api_key == "sk-o"
    cfg = Settings(_env_file=None, openai_api_key="sk-o", embedding_api_key="sk-e", **_BASE)
    assert cfg.resolved_embedding_api_key == "sk-e"


def test_production_rejects_wildcard_cors_and_hides_docs_by_default(clean_settings_env) -> None:
    with pytest.raises(ValueError, match="ALLOWED_ORIGINS"):
        Settings(_env_file=None, openai_api_key="k", app_env="production", allowed_origins="*", **_BASE)
    prod = Settings(_env_file=None, openai_api_key="k", app_env="production", **_BASE)
    assert prod.api_docs_enabled is False
    dev = Settings(_env_file=None, openai_api_key="k", app_env="development", **_BASE)
    assert dev.api_docs_enabled is True
