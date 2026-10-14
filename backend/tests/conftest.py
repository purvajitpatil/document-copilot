"""Shared pytest setup.

Settings are loaded at import time and several fields are required, so give the
suite harmless defaults. This lets `pytest` run in CI (or a fresh checkout) without
a populated backend/.env. Real values from the environment or .env still win.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

_TEST_ENV = {
    "APP_ENV": "test",
    "SUPABASE_URL": "https://test-project.supabase.co",
    "SUPABASE_ANON_KEY": "test-anon-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/postgres",
    "LLM_API_KEY": "test-llm-key",
    "EMBEDDING_API_KEY": "test-embedding-key",
}
for _name, _value in _TEST_ENV.items():
    os.environ.setdefault(_name, _value)


@pytest.fixture(autouse=True)
def _no_live_title_generation():
    """Never let unit tests reach a real LLM for thread titles."""
    with patch("app.chat.orchestrator.suggest_thread_title", AsyncMock(return_value=None)):
        yield


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    from app.ratelimit import reset_chat_limiter

    reset_chat_limiter()
    yield
    reset_chat_limiter()
