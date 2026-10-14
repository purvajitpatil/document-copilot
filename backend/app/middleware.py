"""Pure-ASGI request middleware: request IDs, access logs, security headers.

Written as raw ASGI (not BaseHTTPMiddleware) so server-sent event streams are never
buffered, and so unexpected errors become a JSON 500 *inside* the CORS layer.
"""

from __future__ import annotations

import json
import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings

log = structlog.get_logger("app.request")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_QUIET_PATHS = frozenset({"/health", "/ready"})

_STATIC_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cross-origin-resource-policy", b"same-site"),
]


def _incoming_request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == b"x-request-id":
            candidate = value.decode("latin-1")
            if _REQUEST_ID_RE.match(candidate):
                return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        method = scope["method"]
        path = scope["path"]
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers.append((b"x-request-id", request_id.encode()))
                for name, value in _STATIC_HEADERS:
                    if name not in present:
                        headers.append((name, value))
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                if settings.is_production:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("request.unhandled_error", method=method, path=path)
            if response_started:
                raise
            status_code = 500
            body = json.dumps(
                {"detail": "Internal server error", "requestId": request_id}
            ).encode()
            await send_wrapper(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            logger = log.debug if path in _QUIET_PATHS else log.info
            logger(
                "request.completed",
                method=method,
                path=path,
                status=status_code,
                duration_ms=duration_ms,
            )
            structlog.contextvars.clear_contextvars()
