"""Small in-process sliding-window rate limiter.

State lives in process memory, so each backend replica enforces its own budget.
That is adequate for a single-instance pilot (~40 analysts); put a shared store
(Redis) or an edge limiter in front before scaling out horizontally.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from app.auth.dependencies import CurrentUser, get_current_user
from app.config import settings


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until retry."""
        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(hits[0] + self.window_seconds - now, 0.001)
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under key churn
                self._prune(cutoff)
            return None

    def _prune(self, cutoff: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]


_chat_limiter: SlidingWindowLimiter | None = None


def get_chat_limiter() -> SlidingWindowLimiter:
    global _chat_limiter
    if _chat_limiter is None:
        _chat_limiter = SlidingWindowLimiter(settings.rate_limit_chat_per_minute)
    return _chat_limiter


def reset_chat_limiter() -> None:
    global _chat_limiter
    _chat_limiter = None


async def rate_limit_chat(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Dependency for LLM-backed routes: 429 once a user exceeds their budget."""
    if not settings.rate_limit_enabled:
        return user
    retry_after = get_chat_limiter().check(str(user.id))
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a moment before asking again.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )
    return user


__all__ = [
    "SlidingWindowLimiter",
    "get_chat_limiter",
    "rate_limit_chat",
    "reset_chat_limiter",
]
