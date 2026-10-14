from __future__ import annotations

from app.ratelimit import SlidingWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_limit_then_blocks_with_retry_after() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(3, 60, clock=clock)
    assert [limiter.check("u") for _ in range(3)] == [None, None, None]
    retry = limiter.check("u")
    assert retry is not None and 0 < retry <= 60


def test_window_slides() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 60, clock=clock)
    assert limiter.check("u") is None
    assert limiter.check("u") is not None
    clock.now += 61
    assert limiter.check("u") is None


def test_keys_are_independent() -> None:
    limiter = SlidingWindowLimiter(1, 60, clock=FakeClock())
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") is not None
