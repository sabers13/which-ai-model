"""Rate limiting: concurrency slot plus a sliding one-minute window on a fake clock."""

from __future__ import annotations

import threading
from pathlib import Path

from which_ai_model.core.ratelimit import Limits, RateLimiter, load_limits


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_load_limits_reads_budget_sections(tmp_path: Path) -> None:
    path = tmp_path / "budget.toml"
    path.write_text(
        """
[limits.opencode-go]
max_concurrency = 2
requests_per_minute = 30

[limits.legacy-alias]
max_concurrent = 3
rpm = 5
""",
        encoding="utf-8",
    )
    limits = load_limits(path)
    assert limits["opencode-go"] == Limits(max_concurrency=2, requests_per_minute=30)
    assert limits["legacy-alias"] == Limits(max_concurrency=3, requests_per_minute=5)
    assert load_limits(tmp_path / "missing.toml") == {}


def test_third_request_waits_for_the_minute_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {"opencode-go": Limits(max_concurrency=2, requests_per_minute=2)},
        clock=clock.now,
        sleep=clock.sleep,
    )
    with limiter.acquire("opencode-go"):
        pass
    with limiter.acquire("opencode-go"):
        pass
    assert clock.value == 0.0
    with limiter.acquire("opencode-go"):
        pass
    assert clock.value == 60.0


def test_window_rolls_forward() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {"c": Limits(max_concurrency=1, requests_per_minute=1)},
        clock=clock.now,
        sleep=clock.sleep,
    )
    with limiter.acquire("c"):
        pass
    clock.value = 30
    with limiter.acquire("c"):
        pass
    assert clock.value == 60.0
    assert limiter.calls_in_window("c") == 1


def test_concurrency_slot_blocks_then_releases() -> None:
    limiter = RateLimiter(
        {"c": Limits(max_concurrency=1, requests_per_minute=100)},
        clock=FakeClock().now,
        sleep=FakeClock().sleep,
    )
    acquired: list[str] = []
    holding = threading.Event()
    release = threading.Event()

    def first() -> None:
        with limiter.acquire("c"):
            acquired.append("first")
            holding.set()
            release.wait(timeout=2)

    def second() -> None:
        with limiter.acquire("c"):
            acquired.append("second")

    first_thread = threading.Thread(target=first)
    first_thread.start()
    assert holding.wait(timeout=2)
    second_thread = threading.Thread(target=second)
    second_thread.start()
    import time

    time.sleep(0.1)
    assert acquired == ["first"]
    release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert acquired == ["first", "second"]


def test_unknown_connection_gets_the_conservative_default() -> None:
    limiter = RateLimiter({})
    assert limiter.limits_for("anything") == Limits(max_concurrency=1, requests_per_minute=60)
