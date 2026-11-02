"""Per-connection concurrency and requests/minute limits from ``config/budget.toml``.

T08 owns the file; T02 owns the limiter. Shape::

    [limits.opencode-go]
    max_concurrency = 2
    requests_per_minute = 30

``PLAN.md`` §8 asks for at most 2 concurrent requests and 30 requests/minute per model.
Both the clock and the sleep function are injectable so tests run on a fake clock::

    clock = FakeClock()
    limiter = RateLimiter(limits, clock=clock.now, sleep=clock.sleep)
    with limiter.acquire("opencode-go"):
        ...
"""

from __future__ import annotations

import threading
import time
import tomllib
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_BUDGET_PATH",
    "DEFAULT_LIMITS",
    "Limits",
    "RateLimiter",
    "load_limits",
]

DEFAULT_BUDGET_PATH = Path("config/budget.toml")


@dataclass(frozen=True)
class Limits:
    """Concurrency and pacing for one connection."""

    max_concurrency: int = 1
    requests_per_minute: int = 60

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            msg = "max_concurrency must be >= 1"
            raise ValueError(msg)
        if self.requests_per_minute < 1:
            msg = "requests_per_minute must be >= 1"
            raise ValueError(msg)


#: Used when a connection has no entry: one at a time, politely paced.
DEFAULT_LIMITS = Limits()


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def load_limits(path: Path | str = DEFAULT_BUDGET_PATH) -> dict[str, Limits]:
    """Load ``[limits.<connection-id>]``. Missing file or section means no override."""
    budget_path = Path(path)
    if not budget_path.is_file():
        return {}
    with budget_path.open("rb") as handle:
        data = tomllib.load(handle)
    sections = data.get("limits")
    if not isinstance(sections, dict):
        return {}
    limits: dict[str, Limits] = {}
    for connection_id, row in sections.items():
        if not isinstance(connection_id, str) or not isinstance(row, dict):
            continue
        concurrency = row.get("max_concurrency", row.get("max_concurrent", 1))
        per_minute = row.get("requests_per_minute", row.get("rpm", 60))
        try:
            limits[connection_id] = Limits(
                max_concurrency=_positive_int(concurrency, 1),
                requests_per_minute=_positive_int(per_minute, 60),
            )
        except ValueError:  # pragma: no cover - guarded by _positive_int
            continue
    return limits


class _Semaphore:
    """Tiny counting semaphore with an injectable sleep, so tests never block."""

    def __init__(self, permits: int, sleep: Callable[[float], None]) -> None:
        self._permits = permits
        self._sleep = sleep
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                if self._permits > 0:
                    self._permits -= 1
                    return
            self._sleep(0.05)

    def release(self) -> None:
        with self._lock:
            self._permits += 1


class RateLimiter:
    """Concurrency semaphore plus a sliding one-minute window per connection."""

    def __init__(
        self,
        limits: Mapping[str, Limits] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        window_s: float = 60.0,
    ) -> None:
        self._limits = dict(limits or {})
        self._clock = clock
        self._sleep = sleep
        self._window = window_s
        self._lock = threading.Lock()
        self._calls: dict[str, deque[float]] = {}
        self._semaphores: dict[str, _Semaphore] = {}

    def limits_for(self, connection_id: str) -> Limits:
        """The configured limits, or the conservative default."""
        return self._limits.get(connection_id, DEFAULT_LIMITS)

    def _semaphore(self, connection_id: str) -> _Semaphore:
        with self._lock:
            semaphore = self._semaphores.get(connection_id)
            if semaphore is None:
                semaphore = _Semaphore(self.limits_for(connection_id).max_concurrency, self._sleep)
                self._semaphores[connection_id] = semaphore
            return semaphore

    def _wait_for_window(self, connection_id: str) -> None:
        per_minute = self.limits_for(connection_id).requests_per_minute
        while True:
            with self._lock:
                stamps = self._calls.setdefault(connection_id, deque())
                now = self._clock()
                while stamps and now - stamps[0] >= self._window:
                    stamps.popleft()
                if len(stamps) < per_minute:
                    stamps.append(now)
                    return
                wait = self._window - (now - stamps[0])
            self._sleep(max(wait, 0.0))

    @contextmanager
    def acquire(self, connection_id: str) -> Iterator[None]:
        """Hold one concurrency slot and one window slot for the duration."""
        semaphore = self._semaphore(connection_id)
        semaphore.acquire()
        try:
            self._wait_for_window(connection_id)
            yield
        finally:
            semaphore.release()

    def calls_in_window(self, connection_id: str) -> int:
        """How many starts are currently inside the window (for tests and logs)."""
        with self._lock:
            stamps = self._calls.get(connection_id, deque())
            now = self._clock()
            return sum(1 for stamp in stamps if now - stamp < self._window)
