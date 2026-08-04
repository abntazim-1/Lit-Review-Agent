"""
Resilience primitives: retry-with-backoff and a simple async rate limiter.

Kept dependency-free (no tenacity) so the core logic is auditable in one
place -- this is the kind of code a reviewer should be able to read in
30 seconds and trust.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay_seconds: float,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """
    Exponential backoff with full jitter.

    Attempt 1 runs immediately; failures wait base * 2^(n-1) +/- jitter.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retryable_exceptions as exc:  # noqa: BLE001 - intentional broad catch at boundary
            last_exc = exc
            if on_retry:
                on_retry(attempt, exc)
            if attempt == max_attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            delay = random.uniform(0, delay)
            await asyncio.sleep(delay)
    raise RetryError(f"Exhausted {max_attempts} attempts") from last_exc


class AsyncRateLimiter:
    """
    Enforces a minimum interval between calls, shared across concurrent
    coroutines via a lock. Used to respect ArXiv's "no more than one
    request every 3 seconds" guidance even when multiple research agents
    run in parallel.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call_at: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_at = time.monotonic()


class ConcurrencySemaphorePool:
    """Named semaphores so different resources (arxiv, pdf, llm) get independent caps."""

    def __init__(self) -> None:
        self._pool: dict[str, asyncio.Semaphore] = {}

    def get(self, name: str, limit: int) -> asyncio.Semaphore:
        if name not in self._pool:
            self._pool[name] = asyncio.Semaphore(limit)
        return self._pool[name]
