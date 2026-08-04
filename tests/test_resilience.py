import time

import pytest

from app.utils.resilience import AsyncRateLimiter, RetryError, retry_async


@pytest.mark.asyncio
async def test_retry_async_succeeds_after_transient_failures():
    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("transient")
        return "ok"

    result = await retry_async(flaky, max_attempts=5, base_delay_seconds=0.01)

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_retry_async_raises_retry_error_after_exhausting_attempts():
    async def always_fails():
        raise ValueError("permanent")

    with pytest.raises(RetryError):
        await retry_async(always_fails, max_attempts=3, base_delay_seconds=0.01)


@pytest.mark.asyncio
async def test_rate_limiter_enforces_minimum_interval():
    limiter = AsyncRateLimiter(min_interval_seconds=0.05)

    start = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.04
