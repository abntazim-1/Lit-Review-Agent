"""
Benchmark: Rate-limit middleware correctness and read/write bucket separation.

Proves the CV bullet:
  "Improved API resilience under concurrent load by separating read and write
  rate-limit buckets using a custom Starlette middleware, ensuring
  status-polling endpoints (GET) never compete with expensive LLM-triggering
  write operations."

Run with: pytest tests/test_rate_limit_correctness.py -v -s

Tests the middleware directly -- bypasses FastAPI app state entirely so no
lifespan/Container setup is needed.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.utils.rate_limit_middleware import RateLimitMiddleware


# ---------------------------------------------------------------------------
# Minimal test app — just enough to exercise the middleware
# ---------------------------------------------------------------------------

async def _get_handler(request: Request):
    return JSONResponse({"method": "GET"}, status_code=200)


async def _post_handler(request: Request):
    return JSONResponse({"method": "POST"}, status_code=202)


def _make_test_app(rpm: int = 10):
    """Starlette app with rate-limit middleware, NO FastAPI lifespan needed."""
    app = Starlette(routes=[
        Route("/poll/{job_id}", _get_handler, methods=["GET"]),
        Route("/submit",        _post_handler, methods=["POST"]),
    ])
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)
    return app


def _make_client(rpm: int = 10):
    return AsyncClient(
        transport=ASGITransport(app=_make_test_app(rpm)),
        base_url="http://test"
    )


# ---------------------------------------------------------------------------
# Test 1: GET requests are NEVER rate-limited, regardless of volume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_polling_never_rate_limited():
    """
    100 consecutive GET /poll/{id} requests must ALL return 200.
    429 must never appear -- GET is excluded from the rate-limit bucket.
    """
    async with _make_client(rpm=5) as client:  # very tight limit of 5 RPM
        responses = [
            (await client.get("/poll/fake-job-id")).status_code
            for _ in range(100)
        ]

    status_counts = {s: responses.count(s) for s in set(responses)}
    got_429 = 429 in status_counts

    print(f"\n--- GET /poll/... over 100 requests (RPM limit=5) ---")
    for status, count in sorted(status_counts.items()):
        print(f"  HTTP {status}: {count}x")

    assert not got_429, (
        f"FAIL: GET requests were rate-limited {status_counts.get(429,0)} times "
        f"-- status polling would break during a live pipeline run."
    )
    print("[PASS] GET never returned 429 across 100 requests")
    print("       CV claim: 'status-polling endpoints (GET) never compete with write operations'")


# ---------------------------------------------------------------------------
# Test 2: POST requests ARE rate-limited after the threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_rate_limited_after_threshold():
    """
    POST /submit must return 429 after exceeding the RPM limit.
    Records the exact request number where rate-limiting kicks in.
    """
    rpm_limit = 10
    async with _make_client(rpm=rpm_limit) as client:
        responses = []
        for i in range(rpm_limit + 5):  # fire more than the limit
            r = await client.post("/submit", json={})
            responses.append((i + 1, r.status_code))

    status_codes = [s for _, s in responses]
    has_429 = 429 in status_codes
    first_429_at = next((n for n, s in responses if s == 429), None)
    allowed_before_limit = first_429_at - 1 if first_429_at else len(responses)

    print(f"\n--- POST /submit over {len(responses)} rapid requests (RPM limit={rpm_limit}) ---")
    for req_num, status in responses:
        marker = " <-- first 429" if req_num == first_429_at else ""
        print(f"  Request {req_num:02d}: HTTP {status}{marker}")

    assert has_429, "FAIL: Rate limiter never fired -- middleware may not be active."
    assert allowed_before_limit <= rpm_limit, (
        f"FAIL: Allowed {allowed_before_limit} requests before limiting, expected <= {rpm_limit}"
    )
    print(f"\n[PASS] Rate limit enforced at request #{first_429_at}")
    print(f"       Requests allowed before limiting: {allowed_before_limit}/{rpm_limit}")


# ---------------------------------------------------------------------------
# Test 3: Read bucket does NOT drain write budget (isolation proof)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_flood_does_not_consume_post_budget():
    """
    Core isolation proof:
    - Flood with 50 GETs against a tight 5-RPM limit
    - Then fire 3 POSTs -- they must NOT be 429'd due to the GET flood
    - If GET consumed the rate-limit bucket, all POSTs would fail

    This is the single most important test for the CV bullet.
    """
    rpm_limit = 5  # very tight, so any cross-bucket leakage would show immediately
    async with _make_client(rpm=rpm_limit) as client:

        # Step 1: Flood with 50 GETs (WELL above the 5-RPM limit)
        get_responses = [
            (await client.get("/poll/isolation-test")).status_code
            for _ in range(50)
        ]

        # Step 2: Now fire POSTs -- should succeed (GET used none of the POST budget)
        post_responses = [
            (await client.post("/submit", json={})).status_code
            for _ in range(3)
        ]

    get_429s = get_responses.count(429)
    post_non_429 = [s for s in post_responses if s != 429]

    print(f"\n--- Isolation test: 50 GETs then 3 POSTs (RPM limit={rpm_limit}) ---")
    print(f"  GET results:  {set(get_responses)} (429s: {get_429s})")
    print(f"  POST results: {post_responses}")

    # GETs should NEVER be 429 (regardless of volume)
    assert get_429s == 0, f"FAIL: {get_429s} GET requests were rate-limited"

    # POSTs within limit should succeed (budget untouched by GETs)
    assert len(post_non_429) == 3, (
        f"FAIL: Only {len(post_non_429)}/3 POSTs succeeded after 50-GET flood. "
        f"GET requests may have consumed the write budget."
    )

    print(f"[PASS] All 50 GETs: 0 rate-limited (budget isolation confirmed)")
    print(f"[PASS] All 3 POSTs: succeeded after GET flood")
    print(f"       CV claim: read and write rate-limit buckets are fully isolated")
