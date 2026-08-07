"""Per-client-IP sliding-window rate limiting. Swap for Redis-backed limiting behind a load balancer.

Design intent:
- Only WRITE operations (POST, PUT, PATCH, DELETE) count against the rate limit bucket.
  These are the expensive calls that trigger LLM pipelines and external API usage.
- READ operations (GET) are excluded from rate limiting entirely. Status-polling endpoints
  like GET /reviews/{job_id} must never compete with submission rate limits -- they are
  cheap DB reads that need to remain responsive while the pipeline is running.
- This mirrors the industry-standard approach (e.g. GitHub API: separate read/write limits).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Methods that mutate state and trigger expensive work (LLM calls, ArXiv fetches, etc.)
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._window_seconds = 60.0
        # Separate buckets per client IP so one noisy client can't block others.
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Read-only requests (status polling, health checks, static files) are
        # never rate-limited -- they're cheap and must stay responsive.
        if request.method not in _WRITE_METHODS:
            return await call_next(request)

        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client_id]

        # Evict timestamps outside the current sliding window.
        while window and now - window[0] > self._window_seconds:
            window.popleft()

        if len(window) >= self._limit:
            return JSONResponse(
                {"detail": "Rate limit exceeded. You may submit a new review shortly."},
                status_code=429,
            )

        window.append(now)
        return await call_next(request)
