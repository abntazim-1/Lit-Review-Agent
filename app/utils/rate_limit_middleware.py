"""Per-client-IP sliding-window rate limiting. Swap for Redis-backed limiting behind a load balancer."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._window_seconds = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client_id]
        while window and now - window[0] > self._window_seconds:
            window.popleft()

        if len(window) >= self._limit:
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again shortly."},
                status_code=429,
            )

        window.append(now)
        return await call_next(request)
