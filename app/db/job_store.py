"""
In-process job store for tracking async review jobs.

Deliberately in-memory: a review runs for minutes, not days, and this
service is expected to run as a small number of replicas behind a queue
in front of it (see README "Scaling beyond a single process"). If jobs
need to survive a process restart, back this with Redis/Postgres behind
the same `JobStore` interface -- nothing outside this file needs to change.
"""
from __future__ import annotations

import asyncio
import time

from app.models.schemas import ReviewJob

_JOB_TTL_SECONDS = 3600


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ReviewJob] = {}
        self._created_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: ReviewJob) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job
            self._created_at[job.job_id] = time.monotonic()

    async def get(self, job_id: str) -> ReviewJob | None:
        async with self._lock:
            self._evict_expired()
            return self._jobs.get(job_id)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [jid for jid, t in self._created_at.items() if now - t > _JOB_TTL_SECONDS]
        for jid in expired:
            self._jobs.pop(jid, None)
            self._created_at.pop(jid, None)
