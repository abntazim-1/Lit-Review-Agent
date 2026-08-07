"""
Job store backed by SQLite for persistence across server restarts.

The store exposes the same async interface as before -- nothing outside
this file needs to change. Jobs are JSON-serialized into a single `jobs`
table; the full Pydantic model round-trips cleanly via model_dump_json /
model_validate_json.

Why SQLite and not the existing MemoryStore?
  MemoryStore holds paper/embedding caches that are append-only and never
  updated mid-run. Jobs are heavily mutated during a pipeline run (status,
  logs, results updated on every step). Mixing the two write patterns in
  one connection would require coarse locking; a separate DB file is simpler.

Why this matters:
  uvicorn --reload (used in development) restarts the worker process on
  every file change. With a pure in-memory store every hot-reload wiped
  all running jobs, causing the frontend to poll 404 indefinitely. With
  SQLite the job survives the restart; the pipeline background task however
  does NOT survive (it lives in the old process). The job will therefore
  remain in its last-known status (e.g. 'researching') indefinitely after
  a reload. Future work: move pipeline execution to a persistent worker
  (Celery/ARQ) so the task also survives.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.models.schemas import JobStatus, ReviewJob

_JOB_TTL_SECONDS = 3600  # evict jobs older than 1 hour

# Single-thread executor so all SQLite I/O is serialised (SQLite connections
# are not thread-safe by default without check_same_thread=False).
_DB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job_store")


class JobStore:
    def __init__(self, db_path: str = "./data/jobs.db") -> None:
        self._db_path = db_path
        self._lock = asyncio.Lock()
        # Bootstrap synchronously on init (happens before the event loop
        # starts accepting requests, so blocking here is fine).
        self._init_db()

    # ------------------------------------------------------------------
    # Public async API (identical to the old in-memory interface)
    # ------------------------------------------------------------------

    async def create(self, job: ReviewJob) -> None:
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                _DB_EXECUTOR, self._upsert, job
            )

    async def get(self, job_id: str) -> ReviewJob | None:
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                _DB_EXECUTOR, self._fetch, job_id
            )

    async def save(self, job: ReviewJob) -> None:
        """Persist an updated job. Call this after mutating job.status etc."""
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                _DB_EXECUTOR, self._upsert, job
            )

    # ------------------------------------------------------------------
    # Synchronous DB helpers (run inside the thread-pool executor)
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id    TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    data      TEXT NOT NULL
                )
            """)
            # Evict stale rows from previous sessions.
            cutoff = time.time() - _JOB_TTL_SECONDS
            conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))

    def _upsert(self, job: ReviewJob) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, created_at, data)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET data = excluded.data
                """,
                (job.job_id, job.created_at.timestamp(), job.model_dump_json()),
            )

    def _fetch(self, job_id: str) -> ReviewJob | None:
        cutoff = time.time() - _JOB_TTL_SECONDS
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE job_id = ? AND created_at >= ?",
                (job_id, cutoff),
            ).fetchone()
        if row is None:
            return None
        return ReviewJob.model_validate_json(row[0])

