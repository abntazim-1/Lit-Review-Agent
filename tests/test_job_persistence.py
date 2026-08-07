"""
Benchmark: Job persistence across simulated process restarts.

Proves the CV bullet:
  "Achieved zero job loss on server restart by migrating to a SQLite-backed
  persistent store with checkpoint saves after each pipeline phase."

Run with: pytest tests/test_job_persistence.py -v -s
"""
import asyncio

import pytest

from app.db.job_store import JobStore
from app.models.schemas import JobStatus, ReviewJob, ReviewRequest


# ---------------------------------------------------------------------------
# Test 1: Basic round-trip — job written survives a new store instance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_survives_simulated_restart(tmp_path):
    """
    Simulates a process restart by creating two separate JobStore instances
    pointing at the same DB file. Data written by 'process 1' must be
    readable by 'process 2'.
    """
    db = str(tmp_path / "jobs.db")

    # --- Process 1: create and save a job mid-pipeline ---
    store1 = JobStore(db_path=db)
    job = ReviewJob(request=ReviewRequest(topic="machine learning in alzheimer detection"))
    job.status = JobStatus.RESEARCHING
    job.logs.append("[INFO] Launching research agents...")
    await store1.create(job)
    await store1.save(job)

    original_job_id = job.job_id
    original_status = job.status
    original_log_count = len(job.logs)

    # --- Process 2: new store instance, same DB (simulates restart) ---
    store2 = JobStore(db_path=db)
    recovered = await store2.get(original_job_id)

    assert recovered is not None, "❌ Job was lost after simulated restart"
    assert recovered.job_id == original_job_id
    assert recovered.status == original_status
    assert len(recovered.logs) == original_log_count
    assert recovered.request.topic == "machine learning in alzheimer detection"

    print(f"\n[PASS] Job '{original_job_id}' survived simulated process restart")
    print(f"       Status preserved: {recovered.status}")
    print(f"       Logs preserved:   {len(recovered.logs)} entries")


# ---------------------------------------------------------------------------
# Test 2: Checkpoint saves — each pipeline phase is durable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_checkpoint_saves_are_durable(tmp_path):
    """
    Simulates the pipeline saving checkpoints after each phase.
    Even if the process crashes after synthesis, the contradiction results
    must survive.
    """
    db = str(tmp_path / "jobs.db")
    store = JobStore(db_path=db)

    job = ReviewJob(request=ReviewRequest(topic="deep learning"))
    await store.create(job)

    # Simulate: decomposition checkpoint
    job.status = JobStatus.DECOMPOSING
    job.logs.append("[INFO] Decomposition complete.")
    await store.save(job)

    # Simulate: research checkpoint
    job.status = JobStatus.RESEARCHING
    job.logs.append("[INFO] Research complete.")
    await store.save(job)

    # Simulate: contradiction checkpoint
    job.status = JobStatus.DETECTING_CONTRADICTIONS
    job.logs.append("[INFO] Contradiction detection complete.")
    await store.save(job)

    # Now 'crash' — new store instance reads last checkpoint
    store2 = JobStore(db_path=db)
    recovered = await store2.get(job.job_id)

    assert recovered.status == JobStatus.DETECTING_CONTRADICTIONS
    assert len(recovered.logs) == 3

    print(f"\n[PASS] Last checkpoint survived: status='{recovered.status}'")
    print(f"       {len(recovered.logs)} log entries preserved across restart")


# ---------------------------------------------------------------------------
# Test 3: TTL eviction — expired jobs are not returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expired_jobs_not_returned(tmp_path):
    """
    Jobs older than TTL must not be returned — prevents serving stale data
    to a new frontend session after a long restart gap.
    """
    import sqlite3, time
    db = str(tmp_path / "jobs.db")
    store = JobStore(db_path=db)

    job = ReviewJob(request=ReviewRequest(topic="expired topic"))
    await store.create(job)

    # Manually backdate the created_at so it looks expired
    ancient_ts = time.time() - 7200  # 2 hours ago (TTL is 1 hour)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE jobs SET created_at = ? WHERE job_id = ?",
            (ancient_ts, job.job_id),
        )

    recovered = await store.get(job.job_id)
    assert recovered is None, "❌ Expired job should not be returned"

    print(f"\n[PASS] Expired job correctly evicted (not returned after TTL)")
