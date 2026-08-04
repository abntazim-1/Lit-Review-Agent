"""
FastAPI entry point.

Two endpoints: submit a review job (kicks off the pipeline as a
background task and returns immediately with a job id) and poll for its
status/result. Long-running LLM-and-network-bound work has no business
sitting behind a synchronous HTTP request -- a client polling every few
seconds is the honest contract for a job that can take several minutes.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.container import Container
from app.db.job_store import JobStore
from app.models.schemas import JobHandle, ReviewJob, ReviewRequest
from app.utils.logging_config import configure_logging, get_logger
from app.utils.rate_limit_middleware import RateLimitMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = Container()
    configure_logging(container.settings.log_level, container.settings.log_format)
    app.state.container = container
    app.state.job_store = JobStore()
    logger.info("startup_complete papers_cached=%s", container.memory.paper_count())
    yield
    await container.aclose()


app = FastAPI(
    title="Academic Literature Review Agent",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=20)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def read_root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.post("/reviews", response_model=JobHandle, status_code=202)
async def submit_review(request: ReviewRequest, background_tasks: BackgroundTasks) -> JobHandle:
    container: Container = app.state.container
    job_store = app.state.job_store

    job = ReviewJob(request=request)
    await job_store.create(job)

    async def _run():
        try:
            await container.pipeline.run(job)
        except Exception:  # noqa: BLE001 - pipeline.run already catches internally; this is a last-resort net
            logger.exception("unhandled_pipeline_error job_id=%s", job.job_id)

    background_tasks.add_task(_run)

    return JobHandle(job_id=job.job_id, status=job.status, created_at=job.created_at)


@app.get("/reviews/{job_id}", response_model=ReviewJob)
async def get_review(job_id: str) -> ReviewJob:
    job_store = app.state.job_store
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (or expired)")
    return job


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.error("unhandled_exception path=%s error=%s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
