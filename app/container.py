"""
Composition root. One place that knows how to construct every service
and wire dependencies together -- no service reaches into globals or
constructs its own collaborators. Makes the whole system testable by
swapping this file's wiring in unit tests (e.g. inject a fake LLMClient).
"""
from __future__ import annotations

import asyncio

import httpx

from app.config import Settings, get_settings
from app.core.contradiction_detector import ContradictionDetector
from app.core.evaluation import EvaluationAgent
from app.core.orchestrator import Orchestrator
from app.core.pipeline import ReviewPipeline
from app.core.research_agent import ResearchAgent
from app.core.synthesis import SynthesisAgent
from app.db.memory_store import MemoryStore
from app.services.arxiv_client import ArxivClient
from app.services.embeddings import EmbeddingService
from app.services.llm_client import LLMClient
from app.services.web_search_client import build_web_search_provider


class Container:
    """Long-lived singletons for the process lifetime of the FastAPI app."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.http_client = httpx.AsyncClient(
            headers={"User-Agent": f"{self.settings.app_name}/1.0 (mailto:research-agent@example.com)"},
            follow_redirects=True,
        )
        self.llm = LLMClient(self.settings)
        self.arxiv = ArxivClient(self.settings, self.http_client)
        self.web_search = build_web_search_provider(self.settings, self.http_client)
        self.embeddings = EmbeddingService(self.settings)
        self.memory = MemoryStore(self.settings.sqlite_path)

        # LLM concurrency is intentionally kept very low (3 slots).
        # Groq's free tier allows ~30 requests per minute (RPM) and ~14,400 TPM.
        # With 5 research agents × 8 papers each = up to 40 extraction calls, running
        # 10 concurrently (the old value) saturates both quotas in seconds and causes
        # synthesis to wait many minutes for the TPM window to reset.
        # 3 concurrent LLM calls ≈ 1 call every ~2s at peak, staying well under 30 RPM
        # and giving synthesis a clear quota window when research finishes.
        self._pdf_semaphore = asyncio.Semaphore(4)
        self._llm_semaphore = asyncio.Semaphore(3)

        self.orchestrator = Orchestrator(self.llm, self.settings.max_sub_questions)
        self.contradiction_detector = ContradictionDetector(self.llm)
        self.synthesis_agent = SynthesisAgent(self.llm)
        self.evaluation_agent = EvaluationAgent(self.llm)

        self.pipeline = ReviewPipeline(
            settings=self.settings,
            orchestrator=self.orchestrator,
            research_agent_factory=self._build_research_agent,
            contradiction_detector=self.contradiction_detector,
            synthesis_agent=self.synthesis_agent,
            evaluation_agent=self.evaluation_agent,
        )

    def _build_research_agent(self) -> ResearchAgent:
        return ResearchAgent(
            settings=self.settings,
            llm=self.llm,
            arxiv=self.arxiv,
            web_search=self.web_search,
            embeddings=self.embeddings,
            memory=self.memory,
            http_client=self.http_client,
            pdf_semaphore=self._pdf_semaphore,
            llm_semaphore=self._llm_semaphore,
        )

    async def aclose(self) -> None:
        await self.http_client.aclose()
