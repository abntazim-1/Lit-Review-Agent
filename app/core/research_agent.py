"""
Research agent.

One instance handles exactly one sub-question end to end: search ArXiv
and the web, dedupe against papers already seen (in this job and in the
persistent memory store), fetch full text where possible, and extract
structured claims via the LLM. N of these run concurrently under a
semaphore cap (`max_concurrent_research_agents`) -- that concurrency
cap, not a thread-per-agent model, is what keeps this safe to run
against rate-limited upstreams.
"""
from __future__ import annotations

import asyncio
import datetime
import re

import httpx
import numpy as np

from app.config import Settings
from app.core.prompts import CLAIM_EXTRACTION_SYSTEM
from app.db.memory_store import MemoryStore
from app.models.schemas import ExtractedClaim, PaperFindings, PaperMetadata, ResearchAgentResult, SubQuestion, ResearchCluster
from app.services.arxiv_client import ArxivClient
from app.services.embeddings import EmbeddingService
from app.services.llm_client import LLMClient, LLMError
from app.services.pdf_fetcher import PdfFetchError, fetch_and_extract_text
from app.services.web_search_client import WebSearchProvider
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ResearchAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        arxiv: ArxivClient,
        web_search: WebSearchProvider,
        embeddings: EmbeddingService,
        memory: MemoryStore,
        http_client: httpx.AsyncClient,
        pdf_semaphore: asyncio.Semaphore,
        llm_semaphore: asyncio.Semaphore,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._arxiv = arxiv
        self._web_search = web_search
        self._embeddings = embeddings
        self._memory = memory
        self._http = http_client
        self._pdf_semaphore = pdf_semaphore
        self._llm_semaphore = llm_semaphore

    async def run(self, cluster: ResearchCluster, seen_paper_keys: set[str]) -> list[ResearchAgentResult]:
        results = []
        cluster_seen: set[str] = set()

        current_year = datetime.datetime.now().year
        cutoff_year = current_year - 5

        for sq in cluster.sub_questions:
            result = ResearchAgentResult(sub_question=sq)

            arxiv_papers, web_papers = await asyncio.gather(
                self._arxiv.search(sq.text, self._settings.arxiv_max_results_per_query),
                self._web_search.search(sq.text, self._settings.web_search_max_results),
            )
            candidates = arxiv_papers + web_papers
            result.papers_searched = len(candidates)

            # Filter candidates by publication year (last 5 years)
            filtered_candidates = []
            for paper in candidates:
                if paper.published:
                    match = re.search(r'\b(19\d\d|20\d\d)\b', paper.published)
                    if match:
                        year = int(match.group(1))
                        if year < cutoff_year:
                            logger.info("filtering_out_old_paper key=%s title=%r year=%s", paper.paper_key, paper.title, year)
                            continue
                filtered_candidates.append(paper)

            combined_seen = seen_paper_keys | cluster_seen
            deduped = await self._dedupe(filtered_candidates, combined_seen)
            selected = deduped[: self._settings.max_papers_per_sub_question]

            extraction_tasks = [self._process_paper(paper, sq.id) for paper in selected]
            findings_list = await asyncio.gather(*extraction_tasks, return_exceptions=True)

            for paper, outcome in zip(selected, findings_list):
                if isinstance(outcome, Exception):
                    result.errors.append(f"{paper.paper_key}: {outcome}")
                    logger.warning("paper_processing_failed paper=%s error=%s", paper.paper_key, outcome)
                    continue
                result.findings.append(outcome)
                seen_paper_keys.add(paper.paper_key)
                cluster_seen.add(paper.paper_key)

            result.papers_used = len(result.findings)
            results.append(result)

        return results

    async def _dedupe(self, candidates: list[PaperMetadata], seen_paper_keys: set[str]) -> list[PaperMetadata]:
        # Exact-key dedup first (cheap, catches the common case: same paper found twice).
        unique_by_key: dict[str, PaperMetadata] = {}
        for paper in candidates:
            if paper.paper_key in seen_paper_keys or paper.paper_key in unique_by_key:
                continue
            unique_by_key[paper.paper_key] = paper
        remaining = list(unique_by_key.values())
        if len(remaining) <= 1:
            return remaining

        # Semantic dedup second: catches the same paper surfaced via ArXiv and via a
        # web mirror/blog repost with a differently-worded title.
        texts = [f"{p.title}. {p.abstract[:300]}" for p in remaining]
        try:
            vectors = await self._embeddings.embed(texts)
        except Exception as exc:  # noqa: BLE001 - embedding model issues shouldn't kill the pipeline
            logger.warning("embedding_dedupe_skipped error=%s", exc)
            return remaining

        sims = EmbeddingService.cosine_similarity_matrix(vectors, vectors)
        keep_mask = np.ones(len(remaining), dtype=bool)
        threshold = self._settings.dedup_similarity_threshold
        for i in range(len(remaining)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(remaining)):
                if keep_mask[j] and sims[i, j] >= threshold:
                    keep_mask[j] = False  # drop the later duplicate

        deduped = [p for p, keep in zip(remaining, keep_mask) if keep]
        for paper, vec in zip(remaining, vectors):
            self._memory.upsert_paper(paper, embedding=vec)
        return deduped

    async def _process_paper(self, paper: PaperMetadata, sub_question_id: str) -> PaperFindings:
        cached = self._memory.get_cached_findings(paper.paper_key)
        if cached is not None:
            logger.info("cache_hit paper=%s", paper.paper_key)
            return PaperFindings(paper=paper, sub_question_id=sub_question_id, **cached)

        text, chars_used, failure = await self._get_text(paper)
        async with self._llm_semaphore:
            try:
                extracted = await self._llm.complete_json(
                    system=CLAIM_EXTRACTION_SYSTEM,
                    user=f"Paper title: {paper.title}\n\nText:\n{text}",
                )
                findings = PaperFindings(
                    paper=paper,
                    sub_question_id=sub_question_id,
                    full_text_chars_used=chars_used,
                    methodology_summary=extracted.get("methodology_summary", ""),
                    claims=[ExtractedClaim(**c) for c in extracted.get("claims", [])],
                    limitations=extracted.get("limitations", ""),
                    extraction_failed=False,
                    failure_reason=failure,
                )
            except (LLMError, KeyError, TypeError) as exc:
                findings = PaperFindings(
                    paper=paper,
                    sub_question_id=sub_question_id,
                    full_text_chars_used=chars_used,
                    extraction_failed=True,
                    failure_reason=f"claim extraction failed: {exc}",
                )

        self._memory.save_findings(findings)
        return findings

    async def _get_text(self, paper: PaperMetadata) -> tuple[str, int, str]:
        """Prefer full PDF text; gracefully fall back to the abstract."""
        if paper.pdf_url:
            async with self._pdf_semaphore:
                try:
                    text = await fetch_and_extract_text(
                        paper.pdf_url, http_client=self._http, settings=self._settings
                    )
                    return text, len(text), ""
                except PdfFetchError as exc:
                    logger.info("pdf_fetch_fallback paper=%s reason=%s", paper.paper_key, exc)

        abstract = paper.abstract or paper.title
        return abstract, len(abstract), "used abstract only (full text unavailable)"
