"""
ArXiv search client.

Uses the public ArXiv API (Atom/XML feed) directly -- no API key
required. Respects ArXiv's documented etiquette: a single caller-wide
rate limiter enforces a minimum gap between requests even when several
research agents query concurrently.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

import httpx

from app.config import Settings
from app.models.schemas import PaperMetadata, SourceType
from app.utils.logging_config import get_logger
from app.utils.resilience import AsyncRateLimiter, retry_async

logger = get_logger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client
        self._limiter = AsyncRateLimiter(settings.arxiv_min_request_interval_seconds)

    async def search(self, query: str, max_results: int | None = None) -> list[PaperMetadata]:
        settings = self._settings
        max_results = max_results or settings.arxiv_max_results_per_query
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        async def _call() -> str:
            await self._limiter.acquire()
            resp = await self._http.get(settings.arxiv_base_url, params=params, timeout=20.0)
            resp.raise_for_status()
            return resp.text

        try:
            xml_text = await retry_async(
                _call,
                max_attempts=settings.http_max_retries,
                base_delay_seconds=settings.http_backoff_base_seconds,
                retryable_exceptions=(httpx.HTTPError,),
                on_retry=lambda a, e: logger.warning("arxiv_retry attempt=%s error=%s", a, e),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("arxiv_search_failed query=%r error=%s", query, exc)
            return []

        return self._parse_feed(xml_text)

    def _parse_feed(self, xml_text: str) -> list[PaperMetadata]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("arxiv_parse_failed error=%s", exc)
            return []

        papers: list[PaperMetadata] = []
        for entry in root.findall(f"{_ATOM_NS}entry"):
            entry_id = _text(entry, f"{_ATOM_NS}id")
            if not entry_id:
                continue
            arxiv_id = entry_id.rsplit("/", 1)[-1]
            title = " ".join(_text(entry, f"{_ATOM_NS}title").split())
            summary = " ".join(_text(entry, f"{_ATOM_NS}summary").split())
            published = _text(entry, f"{_ATOM_NS}published")
            authors = [
                _text(author, f"{_ATOM_NS}name")
                for author in entry.findall(f"{_ATOM_NS}author")
            ]
            pdf_url = None
            for link in entry.findall(f"{_ATOM_NS}link"):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href")
                    break
            if pdf_url is None:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            papers.append(
                PaperMetadata(
                    paper_key=f"arxiv:{arxiv_id.split('v')[0]}",
                    title=title or "(untitled)",
                    authors=[a for a in authors if a],
                    published=published[:10] if published else None,
                    source=SourceType.ARXIV,
                    url=entry_id,
                    pdf_url=pdf_url,
                    abstract=summary,
                )
            )
        return papers


def _text(elem: ET.Element, tag: str) -> str:
    node = elem.find(tag)
    return (node.text or "").strip() if node is not None else ""
