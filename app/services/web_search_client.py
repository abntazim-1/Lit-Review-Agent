"""
Pluggable web search provider.

The prompt's original design referenced Brave Search; we support it when
a key is configured, but default to DuckDuckGo's HTML endpoint (no key,
no quota) so the system is runnable out of the box. Swapping providers
is a one-line config change (`WEB_SEARCH_PROVIDER=brave`) and touches no
other module -- that's the point of the `WebSearchProvider` interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import asyncio
import httpx
import re

from app.config import Settings
from app.models.schemas import PaperMetadata, SourceType
from app.utils.logging_config import get_logger
from app.utils.resilience import retry_async

logger = get_logger(__name__)


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        ...


class NullSearchProvider(WebSearchProvider):
    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        return []


class DuckDuckGoProvider(WebSearchProvider):
    """Uses the duckduckgo_search library (HTML backend, no API key)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("duckduckgo_search not installed; returning no web results")
            return []

        def _run() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            import asyncio

            results = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            logger.error("ddg_search_failed query=%r error=%s", query, exc)
            return []

        papers = []
        for r in results:
            url = r.get("href") or r.get("link") or ""
            if not url:
                continue
            papers.append(
                PaperMetadata(
                    paper_key=f"web:{url}",
                    title=r.get("title", "(untitled)"),
                    source=SourceType.WEB,
                    url=url,
                    pdf_url=url if url.lower().endswith(".pdf") else None,
                    abstract=r.get("body", ""),
                )
            )
        return papers


class BraveSearchProvider(WebSearchProvider):
    """Brave Search API -- used when BRAVE_API_KEY is configured."""

    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client

    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        headers = {"X-Subscription-Token": self._settings.brave_api_key, "Accept": "application/json"}
        params = {"q": query, "count": max_results}

        async def _call() -> dict:
            resp = await self._http.get(self._ENDPOINT, headers=headers, params=params, timeout=15.0)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await retry_async(
                _call,
                max_attempts=self._settings.http_max_retries,
                base_delay_seconds=self._settings.http_backoff_base_seconds,
                retryable_exceptions=(httpx.HTTPError,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("brave_search_failed query=%r error=%s", query, exc)
            return []

        results = data.get("web", {}).get("results", [])
        papers = []
        for r in results:
            url = r.get("url", "")
            if not url:
                continue
            papers.append(
                PaperMetadata(
                    paper_key=f"web:{url}",
                    title=r.get("title", "(untitled)"),
                    source=SourceType.WEB,
                    url=url,
                    pdf_url=url if url.lower().endswith(".pdf") else None,
                    abstract=r.get("description", ""),
                )
            )
        return papers


class SerpApiProvider(WebSearchProvider):
    """SerpApi Google Scholar API Search."""

    _ENDPOINT = "https://serpapi.com/search"

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client

    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        if not self._settings.serpapi_api_key:
            logger.error("SerpApi API key not configured; returning empty search results")
            return []

        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self._settings.serpapi_api_key,
            "num": max_results,
        }

        async def _call() -> dict:
            resp = await self._http.get(self._ENDPOINT, params=params, timeout=20.0)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await retry_async(
                _call,
                max_attempts=self._settings.http_max_retries,
                base_delay_seconds=self._settings.http_backoff_base_seconds,
                retryable_exceptions=(httpx.HTTPError,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("serpapi_search_failed query=%r error=%s", query, exc)
            return []

        organic_results = data.get("organic_results", [])
        papers = []
        for r in organic_results:
            url = r.get("link") or ""
            if not url:
                continue

            # Check resources for PDF link
            pdf_url = None
            resources = r.get("resources", [])
            for res in resources:
                if res.get("file_format") == "PDF":
                    pdf_url = res.get("link")
                    break

            if not pdf_url and url.lower().endswith(".pdf"):
                pdf_url = url

            # Extract publication year
            summary = r.get("publication_info", {}).get("summary", "")
            match = re.search(r'\b(19\d\d|20\d\d)\b', summary)
            published = match.group(1) if match else None

            papers.append(
                PaperMetadata(
                    paper_key=f"web:{url}",
                    title=r.get("title", "(untitled)"),
                    source=SourceType.WEB,
                    url=url,
                    pdf_url=pdf_url,
                    abstract=r.get("snippet", ""),
                    published=published,
                )
            )
        return papers


class ScholarApiProvider(WebSearchProvider):
    """ScholarAPI.net Google Scholar API Search."""

    _ENDPOINT = "https://scholarapi.net/api/v1/search"

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client

    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        if not self._settings.scholarapi_api_key:
            logger.error("ScholarAPI API key not configured; returning empty search results")
            return []

        headers = {
            "X-API-Key": self._settings.scholarapi_api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "limit": max_results,
        }

        async def _call() -> Any:
            resp = await self._http.get(self._ENDPOINT, headers=headers, params=params, timeout=20.0)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await retry_async(
                _call,
                max_attempts=self._settings.http_max_retries,
                base_delay_seconds=self._settings.http_backoff_base_seconds,
                retryable_exceptions=(httpx.HTTPError,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("scholarapi_search_failed query=%r error=%s", query, exc)
            return []

        # ScholarAPI might return a list directly, or a dict containing list under results/data/publications
        results = []
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            for key in ["results", "data", "items", "publications", "papers", "documents"]:
                val = data.get(key)
                if isinstance(val, list):
                    results = val
                    break
            else:
                # If it's a dict and none of the list keys match, use the dict values if they look like paper items
                if "title" in data or "name" in data:
                    results = [data]

        papers = []
        for r in results:
            if not isinstance(r, dict):
                continue

            url = r.get("link") or r.get("url") or r.get("href") or ""
            if not url:
                doi = r.get("doi")
                if doi:
                    url = f"https://doi.org/{doi}"
                else:
                    continue

            pdf_url = r.get("pdf_url") or r.get("pdf") or r.get("pdfLink") or None
            if not pdf_url and url.lower().endswith(".pdf"):
                pdf_url = url

            abstract = r.get("snippet") or r.get("abstract") or r.get("description") or r.get("summary") or ""

            # Extract publication year
            year = r.get("year") or r.get("publication_year") or r.get("date")
            published = None
            if year:
                match = re.search(r'\b(19\d\d|20\d\d)\b', str(year))
                if match:
                    published = match.group(1)

            papers.append(
                PaperMetadata(
                    paper_key=f"web:{url}",
                    title=r.get("title") or r.get("name") or "(untitled)",
                    source=SourceType.WEB,
                    url=url,
                    pdf_url=pdf_url,
                    abstract=abstract,
                    published=published,
                )
            )
        return papers


class CompositeSearchProvider(WebSearchProvider):
    def __init__(self, providers: list[WebSearchProvider]) -> None:
        self._providers = providers

    async def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        tasks = [p.search(query, max_results) for p in self._providers]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        combined: list[PaperMetadata] = []
        seen_urls: set[str] = set()

        for res in results_list:
            if isinstance(res, Exception):
                logger.error("sub_provider_search_failed error=%s", res)
                continue
            for paper in res:
                if paper.url not in seen_urls:
                    seen_urls.add(paper.url)
                    combined.append(paper)

        return combined


def build_web_search_provider(settings: Settings, http_client: httpx.AsyncClient) -> WebSearchProvider:
    providers = []
    # Split by comma to support composite search
    provider_names = [p.strip().lower() for p in settings.web_search_provider.split(",")]

    for name in provider_names:
        if name == "brave" and settings.brave_api_key:
            providers.append(BraveSearchProvider(settings, http_client))
        elif name == "serpapi" and settings.serpapi_api_key:
            providers.append(SerpApiProvider(settings, http_client))
        elif name == "scholarapi" and settings.scholarapi_api_key:
            providers.append(ScholarApiProvider(settings, http_client))
        elif name == "duckduckgo":
            providers.append(DuckDuckGoProvider(settings))

    if not providers:
        return NullSearchProvider()
    if len(providers) == 1:
        return providers[0]
    return CompositeSearchProvider(providers)
