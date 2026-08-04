"""
Fetches a paper's PDF and extracts plain text for claim extraction.

Defensive by design: papers fail to download, PDFs are sometimes scans
with no text layer, and some are hundreds of pages. None of that should
crash a research agent -- it should degrade to "use the abstract only"
and say so in `PaperFindings.extraction_failed`.
"""
from __future__ import annotations

import io

import httpx
import pdfplumber

from app.config import Settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PdfFetchError(Exception):
    pass


async def fetch_and_extract_text(
    pdf_url: str,
    *,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> str:
    """Returns extracted text, truncated to `pdf_max_chars_for_llm`. Raises PdfFetchError on failure."""
    try:
        async with http_client.stream(
            "GET", pdf_url, timeout=settings.pdf_fetch_timeout_seconds, follow_redirects=True
        ) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                raise PdfFetchError(f"URL did not return a PDF (content-type={content_type!r})")

            chunks = bytearray()
            async for chunk in resp.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > settings.pdf_max_bytes:
                    raise PdfFetchError(f"PDF exceeds {settings.pdf_max_bytes} byte safety cap")
    except httpx.HTTPError as exc:
        raise PdfFetchError(f"HTTP error fetching PDF: {exc}") from exc

    try:
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(bytes(chunks))) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                if sum(len(t) for t in text_parts) > settings.pdf_max_chars_for_llm:
                    break
    except Exception as exc:  # noqa: BLE001 - pdfplumber raises many exception types for malformed PDFs
        raise PdfFetchError(f"Failed to parse PDF: {exc}") from exc

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise PdfFetchError("No extractable text (likely a scanned image PDF)")

    return full_text[: settings.pdf_max_chars_for_llm]
