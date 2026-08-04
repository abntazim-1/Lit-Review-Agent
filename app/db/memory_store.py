"""
Persistent memory: paper metadata, extracted findings, and their
embeddings, keyed by `paper_key` so repeat queries across different
review jobs reuse prior extraction work instead of re-fetching and
re-parsing the same PDF.

SQLite is intentional here, not a placeholder for "a real database
later": a single review agent doesn't need a clustered store, and
SQLite gives us ACID writes, zero ops, and a file you can back up with
`cp`. Swap the DSN if this ever needs to scale out.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np

from app.models.schemas import ExtractedClaim, PaperFindings, PaperMetadata, SourceType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_key TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    published TEXT,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    pdf_url TEXT,
    abstract TEXT,
    embedding BLOB,
    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS findings (
    paper_key TEXT PRIMARY KEY REFERENCES papers(paper_key),
    methodology_summary TEXT,
    claims_json TEXT NOT NULL,
    limitations TEXT,
    extraction_failed INTEGER NOT NULL DEFAULT 0,
    failure_reason TEXT,
    cached_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
"""


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- papers -------------------------------------------------------- #

    def upsert_paper(self, paper: PaperMetadata, embedding: Optional[np.ndarray] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (paper_key, title, authors, published, source, url, pdf_url, abstract, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_key) DO UPDATE SET
                    title=excluded.title, abstract=excluded.abstract, embedding=COALESCE(excluded.embedding, papers.embedding)
                """,
                (
                    paper.paper_key,
                    paper.title,
                    json.dumps(paper.authors),
                    paper.published,
                    paper.source.value,
                    paper.url,
                    paper.pdf_url,
                    paper.abstract,
                    embedding.astype(np.float32).tobytes() if embedding is not None else None,
                ),
            )

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_key, embedding FROM papers WHERE embedding IS NOT NULL"
            ).fetchall()
        return [(key, np.frombuffer(blob, dtype=np.float32)) for key, blob in rows]

    # -- findings (extraction cache) ------------------------------------ #

    def get_cached_findings(self, paper_key: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT methodology_summary, claims_json, limitations, extraction_failed, failure_reason "
                "FROM findings WHERE paper_key = ?",
                (paper_key,),
            ).fetchone()
        if row is None:
            return None
        methodology_summary, claims_json, limitations, extraction_failed, failure_reason = row
        return {
            "methodology_summary": methodology_summary or "",
            "claims": [ExtractedClaim(**c) for c in json.loads(claims_json)],
            "limitations": limitations or "",
            "extraction_failed": bool(extraction_failed),
            "failure_reason": failure_reason or "",
        }

    def save_findings(self, findings: PaperFindings) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO findings (paper_key, methodology_summary, claims_json, limitations, extraction_failed, failure_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_key) DO UPDATE SET
                    methodology_summary=excluded.methodology_summary,
                    claims_json=excluded.claims_json,
                    limitations=excluded.limitations,
                    extraction_failed=excluded.extraction_failed,
                    failure_reason=excluded.failure_reason
                """,
                (
                    findings.paper.paper_key,
                    findings.methodology_summary,
                    json.dumps([c.model_dump() for c in findings.claims]),
                    findings.limitations,
                    int(findings.extraction_failed),
                    findings.failure_reason,
                ),
            )

    def paper_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
