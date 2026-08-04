"""
Domain models shared across the pipeline: API I/O schemas plus the
internal data structures agents pass to each other. Keeping these in one
module means every stage of the pipeline speaks the same typed language,
which is what makes the "structured" in structured review possible.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# API request / response
# --------------------------------------------------------------------------- #

class ReviewRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    max_sub_questions: Optional[int] = Field(default=None, ge=1, le=8)
    max_papers_per_sub_question: Optional[int] = Field(default=None, ge=1, le=20)


class JobStatus(str, Enum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    RESEARCHING = "researching"
    DETECTING_CONTRADICTIONS = "detecting_contradictions"
    SYNTHESIZING = "synthesizing"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"


class JobHandle(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime


# --------------------------------------------------------------------------- #
# Internal pipeline models
# --------------------------------------------------------------------------- #

class SubQuestion(BaseModel):
    id: str = Field(default_factory=_uid)
    text: str
    rationale: str = ""


class ResearchCluster(BaseModel):
    id: str = Field(default_factory=_uid)
    theme: str
    sub_questions: list[SubQuestion]


class EvaluationResult(BaseModel):
    passed: bool
    feedback: str
    follow_up_questions: list[str] = Field(default_factory=list)


class SourceType(str, Enum):
    ARXIV = "arxiv"
    WEB = "web"


class PaperMetadata(BaseModel):
    """Uniquely identified by `paper_key` (arxiv id or normalized URL)."""
    paper_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: Optional[str] = None
    source: SourceType
    url: str
    pdf_url: Optional[str] = None
    abstract: str = ""


class ExtractedClaim(BaseModel):
    claim: str
    evidence: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class PaperFindings(BaseModel):
    paper: PaperMetadata
    sub_question_id: str
    full_text_chars_used: int = 0
    methodology_summary: str = ""
    claims: list[ExtractedClaim] = Field(default_factory=list)
    limitations: str = ""
    extraction_failed: bool = False
    failure_reason: str = ""


class Contradiction(BaseModel):
    topic: str
    paper_a_key: str
    paper_a_claim: str
    paper_b_key: str
    paper_b_claim: str
    explanation: str


class ResearchAgentResult(BaseModel):
    sub_question: SubQuestion
    findings: list[PaperFindings] = Field(default_factory=list)
    papers_searched: int = 0
    papers_used: int = 0
    errors: list[str] = Field(default_factory=list)


class LiteratureReview(BaseModel):
    topic: str
    background: str
    methodology_comparison: str
    key_findings: str
    open_questions: str
    contradictions: list[Contradiction] = Field(default_factory=list)
    references: list[PaperMetadata] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)


class ReviewJob(BaseModel):
    job_id: str = Field(default_factory=_uid)
    request: ReviewRequest
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    clusters: list[ResearchCluster] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    agent_results: list[ResearchAgentResult] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    result: Optional[LiteratureReview] = None
    error: Optional[str] = None
    logs: list[str] = Field(default_factory=list)
