import pytest

from app.config import Settings
from app.core.contradiction_detector import ContradictionDetector
from app.models.schemas import (
    ExtractedClaim,
    PaperFindings,
    PaperMetadata,
    ResearchAgentResult,
    SourceType,
    SubQuestion,
)
from app.services.llm_client import LLMClient


def _paper(key: str) -> PaperMetadata:
    return PaperMetadata(paper_key=key, title=f"Paper {key}", source=SourceType.ARXIV, url="http://x")


def _findings(key: str, claim: str, sub_question_id: str) -> PaperFindings:
    return PaperFindings(
        paper=_paper(key),
        sub_question_id=sub_question_id,
        claims=[ExtractedClaim(claim=claim, confidence=0.8)],
    )


@pytest.mark.asyncio
async def test_detect_skips_sub_questions_with_fewer_than_two_usable_papers(monkeypatch):
    settings = Settings(anthropic_api_key="test")
    detector = ContradictionDetector(LLMClient(settings))

    sq = SubQuestion(text="Does X improve Y?")
    result = ResearchAgentResult(sub_question=sq, findings=[_findings("a", "X improves Y", sq.id)])

    contradictions = await detector.detect([result])

    assert contradictions == []


@pytest.mark.asyncio
async def test_detect_filters_out_hallucinated_paper_keys(monkeypatch):
    settings = Settings(anthropic_api_key="test")
    detector = ContradictionDetector(LLMClient(settings))

    sq = SubQuestion(text="Does X improve Y?")
    result = ResearchAgentResult(
        sub_question=sq,
        findings=[
            _findings("a", "X improves Y by 10%", sq.id),
            _findings("b", "X has no effect on Y", sq.id),
        ],
    )

    async def fake_complete_json(**kwargs):
        return [
            {
                "paper_a_key": "a",
                "paper_a_claim": "X improves Y by 10%",
                "paper_b_key": "b",
                "paper_b_claim": "X has no effect on Y",
                "explanation": "Direct disagreement on effect of X.",
            },
            {
                # Hallucinated key that doesn't correspond to any real paper -- must be dropped.
                "paper_a_key": "nonexistent",
                "paper_a_claim": "made up",
                "paper_b_key": "b",
                "paper_b_claim": "also made up",
                "explanation": "should be filtered",
            },
        ]

    monkeypatch.setattr(detector._llm, "complete_json", fake_complete_json)

    contradictions = await detector.detect([result])

    assert len(contradictions) == 1
    assert contradictions[0].paper_a_key == "a"
    assert contradictions[0].paper_b_key == "b"
