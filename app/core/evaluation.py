"""
EvaluationAgent: Evaluates if a literature review has answered the research topic
sufficiently or requires further rounds of search.
"""
from __future__ import annotations

from app.core.prompts import EVALUATION_SYSTEM
from app.models.schemas import EvaluationResult, LiteratureReview
from app.services.llm_client import LLMClient
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def evaluate(self, topic: str, review: LiteratureReview) -> EvaluationResult:
        user_prompt = (
            f"Original Topic: {topic}\n\n"
            f"Draft Review:\n"
            f"Background: {review.background}\n\n"
            f"Methodology Comparison: {review.methodology_comparison}\n\n"
            f"Key Findings: {review.key_findings}\n\n"
            f"Open Questions: {review.open_questions}\n\n"
            f"Contradictions detected: {len(review.contradictions)}"
        )

        try:
            raw = await self._llm.complete_json(
                system=EVALUATION_SYSTEM, user=user_prompt
            )
            return EvaluationResult(
                passed=bool(raw.get("passed", True)),
                feedback=str(raw.get("feedback", "")),
                follow_up_questions=list(raw.get("follow_up_questions", [])),
            )
        except Exception as exc:
            logger.warning("evaluation_failed error=%s, failing open (passed=True)", exc)
            return EvaluationResult(
                passed=True,
                feedback=f"Failed to evaluate due to exception: {exc}",
                follow_up_questions=[],
            )
