"""
Contradiction detection.

Runs per sub-question (comparing papers that answer the same question is
meaningful; comparing across unrelated sub-questions mostly isn't) and
only when at least 2 papers produced usable claims. The LLM is asked to
be conservative -- different scope/dataset/setting is not a contradiction,
only genuine incompatible claims are flagged. This runs after all
research agents complete because contradiction detection is inherently
cross-cutting: it needs the full findings set for a sub-question, not a
single paper's view of it.
"""
from __future__ import annotations

from app.core.prompts import CONTRADICTION_SYSTEM
from app.models.schemas import Contradiction, ResearchAgentResult
from app.services.llm_client import LLMClient, LLMError
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ContradictionDetector:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def detect(self, agent_results: list[ResearchAgentResult]) -> list[Contradiction]:
        all_contradictions: list[Contradiction] = []
        for agent_result in agent_results:
            usable = [f for f in agent_result.findings if not f.extraction_failed and f.claims]
            if len(usable) < 2:
                continue

            payload = {
                "sub_question": agent_result.sub_question.text,
                "papers": [
                    {
                        "paper_key": f.paper.paper_key,
                        "title": f.paper.title,
                        "claims": [c.claim for c in f.claims],
                    }
                    for f in usable
                ],
            }
            try:
                raw = await self._llm.complete_json(
                    system=CONTRADICTION_SYSTEM,
                    user=f"Sub-question: {payload['sub_question']}\n\nPapers and claims:\n{payload['papers']}",
                )
            except LLMError as exc:
                logger.warning(
                    "contradiction_detection_failed sub_question=%s error=%s",
                    agent_result.sub_question.id,
                    exc,
                )
                continue

            if not isinstance(raw, list):
                continue

            valid_keys = {f.paper.paper_key for f in usable}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if item.get("paper_a_key") not in valid_keys or item.get("paper_b_key") not in valid_keys:
                    continue
                all_contradictions.append(
                    Contradiction(
                        topic=agent_result.sub_question.text,
                        paper_a_key=item["paper_a_key"],
                        paper_a_claim=item.get("paper_a_claim", ""),
                        paper_b_key=item["paper_b_key"],
                        paper_b_claim=item.get("paper_b_claim", ""),
                        explanation=item.get("explanation", ""),
                    )
                )

        logger.info("contradiction_detection_complete found=%s", len(all_contradictions))
        return all_contradictions
