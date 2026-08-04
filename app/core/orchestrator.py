"""
Orchestrator: the entry point of the pipeline. Its only job is topic ->
sub-questions. Keeping decomposition as its own stage (rather than
folding it into the research agents) means we can log and evaluate the
quality of decomposition independently of downstream search quality.
"""
from __future__ import annotations

from app.core.prompts import DECOMPOSITION_SYSTEM, FOLLOW_UP_DECOMPOSITION_SYSTEM
from app.models.schemas import ResearchCluster, SubQuestion
from app.services.llm_client import LLMClient
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, llm: LLMClient, max_sub_questions: int) -> None:
        self._llm = llm
        self._max_sub_questions = max_sub_questions

    async def decompose(self, topic: str, override_max: int | None = None) -> list[ResearchCluster]:
        limit = override_max or self._max_sub_questions
        user_prompt = (
            f'Topic: "{topic}"\n\n'
            f"Produce at most {limit} thematic clusters. Each cluster must have 3-5 sub-questions."
        )
        raw = await self._llm.complete_json(system=DECOMPOSITION_SYSTEM, user=user_prompt)
        
        clusters = self._parse_clusters(raw)

        if not clusters:
            # Fallback: direct search
            clusters = [
                ResearchCluster(
                    theme="General Overview",
                    sub_questions=[SubQuestion(text=topic, rationale="Fallback: direct topic search")]
                )
            ]

        # Enforce the limits on number of clusters
        clusters = clusters[:limit]
        logger.info("decomposed topic into %s clusters", len(clusters))
        return clusters

    async def decompose_follow_up(
        self, topic: str, feedback: str, previous_questions: list[str]
    ) -> list[ResearchCluster]:
        user_prompt = (
            f'Topic: "{topic}"\n\n'
            f"Previous questions investigated:\n"
            + "\n".join(f"- {q}" for q in previous_questions)
            + f"\n\nFeedback from review:\n{feedback}"
        )

        system_prompt = FOLLOW_UP_DECOMPOSITION_SYSTEM.format(
            topic=topic,
            previous_questions=str(previous_questions),
            feedback=feedback,
        )

        raw = await self._llm.complete_json(system=system_prompt, user=user_prompt)
        clusters = self._parse_clusters(raw)
        logger.info("decomposed follow-up into %s clusters", len(clusters))
        return clusters

    def _parse_clusters(self, raw: Any) -> list[ResearchCluster]:
        if isinstance(raw, dict):
            # Case 1: Wrapped list, e.g. {"clusters": [...]} or {"themes": [...]}
            for val in raw.values():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    raw = val
                    break
            else:
                # Case 2: Dict of theme keys to lists, e.g. {"Theme Name": [{"text": "...", "rationale": "..."}]}
                converted = []
                for theme, questions in raw.items():
                    if isinstance(questions, list):
                        sub_qs = []
                        for q in questions:
                            if isinstance(q, dict) and q.get("text"):
                                sub_qs.append(SubQuestion(text=q["text"].strip(), rationale=q.get("rationale", "").strip()))
                            elif isinstance(q, str):
                                sub_qs.append(SubQuestion(text=q.strip(), rationale=""))
                        if sub_qs:
                            converted.append(ResearchCluster(theme=theme.strip(), sub_questions=sub_qs))
                if converted:
                    return converted

                # Case 3: Single cluster, e.g. {"theme": "...", "sub_questions": [...]}
                if "theme" in raw or "sub_questions" in raw:
                    raw = [raw]

        if not isinstance(raw, list):
            logger.warning("Expected list or convertible dict from decomposition, got: %s", type(raw))
            return []

        clusters = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            # If the item is a dict but doesn't have "theme" and has other keys, it might be a single key-value theme mapping
            if "theme" not in item and "sub_questions" not in item:
                for theme, questions in item.items():
                    if isinstance(questions, list):
                        sub_questions = []
                        for sq in questions:
                            if isinstance(sq, dict) and sq.get("text"):
                                sub_questions.append(
                                    SubQuestion(text=sq["text"].strip(), rationale=sq.get("rationale", "").strip())
                                )
                            elif isinstance(sq, str):
                                sub_questions.append(SubQuestion(text=sq.strip()))
                        if sub_questions:
                            clusters.append(ResearchCluster(theme=theme.strip(), sub_questions=sub_questions))
                continue

            theme = item.get("theme", "General").strip()
            sub_questions = []
            for sq in item.get("sub_questions", []):
                if isinstance(sq, dict) and sq.get("text"):
                    sub_questions.append(
                        SubQuestion(text=sq["text"].strip(), rationale=sq.get("rationale", "").strip())
                    )
                elif isinstance(sq, str):
                    sub_questions.append(SubQuestion(text=sq.strip()))
            if sub_questions:
                clusters.append(ResearchCluster(theme=theme, sub_questions=sub_questions))

        return clusters
