"""
Synthesis: the final stage. Takes every research agent's findings plus
the contradiction list and asks the LLM to write the narrative sections
of the review. References are assembled deterministically (not by the
LLM) so the citation list can never drift from what was actually found.
"""
from __future__ import annotations

import re

from app.core.prompts import SYNTHESIS_SYSTEM
from app.models.schemas import Contradiction, LiteratureReview, PaperMetadata, ResearchAgentResult, SubQuestion
from app.services.llm_client import LLMClient
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_MAX_FINDINGS_CHARS = 40_000  # keep the synthesis prompt bounded regardless of how many papers were processed


class SynthesisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(
        self,
        topic: str,
        sub_questions: list[SubQuestion],
        agent_results: list[ResearchAgentResult],
        contradictions: list[Contradiction],
    ) -> LiteratureReview:
        findings_blob = self._render_findings(agent_results)
        contradictions_blob = "\n".join(
            f"- On '{c.topic}': {c.paper_a_key} claims '{c.paper_a_claim}' while "
            f"{c.paper_b_key} claims '{c.paper_b_claim}' ({c.explanation})"
            for c in contradictions
        ) or "None detected."

        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Sub-questions investigated:\n"
            + "\n".join(f"- {sq.text}" for sq in sub_questions)
            + f"\n\nFindings by paper:\n{findings_blob}\n\n"
            f"Detected contradictions:\n{contradictions_blob}"
        )

        sections = await self._llm.complete_json(
            system=SYNTHESIS_SYSTEM, user=user_prompt, max_tokens=4096
        )

        references = self._collect_references(agent_results)

        return LiteratureReview(
            topic=topic,
            background=sections.get("background", ""),
            methodology_comparison=sections.get("methodology_comparison", ""),
            key_findings=sections.get("key_findings", ""),
            open_questions=sections.get("open_questions", ""),
            contradictions=contradictions,
            references=references,
            sub_questions=sub_questions,
        )

    def _render_findings(self, agent_results: list[ResearchAgentResult]) -> str:
        lines: list[str] = []
        for agent_result in agent_results:
            lines.append(f"\n## Sub-question: {agent_result.sub_question.text}")
            for finding in agent_result.findings:
                if finding.extraction_failed:
                    continue
                lines.append(f"### {finding.paper.title} ({finding.paper.paper_key})")
                lines.append(f"Methodology: {finding.methodology_summary}")
                for claim in finding.claims:
                    lines.append(f"- Claim: {claim.claim} (confidence {claim.confidence})")
                if finding.limitations:
                    lines.append(f"Limitations: {finding.limitations}")

        blob = "\n".join(lines)
        if len(blob) > _MAX_FINDINGS_CHARS:
            logger.warning("findings_blob_truncated original_chars=%s", len(blob))
            blob = blob[:_MAX_FINDINGS_CHARS] + "\n[...truncated for length...]"
        return blob

    def _collect_references(self, agent_results: list[ResearchAgentResult]) -> list[PaperMetadata]:
        seen: dict[str, PaperMetadata] = {}
        for agent_result in agent_results:
            for finding in agent_result.findings:
                if not finding.extraction_failed:
                    seen[finding.paper.paper_key] = finding.paper

        def sort_key(p: PaperMetadata):
            year = 0
            if p.published:
                match = re.search(r'\b(19\d\d|20\d\d)\b', p.published)
                if match:
                    year = int(match.group(1))
            return (-year, p.title)

        return sorted(seen.values(), key=sort_key)
