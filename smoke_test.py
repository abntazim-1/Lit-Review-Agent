"""
Standalone smoke test: exercises the full pipeline (orchestrator -> N
parallel research agents -> contradiction detection -> synthesis) with
every external dependency (LLM, ArXiv, web search, PDF fetch) mocked out,
to prove the wiring is correct end to end without needing real API keys
or network access. Not part of the pytest suite (no network mocking
framework needed) -- run directly: `python smoke_test.py`.
"""
import asyncio
import os

os.environ["ANTHROPIC_API_KEY"] = "test"
os.environ["SQLITE_PATH"] = "./data/smoke_test.db"

from app.container import Container
from app.models.schemas import (
    ExtractedClaim,
    PaperMetadata,
    ReviewJob,
    ReviewRequest,
    SourceType,
)


async def main() -> None:
    container = Container()

    # --- mock the LLM: decomposition, extraction, contradiction, synthesis, evaluation ---
    call_log = []

    async def fake_complete_json(*, system, user, **kwargs):
        call_log.append(system[:40])
        if "decomposition" in system.lower() or "librarian" in system.lower():
            if "feedback" in user.lower() or "previous questions" in user.lower():
                return [
                    {
                        "theme": "Follow-up Theme",
                        "sub_questions": [
                            {"text": "non-English dataset evaluation of attention mechanisms", "rationale": "address feedback gaps"}
                        ]
                    }
                ]
            return [
                {
                    "theme": "Efficiency",
                    "sub_questions": [
                        {"text": "transformer attention mechanisms efficiency", "rationale": "core method"}
                    ]
                },
                {
                    "theme": "Retrieval",
                    "sub_questions": [
                        {"text": "long-context retrieval augmented generation", "rationale": "application area"}
                    ]
                }
            ]
        if "extracting structured findings" in system.lower():
            return {
                "methodology_summary": "Empirical benchmark study.",
                "claims": [{"claim": "Method X improves accuracy by 5%", "evidence": "Table 2", "confidence": 0.8}],
                "limitations": "Only evaluated on English text.",
            }
        if "cross-referencing" in system.lower():
            return []
        if "synthesis" in system.lower():
            return {
                "background": "This area studies efficient attention.",
                "methodology_comparison": "Both papers use benchmark evaluation.",
                "key_findings": "Method X consistently improves accuracy.",
                "open_questions": "Generalization beyond English remains untested.",
            }
        if "evaluation" in system.lower() or "editor" in system.lower():
            if "non-English" in user or "Follow-up" in user:
                return {
                    "passed": True,
                    "feedback": "Review is now complete.",
                    "follow_up_questions": []
                }
            return {
                "passed": False,
                "feedback": "The initial draft lacks non-English benchmark evaluation.",
                "follow_up_questions": ["non-English dataset evaluation of attention mechanisms"]
            }
        return {}

    container.llm.complete_json = fake_complete_json

    # --- mock ArXiv + web search so no network call happens ---
    async def fake_arxiv_search(query, max_results=None):
        return [
            PaperMetadata(
                paper_key=f"arxiv:fake-{query[:10]}",
                title=f"A study on {query}",
                source=SourceType.ARXIV,
                url="http://arxiv.org/abs/fake",
                pdf_url=None,  # forces graceful fallback to abstract-only extraction
                abstract=f"This paper investigates {query} in depth.",
            )
        ]

    container.arxiv.search = fake_arxiv_search
    container.web_search.search = lambda query, max_results: asyncio.sleep(0, result=[])

    job = ReviewJob(request=ReviewRequest(topic="Efficient attention mechanisms in transformers"))
    await container.pipeline.run(job)

    assert job.status.value == "complete", f"Job failed: {job.error}"
    assert job.result is not None
    assert len(job.sub_questions) == 3
    assert len(job.result.references) >= 1
    print("SMOKE TEST PASSED")
    print(f"  sub_questions: {[sq.text for sq in job.sub_questions]}")
    print(f"  papers_used: {sum(len(r.findings) for r in job.agent_results)}")
    print(f"  background: {job.result.background[:80]}...")
    print(f"  references: {[p.paper_key for p in job.result.references]}")

    await container.aclose()


if __name__ == "__main__":
    asyncio.run(main())
