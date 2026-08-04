import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.config import Settings
from app.core.pipeline import ReviewPipeline
from app.core.evaluation import EvaluationAgent
from app.core.orchestrator import Orchestrator
from app.core.research_agent import ResearchAgent
from app.core.contradiction_detector import ContradictionDetector
from app.core.synthesis import SynthesisAgent
from app.models.schemas import (
    ReviewJob,
    ReviewRequest,
    SubQuestion,
    ResearchCluster,
    ResearchAgentResult,
    LiteratureReview,
    EvaluationResult,
)


@pytest.mark.asyncio
async def test_pipeline_runs_evaluation_feedback_loop():
    settings = Settings(
        anthropic_api_key="test",
        max_feedback_loop_iterations=2,
    )

    # 1. Setup mock components
    mock_orchestrator = MagicMock(spec=Orchestrator)
    
    # Mock initial decomposition
    initial_cluster = ResearchCluster(
        theme="Theme A",
        sub_questions=[SubQuestion(text="Q1"), SubQuestion(text="Q2")]
    )
    mock_orchestrator.decompose = AsyncMock(return_value=[initial_cluster])

    # Mock follow-up decomposition
    follow_up_cluster = ResearchCluster(
        theme="Theme B",
        sub_questions=[SubQuestion(text="Q3")]
    )
    mock_orchestrator.decompose_follow_up = AsyncMock(return_value=[follow_up_cluster])

    # Mock research agent factory
    mock_research_agent = MagicMock(spec=ResearchAgent)
    
    async def fake_run(cluster, seen_keys):
        # Return a list of results matching cluster subquestions
        return [
            ResearchAgentResult(sub_question=sq, findings=[], papers_searched=1, papers_used=1)
            for sq in cluster.sub_questions
        ]
    mock_research_agent.run = fake_run
    mock_research_agent_factory = lambda: mock_research_agent

    # Mock contradiction detector
    mock_detector = MagicMock(spec=ContradictionDetector)
    mock_detector.detect = AsyncMock(return_value=[])

    # Mock synthesis agent
    mock_synthesis = MagicMock(spec=SynthesisAgent)
    mock_review = LiteratureReview(
        topic="test topic",
        background="bg",
        methodology_comparison="mc",
        key_findings="kf",
        open_questions="oq",
    )
    mock_synthesis.synthesize = AsyncMock(return_value=mock_review)

    # Mock evaluation agent: first fails, second passes
    mock_evaluation = MagicMock(spec=EvaluationAgent)
    mock_evaluation.evaluate = AsyncMock()
    mock_evaluation.evaluate.side_effect = [
        EvaluationResult(passed=False, feedback="needs Q3", follow_up_questions=["Q3"]),
        EvaluationResult(passed=True, feedback="complete", follow_up_questions=[])
    ]

    # 2. Instantiate pipeline
    pipeline = ReviewPipeline(
        settings=settings,
        orchestrator=mock_orchestrator,
        research_agent_factory=mock_research_agent_factory,
        contradiction_detector=mock_detector,
        synthesis_agent=mock_synthesis,
        evaluation_agent=mock_evaluation,
    )

    job = ReviewJob(request=ReviewRequest(topic="test topic"))

    # 3. Run pipeline
    await pipeline.run(job)

    # 4. Assertions
    assert job.status.value == "complete"
    # Should have 2 clusters initially, plus 1 follow-up = 2 clusters total (lists: initial_cluster, follow_up_cluster)
    assert len(job.clusters) == 2
    # Should have 3 total sub-questions (Q1, Q2, Q3)
    assert len(job.sub_questions) == 3
    # Sub-questions should match in order
    assert [sq.text for sq in job.sub_questions] == ["Q1", "Q2", "Q3"]
    # Verify mock calls
    assert mock_orchestrator.decompose.call_count == 1
    assert mock_orchestrator.decompose_follow_up.call_count == 1
    assert mock_evaluation.evaluate.call_count == 2
