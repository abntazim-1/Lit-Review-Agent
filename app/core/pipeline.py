"""
Top-level pipeline that runs a single review job end to end and updates
its status as it progresses. This is the module the API layer calls;
everything else (orchestrator, research agents, detectors, synthesis)
is a component this wires together.
"""
from __future__ import annotations

import asyncio
import time

from app.config import Settings
from app.core.contradiction_detector import ContradictionDetector
from app.core.evaluation import EvaluationAgent
from app.core.orchestrator import Orchestrator
from app.core.research_agent import ResearchAgent
from app.core.synthesis import SynthesisAgent
from app.models.schemas import JobStatus, ReviewJob
from app.utils.logging_config import get_logger, with_context

logger = get_logger(__name__)


class ReviewPipeline:
    def __init__(
        self,
        settings: Settings,
        orchestrator: Orchestrator,
        research_agent_factory,  # Callable[[], ResearchAgent] -- see container.py
        contradiction_detector: ContradictionDetector,
        synthesis_agent: SynthesisAgent,
        evaluation_agent: EvaluationAgent,
    ) -> None:
        self._settings = settings
        self._orchestrator = orchestrator
        self._research_agent_factory = research_agent_factory
        self._contradiction_detector = contradiction_detector
        self._synthesis_agent = synthesis_agent
        self._evaluation_agent = evaluation_agent
        self._agent_semaphore = asyncio.Semaphore(settings.max_concurrent_research_agents)

    def _log_and_add_to_job(self, job: ReviewJob, message: str, level: str = "INFO") -> None:
        if level == "INFO":
            logger.info("[%s] %s", job.job_id, message)
        elif level == "WARNING":
            logger.warning("[%s] %s", job.job_id, message)
        elif level == "ERROR":
            logger.error("[%s] %s", job.job_id, message)
        job.logs.append(f"[{level}] {message}")

    async def run(self, job: ReviewJob) -> None:
        log = with_context(logger, job_id=job.job_id)
        started = time.monotonic()
        try:
            self._log_and_add_to_job(job, f"Pipeline initialized for topic: '{job.request.topic}'.")

            job.status = JobStatus.DECOMPOSING
            self._log_and_add_to_job(job, "Orchestrating decomposition. Querying local LLM (Phi-3) to partition the niche...")
            initial_clusters = await self._orchestrator.decompose(
                job.request.topic, override_max=self._settings.max_decomposition_clusters
            )
            job.clusters = initial_clusters
            job.sub_questions = [sq for c in initial_clusters for sq in c.sub_questions]

            themes_str = ", ".join(f"'{c.theme}'" for c in initial_clusters)
            self._log_and_add_to_job(
                job,
                f"Decomposition complete. Formed {len(job.clusters)} thematic clusters: {themes_str}. "
                f"Total {len(job.sub_questions)} sub-questions created."
            )

            seen_paper_keys: set[str] = set()
            iteration = 0
            max_iterations = self._settings.max_feedback_loop_iterations

            active_clusters = initial_clusters

            while iteration < max_iterations:
                self._log_and_add_to_job(job, f"Starting iteration loop Round {iteration + 1}...")

                # 1. Research phase for active clusters
                job.status = JobStatus.RESEARCHING
                new_results = await self._run_research_agents_for_clusters(job, active_clusters, seen_paper_keys)
                job.agent_results.extend(new_results)
                total_papers = sum(len(r.findings) for r in job.agent_results)
                self._log_and_add_to_job(job, f"Round {iteration + 1} research phase complete. Total papers analyzed so far: {total_papers}.")

                # 2. Contradiction Detection
                job.status = JobStatus.DETECTING_CONTRADICTIONS
                self._log_and_add_to_job(job, "Running Contradiction Detector across claim extractions...")
                job.contradictions = await self._contradiction_detector.detect(job.agent_results)
                self._log_and_add_to_job(job, f"Contradiction detection complete. Found {len(job.contradictions)} conflicting claims.")

                # 3. Synthesis
                job.status = JobStatus.SYNTHESIZING
                self._log_and_add_to_job(job, "Synthesizing literature review sections (Background, Methodology Comparison, Findings)...")
                job.result = await self._synthesis_agent.synthesize(
                    topic=job.request.topic,
                    sub_questions=job.sub_questions,
                    agent_results=job.agent_results,
                    contradictions=job.contradictions,
                )
                self._log_and_add_to_job(job, "Draft synthesis complete.")

                # 4. Evaluation
                job.status = JobStatus.EVALUATING
                self._log_and_add_to_job(job, "Submitting draft to Academic Critique agent for evaluation...")
                eval_res = await self._evaluation_agent.evaluate(job.request.topic, job.result)
                log.info("evaluation_complete number=%s passed=%s feedback=%s", iteration, eval_res.passed, eval_res.feedback)

                if eval_res.passed:
                    self._log_and_add_to_job(job, "Academic evaluation PASSED! Review successfully finalized.")
                else:
                    self._log_and_add_to_job(
                        job,
                        f"Academic evaluation REJECTED. Feedback: '{eval_res.feedback}'",
                        "WARNING"
                    )

                if eval_res.passed or (iteration + 1 >= max_iterations):
                    break

                # Prepare for follow-up iteration
                previous_questions = [sq.text for sq in job.sub_questions]
                self._log_and_add_to_job(job, "Decomposing evaluation feedback into follow-up research questions...")
                follow_up_clusters = await self._orchestrator.decompose_follow_up(
                    topic=job.request.topic,
                    feedback=eval_res.feedback,
                    previous_questions=previous_questions
                )

                if not follow_up_clusters:
                    self._log_and_add_to_job(job, "No follow-up questions generated. Completing loop early.")
                    break

                # Append new clusters and sub-questions
                job.clusters.extend(follow_up_clusters)
                new_sqs = [sq for c in follow_up_clusters for sq in c.sub_questions]
                job.sub_questions.extend(new_sqs)

                follow_up_themes = ", ".join(f"'{c.theme}'" for c in follow_up_clusters)
                self._log_and_add_to_job(
                    job,
                    f"Generated {len(follow_up_clusters)} new clusters: {follow_up_themes}. "
                    f"Added {len(new_sqs)} follow-up sub-questions."
                )

                active_clusters = follow_up_clusters
                iteration += 1

            job.status = JobStatus.COMPLETE
            duration = round((time.monotonic() - started) * 1000)
            self._log_and_add_to_job(job, f"Pipeline complete. Total duration: {duration}ms.")
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            self._log_and_add_to_job(job, f"Pipeline execution failed: {exc}", "ERROR")
            log.error("job_failed error=%s", exc, exc_info=True)

    async def _run_research_agents_for_clusters(self, job: ReviewJob, clusters, seen_paper_keys):
        async def _run_one(cluster):
            async with self._agent_semaphore:
                self._log_and_add_to_job(job, f"Launching research agent for theme: '{cluster.theme}'...")
                agent = self._research_agent_factory()
                results = await agent.run(cluster, seen_paper_keys)
                for res in results:
                    self._log_and_add_to_job(
                        job,
                        f"Research complete for: '{res.sub_question.text}'. "
                        f"Found {res.papers_searched} candidate papers, selected {len(res.findings)} papers for extraction."
                    )
                    if res.errors:
                        for err in res.errors:
                            self._log_and_add_to_job(job, f"Paper processing error: {err}", "WARNING")
                return results

        results_list = await asyncio.gather(*(_run_one(c) for c in clusters))
        flat_results = []
        for res in results_list:
            flat_results.extend(res)
        return flat_results
