from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from ghostlab.campaign.analyze import (
    CandidateEvaluation,
    interaction_analysis,
    paired_analysis,
)
from ghostlab.campaign.cache import CacheKey, ContentAddressedCache
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.controller import CampaignStage, promote_stage
from ghostlab.campaign.jobs import build_jobs
from ghostlab.campaign.models import (
    CampaignJob,
    CampaignResources,
    CandidateSpec,
    JobOutcome,
    ResourceRequest,
    TechniqueSpec,
)
from ghostlab.campaign.planner import backward_ablations, plan_candidates
from ghostlab.campaign.proposal import propose_candidate
from ghostlab.campaign.runner import run_jobs
from ghostlab.campaign.scheduler import schedule_waves

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    "state.raw_history",
    "retrieval.sparse",
    "ranking.constraint_gbdt",
    "prior.quality",
    "guard.override_fallback",
)


class CatalogPlannerTests(unittest.TestCase):
    def test_technique_paths_cannot_escape_repository(self) -> None:
        with self.assertRaises(ValidationError):
            TechniqueSpec(
                id="bad",
                family="test",
                availability="available",
                source="../outside.py",
            )

    def test_v2_extends_v1_and_unavailable_techniques_are_not_executable(self) -> None:
        catalog = load_catalog(PROJECT_ROOT / "configs/techniques/catalog_v2.json")
        self.assertIn("retrieval.sparse", catalog.techniques)
        self.assertTrue(catalog.techniques["retrieval.sparse"].executable)
        self.assertTrue(catalog.techniques["question.candidate_eig.v1"].executable)
        self.assertFalse(catalog.techniques["retrieval.splade_rescue.v1"].executable)

        plan = plan_candidates(
            catalog,
            baseline_id="champion",
            baseline_techniques=BASELINE,
            technique_ids=(
                "question.candidate_eig.v1",
                "retrieval.splade_rescue.v1",
            ),
            max_order=2,
        )
        self.assertEqual(len(plan.candidates), 2)  # control plus available BOHB
        self.assertTrue(
            any(
                "unavailable technique" in reason
                for item in plan.skipped
                for reason in item.reasons
            )
        )

    def test_candidate_hash_is_order_independent_and_ablations_are_complete(
        self,
    ) -> None:
        left = CandidateSpec(
            candidate_id="left",
            baseline_id="base",
            techniques=("a", "b"),
            parameters=(("x", 1), ("y", 2)),
            complexity=2,
            generation="pair",
        )
        right = CandidateSpec(
            candidate_id="right",
            baseline_id="base",
            techniques=("b", "a"),
            parameters=(("y", 2), ("x", 1)),
            complexity=2,
            generation="pair",
        )
        self.assertEqual(left.canonical_hash(), right.canonical_hash())
        self.assertEqual(len(backward_ablations(left)), 2)

    def test_candidate_limit_bounds_higher_order_enumeration(self) -> None:
        catalog = load_catalog(PROJECT_ROOT / "configs/techniques/catalog_v2.json")
        available = tuple(
            item.id for item in catalog.available() if item.id not in set(BASELINE)
        )
        plan = plan_candidates(
            catalog,
            baseline_id="champion",
            baseline_techniques=BASELINE,
            technique_ids=available,
            max_order=8,
            candidate_limit=3,
        )
        self.assertEqual(len(plan.candidates), 3)


class SchedulingAndResumeTests(unittest.TestCase):
    @staticmethod
    def job(identifier: str, *, heavy: bool = False) -> CampaignJob:
        return CampaignJob(
            job_id=identifier,
            candidate_hash="a" * 64,
            fidelity="f0",
            seed=1,
            resources=ResourceRequest(cpu=1, memory_gb=2, heavy_model=heavy),
        )

    def test_scheduler_separates_heavy_jobs(self) -> None:
        waves = schedule_waves(
            (self.job("a", heavy=True), self.job("b", heavy=True), self.job("c")),
            CampaignResources(cpu_jobs=3, memory_gb=8, heavy_model_jobs=1),
        )
        self.assertEqual(len(waves), 2)
        self.assertTrue(
            all(
                sum(job.resources.heavy_model for job in wave.jobs) <= 1
                for wave in waves
            )
        )

    def test_f2_jobs_cover_every_outer_fold_and_seed(self) -> None:
        catalog = load_catalog(PROJECT_ROOT / "configs/techniques/catalog_v2.json")
        candidate = CandidateSpec(
            candidate_id="control",
            baseline_id="champion",
            techniques=BASELINE,
            complexity=0,
            generation="control",
        )
        jobs = build_jobs(
            catalog,
            (candidate,),
            fidelity="f2",
            outer_fold_count=5,
            seeds=(1, 2),
        )
        self.assertEqual(len(jobs), 10)
        self.assertEqual({job.outer_fold for job in jobs}, set(range(5)))

    def test_promotion_keeps_best_and_audit_candidate(self) -> None:
        catalog = load_catalog(PROJECT_ROOT / "configs/techniques/catalog_v2.json")
        candidates = tuple(
            CandidateSpec(
                candidate_id=f"candidate-{index}",
                baseline_id="base",
                techniques=BASELINE,
                parameters=(("variant", index),),
                complexity=index,
                generation="single",
            )
            for index in range(5)
        )
        jobs = build_jobs(
            catalog, candidates, fidelity="f0", outer_fold_count=5, seeds=(1,)
        )
        scores = {
            candidate.canonical_hash(): float(dict(candidate.parameters)["variant"])
            for candidate in candidates
        }
        outcomes = {
            job.job_id: JobOutcome(
                job_id=job.job_id,
                state="complete",
                score=scores[job.candidate_hash],
            )
            for job in jobs
        }
        promoted = promote_stage(
            catalog,
            CampaignStage("f0", candidates, jobs),
            outcomes,
            next_fidelity="f1",
            candidate_limit=3,
            exploration_fraction=0.34,
            seed=7,
            outer_fold_count=5,
            seeds=(1,),
        )
        promoted_ids = {item.candidate_id for item in promoted.candidates}
        self.assertEqual(len(promoted_ids), 3)
        self.assertIn("candidate-4", promoted_ids)
        self.assertIn("candidate-3", promoted_ids)

    def test_runner_resumes_completed_jobs(self) -> None:
        calls: list[str] = []

        def evaluate(job: CampaignJob) -> JobOutcome:
            calls.append(job.job_id)
            return JobOutcome(job_id=job.job_id, state="complete", score=0.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            jobs = (self.job("a"), self.job("b"))
            first = run_jobs(
                jobs,
                manifest_hash="manifest",
                resources=CampaignResources(cpu_jobs=2, memory_gb=8),
                checkpoint_path=path,
                evaluator=evaluate,
            )
            second = run_jobs(
                jobs,
                manifest_hash="manifest",
                resources=CampaignResources(cpu_jobs=2, memory_gb=8),
                checkpoint_path=path,
                evaluator=evaluate,
            )
            self.assertEqual(first, second)
            self.assertEqual(calls, ["a", "b"])

    def test_cache_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ContentAddressedCache(Path(directory))
            key = CacheKey("features", (("fold", "1"),))
            target = cache.write_bytes(key, b"valid")
            self.assertEqual(cache.read_bytes(key), b"valid")
            target.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                cache.read_bytes(key)


class AnalysisProposalTests(unittest.TestCase):
    @staticmethod
    def evaluation(
        identifier: str, rewards: tuple[float, ...], *, complexity: int = 1
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            identifier,
            complexity,
            sum(rewards) / len(rewards),
            rewards,
            {"buying": sum(rewards) / len(rewards)},
        )

    def test_paired_and_interaction_analysis(self) -> None:
        base = self.evaluation("base", (0.1, 0.2, 0.3), complexity=0)
        first = self.evaluation("first", (0.2, 0.3, 0.4))
        second = self.evaluation("second", (0.15, 0.25, 0.35))
        both = self.evaluation("both", (0.3, 0.4, 0.5), complexity=2)
        analysis = paired_analysis(first, base, resamples=100, seed=7)
        self.assertAlmostEqual(analysis.mean_delta, 0.1)
        aggregate, per_session = interaction_analysis(base, first, second, both)
        self.assertAlmostEqual(aggregate, 0.05)
        self.assertTrue(all(abs(value - 0.05) < 1e-9 for value in per_session))

    def test_proposal_uses_simplicity_inside_tie_band(self) -> None:
        base = self.evaluation("base", (0.5, 0.5, 0.5), complexity=0)
        simple = self.evaluation("simple", (0.51, 0.51, 0.51), complexity=1)
        complex_candidate = self.evaluation(
            "complex", (0.512, 0.512, 0.512), complexity=4
        )
        analyses = {
            "simple": paired_analysis(simple, base, resamples=100),
            "complex": paired_analysis(complex_candidate, base, resamples=100),
        }
        proposal = propose_candidate(
            (simple, complex_candidate), analyses, tie_band=0.005
        )
        self.assertEqual(proposal.candidate_id, "simple")


if __name__ == "__main__":
    unittest.main()
