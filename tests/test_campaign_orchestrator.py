from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import TechniqueCatalog, load_catalog
from ghostlab.campaign.models import (
    CampaignManifest,
    CampaignResources,
    CandidateSpec,
    FidelityBudget,
    JobOutcome,
)
from ghostlab.campaign.orchestrator import (
    AutonomousCampaign,
    CampaignOptions,
    FrozenInputs,
    verify_frozen_inputs,
)
from ghostlab.retrieval.residual import TECHNIQUE_ID as RESIDUAL_TECHNIQUE_ID
from ghostlab.training.protocol import FitReceipt

ROOT = Path(__file__).resolve().parents[1]


def _manifest(*techniques: str) -> CampaignManifest:
    return CampaignManifest(
        campaign_id="orchestrator-test",
        parent_commit="abcdef1",
        catalog_hash="0" * 64,
        dataset_hash="1" * 64,
        adaptive_split_hash="2" * 64,
        nested_split_hash="3" * 64,
        baseline_presets=("configs/suites/keyword_research.json",),
        baseline_techniques=("retrieval.sparse", "prior.quality"),
        baseline_techniques_by_preset={
            "configs/suites/keyword_research.json": (
                "retrieval.sparse",
                "prior.quality",
            )
        },
        technique_ids=techniques,
        max_order=2,
        candidate_limit=20,
        fidelity_budgets=FidelityBudget(f0=1, f1=1, f2=1),
        seeds=(11, 13),
        max_wall_seconds=60,
        resources=CampaignResources(cpu_jobs=2),
    )


class FakeEvaluatorFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, candidates: tuple[CandidateSpec, ...]):
        by_hash = {item.canonical_hash(): item for item in candidates}

        def evaluate(job):  # type: ignore[no-untyped-def]
            self.calls += 1
            candidate = by_hash[job.candidate_hash]
            parameter_gain = sum(
                float(value) * 0.00001
                for _, value in candidate.parameters
                if isinstance(value, (int, float))
            )
            score = 0.4 + candidate.complexity * 0.02 + parameter_gain
            return JobOutcome(
                job_id=job.job_id,
                state="complete",
                score=score,
                session_rewards=(score, score + 0.001),
                scenario_scores={"buying": score, "browsing": score},
            )

        return evaluate


class NonImprovingEvaluatorFactory(FakeEvaluatorFactory):
    def __call__(self, candidates: tuple[CandidateSpec, ...]):
        by_hash = {item.canonical_hash(): item for item in candidates}

        def evaluate(job):  # type: ignore[no-untyped-def]
            self.calls += 1
            candidate = by_hash[job.candidate_hash]
            score = 0.4 if candidate.generation == "control" else 0.39
            return JobOutcome(
                job_id=job.job_id,
                state="complete",
                score=score,
                session_rewards=(score, score),
                scenario_scores={"buying": score, "browsing": score},
            )

        return evaluate


class FlatImprovementEvaluatorFactory(FakeEvaluatorFactory):
    def __call__(self, candidates: tuple[CandidateSpec, ...]):
        by_hash = {item.canonical_hash(): item for item in candidates}

        def evaluate(job):  # type: ignore[no-untyped-def]
            self.calls += 1
            candidate = by_hash[job.candidate_hash]
            score = 0.4 if candidate.generation == "control" else 0.5
            return JobOutcome(
                job_id=job.job_id,
                state="complete",
                score=score,
                session_rewards=(score, score),
                scenario_scores={"buying": score, "browsing": score},
            )

        return evaluate


class FoldFitEvaluatorFactory(FakeEvaluatorFactory):
    def __call__(self, candidates: tuple[CandidateSpec, ...]):
        by_hash = {item.canonical_hash(): item for item in candidates}

        def evaluate(job):  # type: ignore[no-untyped-def]
            self.calls += 1
            candidate = by_hash[job.candidate_hash]
            score = 0.4 if candidate.generation == "control" else 0.5
            receipts = (
                (
                    FitReceipt(
                        technique_id=RESIDUAL_TECHNIQUE_ID,
                        outer_fold=job.outer_fold or 0,
                        inner_fold=0,
                        seed=job.seed,
                        train_sample_ids_sha256="1" * 64,
                        validation_sample_ids_sha256="2" * 64,
                        asset_path="artifacts/test-residual.joblib",
                        asset_sha256="3" * 64,
                    ),
                )
                if RESIDUAL_TECHNIQUE_ID in candidate.techniques
                else ()
            )
            return JobOutcome(
                job_id=job.job_id,
                state="complete",
                score=score,
                session_rewards=(score, score + 0.001),
                scenario_scores={"buying": score, "browsing": score},
                fit_receipts=receipts,
            )

        return evaluate


def _campaign(
    tmp_path: Path,
    factory: FakeEvaluatorFactory,
    *,
    techniques: tuple[str, ...] = ("question.candidate_eig.v1",),
    fit_capable_techniques: frozenset[str] = frozenset(),
) -> AutonomousCampaign:
    return AutonomousCampaign(
        manifest=_manifest(*techniques),
        catalog=load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        registry=default_binding_registry(),
        evaluator_factory=factory,
        checkpoint_path=tmp_path / "checkpoint.json",
        evidence_path=tmp_path / "evidence.json",
        outer_folds=(("a",), ("b",), ("c",), ("d",), ("e",)),
        project_root=ROOT,
        fit_capable_techniques=fit_capable_techniques,
        options=CampaignOptions(
            f1_candidates=2,
            f2_candidates=4,
            hpo_trials_per_structure=3,
            higher_order_rounds=0,
            bootstrap_resamples=50,
        ),
    )


def test_campaign_runs_f0_f1_f2_resumes_and_fails_closed(tmp_path: Path) -> None:
    factory = FakeEvaluatorFactory()
    campaign = _campaign(tmp_path, factory)
    first = campaign.run()
    first_calls = factory.calls
    second = campaign.run()

    assert first["highest_fidelity"] == "f2"
    assert first["confirmation_status"] == "independent_development_confirmation"
    assert first["selection_evidence_class"] == "prospective_disjoint_confirmation"
    assert first["proposal"] is not None
    assert first["confirmed_top3"]
    assert first["split_evidence"]["search_sample_count"] == 3
    assert first["split_evidence"]["confirmation_sample_count"] == 2
    assert first["split_evidence"]["overlap_count"] == 0
    assert first["split_evidence"]["f2_seeds"] == (11,)
    assert first["conditional_hpo"]["classification"] == (
        "adaptive_conditional_bohb_f1_racing"
    )
    hpo_rungs = first["conditional_hpo"]["diagnostics"]
    assert hpo_rungs
    assert any(row.get("pruned", 0) > 0 for row in hpo_rungs)
    assert {row["seed_budget"] for row in hpo_rungs if "seed_budget" in row} >= {
        1,
        2,
    }
    first_batch = next(row for row in hpo_rungs if row.get("adaptive_batch") == 0)
    second_batch = next(row for row in hpo_rungs if row.get("adaptive_batch") == 1)
    assert second_batch["prior_observations"] > first_batch["prior_observations"]
    assert first["leaderboards"]["f2"]
    assert second["stage_counts"] == first["stage_counts"]
    assert factory.calls == first_calls
    persisted = json.loads((tmp_path / "evidence.json").read_text())
    assert persisted["manifest_hash"] == second["manifest_hash"]
    assert persisted["stage_counts"] == second["stage_counts"]


def test_search_and_confirmation_jobs_use_disjoint_declared_folds(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, FakeEvaluatorFactory())
    candidate = CandidateSpec(
        candidate_id="control",
        baseline_id="configs/suites/keyword_research.json",
        techniques=("retrieval.sparse", "prior.quality"),
        complexity=0,
        generation="control",
    )
    f0 = campaign._stage("f0", (candidate,))
    f1 = campaign._stage("f1", (candidate,))
    f2 = campaign._stage("f2", (candidate,))
    # Search jobs deliberately carry no single fold: OfflineCampaignEvaluator
    # samples only from the union of the manifest's frozen search folds.
    assert {job.outer_fold for job in f0.jobs} == {None}
    assert {job.outer_fold for job in f1.jobs} == {None}
    assert {job.outer_fold for job in f2.jobs} == {1, 4}


def test_hpo_suggestions_jointly_bind_all_eligible_parameters(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, FakeEvaluatorFactory())
    root = CandidateSpec(
        candidate_id="eig",
        baseline_id="configs/suites/keyword_research.json",
        techniques=("retrieval.sparse", "prior.quality", "question.candidate_eig.v1"),
        complexity=1,
        generation="single",
    )
    parameters = campaign.search_space.for_techniques(root.techniques)
    assert {item.name for item in parameters} == {
        "eig_candidate_k",
        "question_value_margin",
    }
    margin = next(item for item in parameters if item.name == "question_value_margin")
    assert margin.low == 0.0


def test_non_improving_candidate_is_not_a_confirmed_proposal(tmp_path: Path) -> None:
    report = _campaign(tmp_path, NonImprovingEvaluatorFactory()).run()
    assert report["proposal"] is None
    assert report["confirmed_top3"] == []
    classifications = {item["classification"] for item in report["safety"]}
    assert "no_confirmed_improvement" in classifications


def test_behaviorally_identical_hpo_variants_do_not_fill_top_three(
    tmp_path: Path,
) -> None:
    report = _campaign(tmp_path, FlatImprovementEvaluatorFactory()).run()
    assert report["confirmed_top3"] == []


def test_fit_required_candidate_is_not_allowed_into_f2(tmp_path: Path) -> None:
    campaign = _campaign(
        tmp_path,
        FakeEvaluatorFactory(),
        techniques=("ranking.reward_lambdamart.v1",),
    )
    report = campaign.run()
    f2 = report["leaderboards"]["f2"]
    assert f2
    assert all(
        "ranking.reward_lambdamart.v1" not in row["candidate"]["techniques"]
        for row in f2
    )
    assert report["proposal"] is None


def test_fold_fitted_residual_can_reach_f2_with_complete_receipts(
    tmp_path: Path,
) -> None:
    campaign = _campaign(
        tmp_path,
        FoldFitEvaluatorFactory(),
        techniques=(RESIDUAL_TECHNIQUE_ID,),
        fit_capable_techniques=frozenset({RESIDUAL_TECHNIQUE_ID}),
    )
    report = campaign.run()

    residual_rows = [
        row
        for row in report["leaderboards"]["f2"]
        if RESIDUAL_TECHNIQUE_ID in row["candidate"]["techniques"]
    ]
    assert residual_rows
    classifications = {
        row["classification"]
        for row in report["safety"]
        if RESIDUAL_TECHNIQUE_ID in row["candidate"]["techniques"]
    }
    assert "fit_evidence_incomplete" not in classifications
    assert "research_only_not_selection_safe" not in classifications


def test_default_frozen_fold_roles_resolve_to_exact_90_60_counts(
    tmp_path: Path,
) -> None:
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    folds = tuple(tuple(str(value) for value in fold) for fold in nested["outer_folds"])
    campaign = AutonomousCampaign(
        manifest=_manifest("question.candidate_eig.v1"),
        catalog=load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        registry=default_binding_registry(),
        evaluator_factory=FakeEvaluatorFactory(),
        checkpoint_path=tmp_path / "checkpoint.json",
        evidence_path=tmp_path / "evidence.json",
        outer_folds=folds,
        project_root=ROOT,
    )
    evidence = campaign._split_evidence()
    assert evidence["search_sample_count"] == 90
    assert evidence["confirmation_sample_count"] == 60
    assert evidence["overlap_count"] == 0
    assert (
        evidence["search_sample_ids_sha256"]
        != evidence["confirmation_sample_ids_sha256"]
    )


def test_verify_frozen_inputs_checks_hashes_and_split_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "development.jsonl"
    catalog_path = tmp_path / "catalog.json"
    adaptive = tmp_path / "adaptive.json"
    nested = tmp_path / "nested.json"
    dataset.write_text("development\n")
    catalog_path.write_text("catalog\n")
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    adaptive.write_text(
        json.dumps({"dataset_sha256": dataset_hash, "sample_ids": ["a", "b"]})
    )
    nested.write_text(
        json.dumps({"adaptive_sample_ids": ["a", "b"], "outer_folds": [["a"], ["b"]]})
    )
    catalog_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    manifest = _manifest("question.candidate_eig.v1").model_copy(
        update={
            "catalog_hash": catalog_hash,
            "dataset_hash": dataset_hash,
            "adaptive_split_hash": hashlib.sha256(adaptive.read_bytes()).hexdigest(),
            "nested_split_hash": hashlib.sha256(nested.read_bytes()).hexdigest(),
            "search_outer_folds": (0,),
            "confirmation_outer_folds": (1,),
        }
    )
    catalog = TechniqueCatalog(2, {}, catalog_hash)
    verified = verify_frozen_inputs(
        manifest,
        catalog,
        FrozenInputs(catalog_path, dataset, adaptive, nested),
    )
    assert verified["dataset"] == dataset_hash
