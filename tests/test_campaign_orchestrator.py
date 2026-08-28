from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ghostlab.campaign.analyze import CandidateEvaluation
from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import TechniqueCatalog, load_catalog
from ghostlab.campaign.controller import initial_stage, promote_stage
from ghostlab.campaign.models import (
    CampaignManifest,
    CampaignResources,
    CandidateSpec,
    ChampionComparison,
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


def test_control_only_anchor_uses_one_slot_without_halving_search_budget() -> None:
    champion_id = "configs/suites/champion_guarded.json"
    pure_id = "configs/suites/keyword_research.json"
    manifest = _manifest("question.candidate_eig.v1").model_copy(
        update={
            "baseline_presets": (champion_id, pure_id),
            "baseline_techniques_by_preset": {
                champion_id: ("retrieval.sparse", "prior.quality"),
                pure_id: ("retrieval.sparse", "prior.quality"),
            },
            "baseline_search_modes": {
                champion_id: "control_only",
                pure_id: "composable",
            },
            "candidate_limit": 10,
        }
    )
    _, stage = initial_stage(
        load_catalog(ROOT / "configs/techniques/catalog_v2.json"), manifest
    )
    champion_candidates = [
        item for item in stage.candidates if item.baseline_id == champion_id
    ]
    pure_candidates = [item for item in stage.candidates if item.baseline_id == pure_id]
    assert len(champion_candidates) == 1
    assert champion_candidates[0].generation == "control"
    assert len(pure_candidates) > 1
    assert len(stage.candidates) <= manifest.candidate_limit


def test_promotion_preserves_low_scoring_matched_control() -> None:
    catalog = load_catalog(ROOT / "configs/techniques/catalog_v2.json")
    manifest = _manifest("question.candidate_eig.v1", "state.raw_history")
    _, stage = initial_stage(catalog, manifest)
    candidates_by_hash = {item.canonical_hash(): item for item in stage.candidates}
    outcomes = {
        job.job_id: JobOutcome(
            job_id=job.job_id,
            state="complete",
            score=(
                0.10
                if candidates_by_hash[job.candidate_hash].generation == "control"
                else 0.80
            ),
            session_rewards=(0.10,)
            if candidates_by_hash[job.candidate_hash].generation == "control"
            else (0.80,),
        )
        for job in stage.jobs
    }
    promoted = promote_stage(
        catalog,
        stage,
        outcomes,
        next_fidelity="f1",
        candidate_limit=2,
        exploration_fraction=0.0,
        seed=11,
        outer_fold_count=5,
        seeds=(11,),
    )
    assert len(promoted.candidates) == 2
    assert any(item.generation == "control" for item in promoted.candidates)


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


def test_same_fold_champion_comparison_reports_metrics_and_never_auto_promotes(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, FakeEvaluatorFactory())
    candidate = CandidateEvaluation(
        candidate_id="candidate",
        complexity=1,
        score=0.82,
        session_rewards=(0.80, 0.82, 0.84, 0.83, 0.81, 0.82),
        scenario_scores={"buying": 0.83, "browsing": 0.81},
        hit_rate_at_10=0.90,
        mrr=0.61,
        mttc=3.1,
    )
    champion = CandidateEvaluation(
        candidate_id="frozen-champion",
        complexity=0,
        score=0.70,
        session_rewards=(0.68, 0.70, 0.72, 0.71, 0.69, 0.70),
        scenario_scores={"buying": 0.71, "browsing": 0.69},
        hit_rate_at_10=0.84,
        mrr=0.54,
        mttc=3.8,
    )

    comparison = campaign._champion_comparison(
        candidate, champion, fit_receipts_verified=True
    )

    assert comparison.candidate_metrics.technical_score == 0.82
    assert comparison.champion_metrics.technical_score == 0.70
    assert comparison.technical_score_delta == pytest.approx(0.12)
    assert comparison.hit_rate_at_10_delta == pytest.approx(0.06)
    assert comparison.mrr_delta == pytest.approx(0.07)
    assert comparison.mttc_delta == pytest.approx(-0.7)
    assert comparison.beats_champion_point_estimate is True
    assert comparison.paired_session_count == 6
    assert comparison.statistically_supported is True
    assert comparison.promotion_recommended is True
    assert comparison.automatic_promotion is False


def test_champion_comparison_rejects_inconsistent_decision_flags() -> None:
    valid = {
        "champion_candidate_id": "champion",
        "champion_baseline_id": "configs/suites/champion_guarded.json",
        "candidate_metrics": {"technical_score": 0.8},
        "champion_metrics": {"technical_score": 0.7},
        "technical_score_delta": 0.1,
        "paired_mean_delta": 0.1,
        "confidence_interval": [0.05, 0.15],
        "randomization_pvalue": 0.01,
        "paired_session_count": 5,
        "wins": 4,
        "ties": 0,
        "losses": 1,
        "beats_champion_point_estimate": True,
        "statistically_supported": True,
        "no_material_scenario_regression": True,
        "fit_receipts_verified": True,
        "promotion_recommended": True,
        "automatic_promotion": False,
    }
    ChampionComparison.model_validate(valid)
    valid["promotion_recommended"] = False
    with pytest.raises(ValueError, match="promotion recommendation"):
        ChampionComparison.model_validate(valid)


def test_campaign_keeps_champion_control_and_emits_same_fold_comparisons(
    tmp_path: Path,
) -> None:
    champion_id = "configs/suites/champion_guarded.json"
    pure_id = "configs/suites/keyword_research.json"
    manifest = _manifest("question.candidate_eig.v1").model_copy(
        update={
            "baseline_presets": (champion_id, pure_id),
            "baseline_techniques_by_preset": {
                champion_id: ("retrieval.sparse", "prior.quality"),
                pure_id: ("retrieval.sparse", "prior.quality"),
            },
            "baseline_search_modes": {
                champion_id: "control_only",
                pure_id: "composable",
            },
        }
    )
    campaign = AutonomousCampaign(
        manifest=manifest,
        catalog=load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        registry=default_binding_registry(),
        evaluator_factory=FakeEvaluatorFactory(),
        checkpoint_path=tmp_path / "checkpoint.json",
        evidence_path=tmp_path / "evidence.json",
        outer_folds=(("a",), ("b",), ("c",), ("d",), ("e",)),
        project_root=ROOT,
        options=CampaignOptions(
            f1_candidates=4,
            f2_candidates=4,
            hpo_trials_per_structure=0,
            higher_order_rounds=0,
            bootstrap_resamples=50,
        ),
    )

    report = campaign.run()

    champion_rows = [
        item
        for item in report["safety"]
        if item["candidate"]["baseline_id"] == champion_id
    ]
    challenger_rows = [
        item
        for item in report["safety"]
        if item["candidate"]["baseline_id"] == pure_id
        and item["candidate"]["generation"] != "control"
    ]
    assert len(champion_rows) == 1
    assert champion_rows[0]["classification"] == "anchor_control"
    assert challenger_rows
    assert all(item["champion_comparison"] is not None for item in challenger_rows)
    assert all(
        item["champion_comparison"]["automatic_promotion"] is False
        for item in challenger_rows
    )


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
