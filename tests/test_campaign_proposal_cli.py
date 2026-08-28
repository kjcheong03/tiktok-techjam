from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ghostlab.campaign.models import (
    CampaignManifest,
    CampaignResources,
    CandidateSpec,
    FidelityBudget,
    JobOutcome,
)
from ghostlab.campaign.proposal_from_campaign import (
    _job_id,
    materialize_confirmed_campaign_top_three,
)
from ghostlab.campaign.runner import CampaignCheckpoint
from ghostlab.research.technique_suite import UnifiedTechniqueConfig
from ghostlab.retrieval.residual import TECHNIQUE_ID as RESIDUAL_TECHNIQUE_ID
from ghostlab.training.protocol import FitReceipt


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_hash(values: set[str]) -> str:
    encoded = json.dumps(
        sorted(values), separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixture(root: Path, *, include_residual: bool = False) -> dict[str, str]:
    baseline_id = "configs/suites/baseline.json"
    baseline = UnifiedTechniqueConfig(
        experiment_id="baseline",
        state_variant="current",
        question_variant="fixed",
        question_order=(),
        retrieval_route="keyword",
        dense_backend="off",
        quality_prior_weight=0.2,
    )
    _write(root / baseline_id, baseline.model_dump(mode="json"))
    technique_ids = (
        "retrieval.sparse",
        "prior.quality",
        "state.raw_history",
        "question.adaptive_heuristic",
        RESIDUAL_TECHNIQUE_ID if include_residual else "filter.structured",
    )
    _write(
        root / "configs/techniques/catalog.json",
        {
            "schema_version": 2,
            "techniques": [
                {
                    "id": item,
                    "family": item.split(".", 1)[0],
                    "availability": "available",
                    "execution_class": "core",
                    "selection_safe": True,
                    "fit_required": item == RESIDUAL_TECHNIQUE_ID,
                }
                for item in technique_ids
            ],
        },
    )
    adaptive = {
        "dataset_sha256": "a" * 64,
        "sample_ids": ["s0", "s1", "s2"],
    }
    nested = {
        "dataset_sha256": "a" * 64,
        "adaptive_sample_ids": ["s0", "s1", "s2"],
        "outer_folds": [["s0"], ["s1"], ["s2"]],
    }
    _write(root / "configs/splits/adaptive.json", adaptive)
    _write(root / "configs/splits/nested.json", nested)
    catalog_path = root / "configs/techniques/catalog.json"
    adaptive_path = root / "configs/splits/adaptive.json"
    nested_path = root / "configs/splits/nested.json"
    manifest = CampaignManifest(
        campaign_id="confirmed-campaign",
        parent_commit="0123456789abcdef",
        catalog_hash=_hash(catalog_path),
        dataset_hash="a" * 64,
        adaptive_split_hash=_hash(adaptive_path),
        nested_split_hash=_hash(nested_path),
        baseline_presets=(baseline_id,),
        baseline_techniques=("retrieval.sparse", "prior.quality"),
        baseline_techniques_by_preset={
            baseline_id: ("retrieval.sparse", "prior.quality")
        },
        technique_ids=technique_ids[2:],
        max_order=1,
        candidate_limit=10,
        fidelity_budgets=FidelityBudget(f0=1, f1=2, f2=3),
        seeds=(11,),
        max_wall_seconds=60,
        resources=CampaignResources(cpu_jobs=1),
    )
    manifest_path = root / "artifacts/campaign/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    candidates = (
        CandidateSpec(
            candidate_id="candidate-state",
            baseline_id=baseline_id,
            techniques=(
                "retrieval.sparse",
                "prior.quality",
                "state.raw_history",
            ),
            complexity=1,
            generation="single",
        ),
        CandidateSpec(
            candidate_id="candidate-query",
            baseline_id=baseline_id,
            techniques=(
                "retrieval.sparse",
                "prior.quality",
                "question.adaptive_heuristic",
            ),
            complexity=1,
            generation="single",
        ),
        CandidateSpec(
            candidate_id="candidate-filter",
            baseline_id=baseline_id,
            techniques=(
                "retrieval.sparse",
                "prior.quality",
                technique_ids[-1],
            ),
            complexity=1,
            generation="single",
            parameters=(
                ("residual_rerank_depth", 5),
                ("residual_model_weight", 0.75),
            )
            if include_residual
            else (),
        ),
    )
    scores = (0.64, 0.63, 0.625)
    safety: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    outcomes: dict[str, JobOutcome] = {}
    for candidate, score in zip(candidates, scores, strict=True):
        mean_delta = score - 0.60
        confirmation_job_ids = [_job_id(candidate, fold, 11) for fold in (1, 2)]
        safety.append(
            {
                "candidate": candidate.model_dump(mode="json"),
                "candidate_hash": candidate.canonical_hash(),
                "confirmation_job_ids": confirmation_job_ids,
                "evaluation": {
                    "candidate_id": candidate.candidate_id,
                    "complexity": candidate.complexity,
                    "score": score,
                    "session_rewards": [score, score, score],
                    "scenario_scores": {"buying": score, "browsing": score},
                    "latency_p95_ms": 10.0 + candidate.complexity,
                    "memory_mb": 20.0,
                },
                "analysis": {
                    "candidate_id": candidate.candidate_id,
                    "baseline_id": "control-baseline",
                    "mean_delta": mean_delta,
                    "confidence_interval": [mean_delta - 0.01, mean_delta + 0.01],
                    "randomization_pvalue": 0.05,
                    "wins": 2,
                    "ties": 0,
                    "losses": 1,
                    "scenario_deltas": {"buying": 0.01, "browsing": 0.01},
                },
                "classification": "proposal_eligible",
                "reason": "prospective disjoint confirmation passed",
            }
        )
        summaries.append(
            {
                "candidate_id": candidate.candidate_id,
                "baseline_id": baseline_id,
                "score": score,
                "mean_delta": mean_delta,
                "classification": "package_eligible_proposal_only",
                "candidate_hash": candidate.canonical_hash(),
                "confirmation_job_ids": confirmation_job_ids,
            }
        )
        for fold in (1, 2):
            job_id = _job_id(candidate, fold, 11)
            receipts: tuple[FitReceipt, ...] = ()
            if include_residual and RESIDUAL_TECHNIQUE_ID in candidate.techniques:
                asset = root / f"artifacts/models/residual-fold-{fold}.joblib"
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(f"fold-{fold}".encode())
                receipts = (
                    FitReceipt(
                        technique_id=RESIDUAL_TECHNIQUE_ID,
                        outer_fold=fold,
                        inner_fold=0,
                        seed=11,
                        train_sample_ids_sha256=_ids_hash({"s0"}),
                        validation_sample_ids_sha256=_ids_hash({f"s{fold}"}),
                        asset_path=asset.relative_to(root).as_posix(),
                        asset_sha256=_hash(asset),
                    ),
                )
            outcomes[job_id] = JobOutcome(
                job_id=job_id,
                state="complete",
                score=score,
                session_rewards=(score,),
                scenario_scores={"buying": score},
                fit_receipts=receipts,
            )
    checkpoint_path = root / "artifacts/campaign/checkpoint.json"
    checkpoint_path.write_text(
        CampaignCheckpoint(
            manifest_hash=manifest.canonical_hash(), outcomes=outcomes
        ).model_dump_json(indent=2)
        + "\n"
    )
    candidate_ids = [item.candidate_id for item in candidates]
    evidence = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "manifest_hash": manifest.canonical_hash(),
        "parent_commit": manifest.parent_commit,
        "protected_holdout_access": "forbidden",
        "highest_fidelity": "f2",
        "selection_evidence_class": "prospective_disjoint_confirmation",
        "confirmation_status": "independent_development_confirmation",
        "split_evidence": {
            "search_outer_folds": [0],
            "confirmation_outer_folds": [1, 2],
            "search_sample_count": 1,
            "confirmation_sample_count": 2,
            "search_sample_ids_sha256": _ids_hash({"s0"}),
            "confirmation_sample_ids_sha256": _ids_hash({"s1", "s2"}),
            "overlap_count": 0,
            "f2_seeds": [11],
        },
        "independent_confirmation": {
            "status": "confirmed",
            "method": "prospective_disjoint_development_confirmation",
            "manifest_hash": manifest.canonical_hash(),
            "candidate_ids": candidate_ids,
        },
        "confirmed_top3": summaries,
        "safety": safety,
    }
    evidence_path = root / "artifacts/reports/evidence.json"
    _write(evidence_path, evidence)
    return {
        "manifest_path": str(manifest_path.relative_to(root)),
        "catalog_path": str(catalog_path.relative_to(root)),
        "evidence_path": str(evidence_path.relative_to(root)),
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "adaptive_split_path": str(adaptive_path.relative_to(root)),
        "nested_split_path": str(nested_path.relative_to(root)),
        "baseline_id": baseline_id,
        "output_dir": "artifacts/proposals/confirmed",
    }


def test_materializes_only_independently_confirmed_campaign_candidates(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    bundle = materialize_confirmed_campaign_top_three(
        project_root=tmp_path, **arguments
    )
    manifest = json.loads(bundle.manifest_path.read_text())
    assert len(bundle.preset_paths) == 3
    assert manifest["automatic_promotion"] is False
    assert manifest["f3_access"] == "forbidden"
    assert all(item["confirmed"] and item["safe"] for item in manifest["candidates"])
    assert all(
        "development-confirmed" in item["notes"][0]
        for item in manifest["candidates"]
    )


def test_rejects_adaptive_overlap_evidence(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    evidence_path = tmp_path / arguments["evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["confirmation_status"] = "withheld_selection_overlap"
    _write(evidence_path, evidence)
    with pytest.raises(ValueError, match="independent development confirmation"):
        materialize_confirmed_campaign_top_three(project_root=tmp_path, **arguments)


def test_rejects_tampered_confirmation_sample_hash(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    evidence_path = tmp_path / arguments["evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["split_evidence"]["confirmation_sample_ids_sha256"] = "0" * 64
    _write(evidence_path, evidence)
    with pytest.raises(ValueError, match="confirmation sample hash"):
        materialize_confirmed_campaign_top_three(project_root=tmp_path, **arguments)


def test_rejects_zero_delta_as_no_improvement(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    evidence_path = tmp_path / arguments["evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    candidate_id = evidence["confirmed_top3"][0]["candidate_id"]
    evidence["confirmed_top3"][0]["mean_delta"] = 0.0
    safety = next(
        item
        for item in evidence["safety"]
        if item["candidate"]["candidate_id"] == candidate_id
    )
    safety["analysis"]["mean_delta"] = 0.0
    safety["analysis"]["confidence_interval"] = [-0.01, 0.01]
    _write(evidence_path, evidence)
    with pytest.raises(ValueError, match="fewer than three"):
        materialize_confirmed_campaign_top_three(project_root=tmp_path, **arguments)


def test_rejects_missing_confirmed_checkpoint_job(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    checkpoint_path = tmp_path / arguments["checkpoint_path"]
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["outcomes"].pop(next(iter(checkpoint["outcomes"])))
    _write(checkpoint_path, checkpoint)
    with pytest.raises(ValueError, match="confirmation checkpoint job"):
        materialize_confirmed_campaign_top_three(project_root=tmp_path, **arguments)


def test_packages_fold_fitted_residual_asset_from_verified_receipt(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path, include_residual=True)
    bundle = materialize_confirmed_campaign_top_three(
        project_root=tmp_path, **arguments
    )
    presets = [json.loads(path.read_text()) for path in bundle.preset_paths]
    residual = next(item for item in presets if item["residual_reranker_enabled"])
    assert residual["residual_model_asset"].startswith(
        "artifacts/models/residual-fold-"
    )
    assert residual["residual_rerank_depth"] == 5
    assert residual["residual_model_weight"] == 0.75


def test_rejects_tampered_fold_fitted_residual_asset(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path, include_residual=True)
    checkpoint = json.loads((tmp_path / arguments["checkpoint_path"]).read_text())
    residual_outcome = next(
        item for item in checkpoint["outcomes"].values() if item["fit_receipts"]
    )
    asset = tmp_path / residual_outcome["fit_receipts"][0]["asset_path"]
    asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset hash"):
        materialize_confirmed_campaign_top_three(project_root=tmp_path, **arguments)
