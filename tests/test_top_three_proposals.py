from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from ghostlab.campaign.analyze import CandidateEvaluation, PairedAnalysis
from ghostlab.campaign.proposal_compare import compare_proposal_reports
from ghostlab.campaign.proposal_materializer import materialize_top_three
from ghostlab.campaign.top_three import CandidatePackage, select_top_three
from ghostlab.research.technique_suite import UnifiedTechniqueConfig


def config(identifier: str) -> UnifiedTechniqueConfig:
    return UnifiedTechniqueConfig(
        experiment_id=identifier,
        state_variant="raw_history",
        question_variant="sequence",
        question_order=("other",),
        retrieval_route="keyword",
        dense_backend="off",
    )


def evaluation(
    identifier: str,
    score: float,
    *,
    complexity: int,
    latency: float,
    memory: float,
) -> CandidateEvaluation:
    rewards = (score - 0.01, score, score + 0.01)
    return CandidateEvaluation(
        candidate_id=identifier,
        complexity=complexity,
        score=score,
        session_rewards=rewards,
        scenario_scores={"buying": score, "browsing": score},
        latency_p95_ms=latency,
        memory_mb=memory,
    )


def analysis(
    identifier: str,
    *,
    mean_delta: float,
    lower: float,
    scenario_delta: float = 0.0,
) -> PairedAnalysis:
    return PairedAnalysis(
        candidate_id=identifier,
        baseline_id="champion",
        mean_delta=mean_delta,
        confidence_interval=(lower, mean_delta + 0.02),
        randomization_pvalue=0.05,
        wins=2,
        ties=0,
        losses=1,
        scenario_deltas={"buying": scenario_delta, "browsing": scenario_delta},
    )


def package(
    root: Path,
    identifier: str,
    *,
    confirmed: bool = True,
    safe: bool = True,
    extra: str = "core",
) -> CandidatePackage:
    asset = root / "assets" / f"{identifier}.json"
    evidence = root / "evidence" / f"{identifier}.json"
    asset.parent.mkdir(exist_ok=True)
    evidence.parent.mkdir(exist_ok=True)
    asset.write_text(json.dumps({"candidate": identifier}), encoding="utf-8")
    evidence.write_text(json.dumps({"validated": True}), encoding="utf-8")
    return CandidatePackage(
        candidate_id=identifier,
        config=config(identifier),
        dependency_extras=(extra,),
        assets=(str(asset.relative_to(root)),),
        evidence_refs=(str(evidence.relative_to(root)),),
        confirmed=confirmed,
        safe=safe,
    )


def selection_fixture(root: Path):
    values = (
        evaluation("score", 0.66, complexity=5, latency=80, memory=400),
        evaluation("robust", 0.65, complexity=4, latency=60, memory=350),
        evaluation("efficient", 0.645, complexity=1, latency=5, memory=40),
        evaluation("unsafe_high", 0.70, complexity=2, latency=10, memory=50),
        evaluation("unconfirmed", 0.68, complexity=2, latency=10, memory=50),
    )
    analyses = {
        "score": analysis("score", mean_delta=0.03, lower=0.005),
        "robust": analysis("robust", mean_delta=0.025, lower=0.02),
        "efficient": analysis("efficient", mean_delta=0.015, lower=0.001),
        "unsafe_high": analysis("unsafe_high", mean_delta=0.07, lower=0.05),
        "unconfirmed": analysis("unconfirmed", mean_delta=0.05, lower=0.03),
    }
    packages = {
        "score": package(root, "score", extra="gbdt"),
        "robust": package(root, "robust"),
        "efficient": package(root, "efficient"),
        "unsafe_high": package(root, "unsafe_high", safe=False),
        "unconfirmed": package(root, "unconfirmed", confirmed=False),
    }
    return select_top_three(
        values,
        analyses,
        packages,
        baseline_id="champion",
        project_root=root,
    )


def test_selection_produces_three_distinct_roles_and_excludes_unready() -> None:
    with tempfile.TemporaryDirectory() as directory:
        selected = selection_fixture(Path(directory))
    assert selected.score_leader.evaluation.candidate_id == "score"
    assert selected.robust_leader.evaluation.candidate_id == "robust"
    assert selected.efficient_alternative.evaluation.candidate_id == "efficient"
    assert len({item.evaluation.candidate_id for item in selected.candidates}) == 3
    assert dict(selected.excluded) == {
        "unconfirmed": "candidate package is unconfirmed",
        "unsafe_high": "candidate package is marked unsafe",
    }


def test_selection_excludes_behaviorally_duplicate_candidates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values = (
            evaluation("a", 0.66, complexity=1, latency=10, memory=10),
            evaluation("a_duplicate", 0.66, complexity=2, latency=20, memory=20),
            evaluation("b", 0.65, complexity=1, latency=10, memory=10),
            evaluation("c", 0.64, complexity=1, latency=10, memory=10),
        )
        analyses = {
            item.candidate_id: analysis(
                item.candidate_id, mean_delta=0.01, lower=0.0
            )
            for item in values
        }
        packages = {
            item.candidate_id: package(root, item.candidate_id) for item in values
        }
        selected = select_top_three(
            values,
            analyses,
            packages,
            baseline_id="champion",
            project_root=root,
        )
    assert {item.evaluation.candidate_id for item in selected.candidates} == {
        "a",
        "b",
        "c",
    }
    assert "behaviorally duplicates a" in dict(selected.excluded)["a_duplicate"]


def test_selection_fails_closed_for_missing_evidence_and_scenario_regression() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values = tuple(
            evaluation(name, 0.6, complexity=1, latency=1, memory=1)
            for name in ("a", "b", "c")
        )
        analyses = {
            name: analysis(name, mean_delta=0.01, lower=0.0)
            for name in ("a", "b", "c")
        }
        analyses["a"] = analysis(
            "a", mean_delta=0.01, lower=0.0, scenario_delta=-0.03
        )
        packages = {name: package(root, name) for name in ("a", "b", "c")}
        Path(root / packages["b"].evidence_refs[0]).unlink()
        with pytest.raises(ValueError, match="fewer than three"):
            select_top_three(
                values,
                analyses,
                packages,
                baseline_id="champion",
                project_root=root,
            )


def test_materializer_is_idempotent_and_renders_complete_human_runbook() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "configs/suites/champion_guarded.json"
        split = root / "configs/splits/adaptive_v1.json"
        baseline.parent.mkdir(parents=True)
        split.parent.mkdir(parents=True)
        baseline.write_text(config("baseline").model_dump_json(), encoding="utf-8")
        split.write_text(json.dumps({"sample_ids": ["one"]}), encoding="utf-8")
        selected = selection_fixture(root)
        output = root / "artifacts/proposals/w2_top_three"
        first = materialize_top_three(
            selected,
            project_root=root,
            output_dir=output,
            baseline_config_path="configs/suites/champion_guarded.json",
            rollback_commit="0123456789abcdef",
        )
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*first.preset_paths, first.manifest_path, first.guide_path)
        }
        second = materialize_top_three(
            selected,
            project_root=root,
            output_dir=output,
            baseline_config_path="configs/suites/champion_guarded.json",
            rollback_commit="0123456789abcdef",
        )
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*second.preset_paths, second.manifest_path, second.guide_path)
        }
        guide = first.guide_path.read_text(encoding="utf-8")
        manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert before == after
    assert "uv run --extra gbdt python -m scripts.run_unified_preset" in guide
    assert "scripts.compare_proposal_reports" in guide
    assert "Gate A" in guide and "Gate B" in guide and "One-shot F3" in guide
    assert "Known-good commit: `0123456789abcdef`" in guide
    assert len(manifest["candidates"]) == 3
    assert manifest["automatic_promotion"] is False
    assert manifest["f3_access"] == "forbidden"
    assert all(item["preset"]["sha256"] for item in manifest["candidates"])
    assert all(item["assets"][0]["sha256"] for item in manifest["candidates"])


def test_materializer_refuses_to_overwrite_an_immutable_proposal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "configs/suites/champion_guarded.json"
        split = root / "configs/splits/adaptive_v1.json"
        baseline.parent.mkdir(parents=True)
        split.parent.mkdir(parents=True)
        baseline.write_text(config("baseline").model_dump_json(), encoding="utf-8")
        split.write_text(json.dumps({"sample_ids": ["one"]}), encoding="utf-8")
        selected = selection_fixture(root)
        output = root / "artifacts/proposals/w2_top_three"
        materialize_top_three(
            selected,
            project_root=root,
            output_dir=output,
            baseline_config_path="configs/suites/champion_guarded.json",
            rollback_commit="0123456789abcdef",
        )
        (output / "score_leader.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="immutable proposal"):
            materialize_top_three(
                selected,
                project_root=root,
                output_dir=output,
                baseline_config_path="configs/suites/champion_guarded.json",
                rollback_commit="0123456789abcdef",
            )


def test_materializer_rejects_protected_paths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "configs/suites/champion.json"
        split = root / "configs/splits/protected/f3.json"
        baseline.parent.mkdir(parents=True)
        split.parent.mkdir(parents=True)
        baseline.write_text(config("baseline").model_dump_json(), encoding="utf-8")
        split.write_text("{}", encoding="utf-8")
        selected = selection_fixture(root)
        with pytest.raises(ValueError, match="protected data"):
            materialize_top_three(
                selected,
                project_root=root,
                output_dir=root / "artifacts/proposals",
                baseline_config_path="configs/suites/champion.json",
                split_path="configs/splits/protected/f3.json",
                rollback_commit="0123456",
            )


def test_report_comparison_is_paired_and_bounded() -> None:
    session_base = {
        "sample_id": "one",
        "scenario_type": "buying",
        "hit": True,
        "first_hit_turn": 2,
        "best_rank": 2,
        "reciprocal_rank": 0.5,
    }
    session_better = {**session_base, "best_rank": 1, "reciprocal_rank": 1.0}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "baseline.json"
        candidate = root / "candidate.json"
        baseline.write_text(json.dumps({"sessions": [session_base]}), encoding="utf-8")
        candidate.write_text(
            json.dumps({"sessions": [session_better]}), encoding="utf-8"
        )
        result = compare_proposal_reports(baseline, {"score_leader": candidate})
    assert result["sample_count"] == 1
    assert result["candidates"]["score_leader"]["mean_paired_delta"] > 0
    assert result["decision_boundary"].startswith("comparison only")
