from __future__ import annotations

import json
from pathlib import Path

from ghostlab.runtime.adaptive_components import (
    CausalRelevanceScorer,
    causal_chat_template_options,
)
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from scripts.compare_local_llm_rankers import (
    MODELS,
    ROOT,
    lineage_safe_sample_ids,
    paired_trial_evidence,
    required_asset_evidence,
    select_model_winners,
    symmetric_trial_matrix,
    trial_ledger_evidence,
)


def _result(model_id: str, *, depth: int, weight: float, score: float) -> dict:
    return {
        "model_id": model_id,
        "depth": depth,
        "weight": weight,
        "primary_backend_valid": True,
        "output_constraint_violations": 0,
        "confirmed_target_removal_count": 0,
        "target_demoted_from_top10": 0,
        "recommended_technical_score": score,
        "mrr": score,
        "hit_rate_at_10": score,
        "fallback_rate": 0.0,
        "p95_semantic_latency_ms": 10.0,
        "peak_worker_memory_mb": 100.0,
    }


def test_four_models_receive_the_same_depth_weight_grid() -> None:
    matrix = symmetric_trial_matrix(tuple(MODELS), (10, 20, 30), (0.2, 0.5))
    expected = {(depth, weight) for depth in (10, 20, 30) for weight in (0.2, 0.5)}
    assert len(matrix) == len(MODELS) * len(expected)
    for model_id in MODELS:
        actual = {
            (item["depth"], item["weight"])
            for item in matrix
            if item["model_id"] == model_id
        }
        assert actual == expected


def test_each_model_selects_its_own_optimal_setting() -> None:
    results = []
    for index, model_id in enumerate(MODELS):
        results.extend(
            (
                _result(model_id, depth=10, weight=0.2, score=0.7 + index * 0.01),
                _result(model_id, depth=20, weight=0.5, score=0.8 + index * 0.01),
            )
        )
    winners = select_model_winners(results)
    assert [item["model_id"] for item in winners] == list(MODELS)
    assert all(item["depth"] == 20 and item["weight"] == 0.5 for item in winners)


def test_complete_trial_ledger_counts_failures_as_attempted() -> None:
    matrix = symmetric_trial_matrix(("first", "second"), (10,), (0.2, 0.5))
    results = [
        {**trial, "status": "failed" if index == 0 else "complete"}
        for index, trial in enumerate(matrix)
    ]
    evidence = trial_ledger_evidence(matrix, results)
    assert evidence["all_trials_attempted_once"] is True
    assert evidence["attempted_trial_count"] == 4
    assert evidence["status_counts"] == {"complete": 3, "failed": 1}


def test_missing_or_duplicate_grid_trial_invalidates_ledger() -> None:
    matrix = symmetric_trial_matrix(("first", "second"), (10,), (0.2, 0.5))
    missing = [{**trial, "status": "complete"} for trial in matrix[:-1]]
    assert trial_ledger_evidence(matrix, missing)["all_trials_attempted_once"] is False
    duplicated = [*missing, missing[0], {**matrix[-1], "status": "complete"}]
    assert (
        trial_ledger_evidence(matrix, duplicated)["all_trials_attempted_once"] is False
    )


def test_partial_asset_directory_is_not_verified(tmp_path: Path) -> None:
    manifest_path = tmp_path / "asset.json"
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "config.json").write_text("{}", encoding="utf-8")
    (destination / "model.safetensors").write_bytes(b"partial")
    manifest_path.write_text(
        json.dumps(
            {
                "destination": "model",
                "model_name": "example/model",
                "revision": "pinned",
            }
        ),
        encoding="utf-8",
    )
    evidence = required_asset_evidence(
        {"example": {"path": "model", "manifest": "asset.json"}},
        root=tmp_path,
    )
    assert evidence["all_verified"] is False
    assert evidence["models"][0]["verified"] is False


def test_complete_receipted_asset_is_verified(tmp_path: Path) -> None:
    manifest_path = tmp_path / "asset.json"
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "config.json").write_text("{}", encoding="utf-8")
    (destination / "model.safetensors").write_bytes(b"complete")
    (destination / ".ghostlab_asset.json").write_text(
        json.dumps({"model_name": "example/model", "revision": "pinned"}),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "destination": "model",
                "model_name": "example/model",
                "revision": "pinned",
            }
        ),
        encoding="utf-8",
    )
    evidence = required_asset_evidence(
        {"example": {"path": "model", "manifest": "asset.json"}},
        root=tmp_path,
    )
    assert evidence["all_verified"] is True


def test_qwen3_disables_thinking_for_direct_yes_no_scoring() -> None:
    assert causal_chat_template_options("qwen3-0.6b") == {"enable_thinking": False}
    assert causal_chat_template_options("Qwen/Qwen3-0.6B") == {"enable_thinking": False}
    assert causal_chat_template_options("gemma-3-1b-it") == {}

    class RecordingTokenizer:
        chat_template = "template"

        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def apply_chat_template(self, messages: object, **options: object) -> str:
            del messages
            self.options = options
            return "rendered prompt"

    tokenizer = RecordingTokenizer()
    scorer = object.__new__(CausalRelevanceScorer)
    scorer.tokenizer = tokenizer
    scorer.chat_template_options = causal_chat_template_options("qwen3-0.6b")
    assert scorer._prompt("request", "product") == "rendered prompt"
    assert tokenizer.options["enable_thinking"] is False


def test_candidate_pool_or_session_mismatch_invalidates_pairing() -> None:
    common = {
        "status": "complete",
        "ordered_session_ids_sha256": "sessions-a",
        "candidate_pool_sha256": "pool-a",
    }
    assert paired_trial_evidence([common, dict(common)])["paired_candidate_pools"]

    different_pool = {**common, "candidate_pool_sha256": "pool-b"}
    assert not paired_trial_evidence([common, different_pool])["paired_candidate_pools"]

    different_sessions = {**common, "ordered_session_ids_sha256": "sessions-b"}
    assert not paired_trial_evidence([common, different_sessions])[
        "paired_ordered_sessions"
    ]


def test_development_sample_selection_preserves_lineage_folds() -> None:
    datasets = (
        "data/public_set.jsonl",
        "data/synthetic_1000_public_like.jsonl",
        "data/independent_template_1000.jsonl",
    )
    corpus = load_adaptive_training_corpus(ROOT, datasets)
    manifest = load_lineage_manifest(
        ROOT / "data/splits/adaptive_hybrid_lineage_75_25_v1.json", corpus
    )
    development = subset_corpus(corpus, manifest, "development")
    folds = lineage_safe_sample_ids(development, manifest, 60)
    assert len(folds) == 5
    assert all(fold for fold in folds)
    owners = {}
    for fold_index, fold in enumerate(folds):
        for sample_id in fold:
            assert sample_id in manifest.development_ids
            assert development.samples[sample_id]["scenario_type"] == "browsing"
            group = manifest.group_by_sample[sample_id]
            assert owners.setdefault(group, fold_index) == fold_index


def test_candidate_pool_pairing_ignores_random_runtime_session_ids() -> None:
    from ghostlab.runtime.adaptive_hybrid import AdaptiveCandidateSnapshot
    from scripts.compare_local_llm_rankers import _candidate_pool_evidence

    def agent_with(session_id: str):
        agent = type("AgentEvidence", (), {})()
        agent.candidate_snapshots = [
            AdaptiveCandidateSnapshot(
                session_id=session_id,
                turn=2,
                query="summer wedding",
                route="browsing",
                candidates=("A", "B", "C"),
                overloaded=False,
                pre_semantic_candidates=("B", "A", "C"),
            )
        ]
        agent.traces = [
            type(
                "TraceEvidence",
                (),
                {"session_id": session_id, "turn": 2, "semantic_executed": True},
            )()
        ]
        return agent

    left = _candidate_pool_evidence(agent_with("random-a"))
    right = _candidate_pool_evidence(agent_with("random-b"))

    assert left == right
    assert left[1] == 1


def test_semantic_rescue_uses_runtime_to_dataset_session_mapping() -> None:
    from ghostlab.runtime.adaptive_hybrid import AdaptiveCandidateSnapshot
    from scripts.compare_local_llm_rankers import _semantic_rescue_metrics

    agent = type("AgentEvidence", (), {})()
    pre_semantic = (*(f"P{index}" for index in range(10)), "TARGET")
    agent.candidate_snapshots = [
        AdaptiveCandidateSnapshot(
            session_id="runtime-random",
            turn=2,
            query="summer wedding",
            route="browsing",
            candidates=pre_semantic,
            overloaded=False,
            pre_semantic_candidates=pre_semantic,
            post_semantic_candidates=("TARGET", *pre_semantic[:-1]),
        )
    ]
    agent.traces = [
        type(
            "TraceEvidence",
            (),
            {
                "session_id": "runtime-random",
                "turn": 2,
                "semantic_executed": True,
                "top_ids": ("A", "B"),
            },
        )()
    ]
    samples = {"dataset-id": {"ground_truth": {"parent_asin": "TARGET"}}}

    result = _semantic_rescue_metrics(
        agent,
        samples,
        {"runtime-random": "dataset-id"},
    )

    assert result["semantic_target_turns"] == 1
    assert result["target_rescued_into_top10"] == 1
    assert result["target_demoted_from_top10"] == 0
