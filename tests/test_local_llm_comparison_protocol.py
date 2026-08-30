from __future__ import annotations

from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from scripts.compare_local_llm_rankers import (
    MODELS,
    ROOT,
    lineage_safe_sample_ids,
    paired_trial_evidence,
    select_model_winners,
    symmetric_trial_matrix,
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


def test_candidate_pool_or_session_mismatch_invalidates_pairing() -> None:
    common = {
        "status": "complete",
        "ordered_session_ids_sha256": "sessions-a",
        "candidate_pool_sha256": "pool-a",
    }
    assert paired_trial_evidence([common, dict(common)])["paired_candidate_pools"]

    different_pool = {**common, "candidate_pool_sha256": "pool-b"}
    assert not paired_trial_evidence([common, different_pool])[
        "paired_candidate_pools"
    ]

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
