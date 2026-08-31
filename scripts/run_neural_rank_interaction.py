from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.retrieval.gbdt import (
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
    fit_lambdamart,
)
from ghostlab.retrieval.neural_rank import (
    MODEL_NAME,
    MODEL_REVISION,
    NEURAL_METADATA_FEATURES,
    PASSAGE_SCHEMA_VERSION,
    NeuralGBDTFeatureStore,
    NeuralScoreCache,
    PinnedCrossEncoderScorer,
    query_hash,
    score_cache_identity,
    write_score_cache,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from scripts.run_gbdt_reranker import (
    FIELD_WEIGHTS,
    QUESTION_ORDER,
    RankingGroup,
    build_agent,
    collect_groups,
    paired_evidence,
    sha256_file,
    stability,
    summarized_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/neural_rank_interaction_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/neural_rank_interaction_v1.json"
AUDITED_REPORT_PATH = ROOT / "artifacts/reports/gbdt_reranker_v1.json"
CACHE_PATH = ROOT / "artifacts/cache/neural_rank_scores_v1.jsonl"
RUNTIME_MODEL_PATH = ROOT / "artifacts/cache/neural_rank_outer_fold0_model.json"
CROSS_ENCODER_REPORT_PATH = (
    ROOT.parent
    / "techjam-cross-encoder/artifacts/reports/challenger_cross_encoder_v1.json"
)
SEED = 20260826


def trajectory_pairs(
    samples: dict[str, dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    quality: CatalogQualityReranker,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    pairs: list[tuple[str, str]] = []
    query_count = 0
    for sample_id in sorted(samples):
        environment = ReplayEnvironment(samples[sample_id], categories, products)
        observation = environment.observe()
        messages: list[str] = []
        while not environment.done:
            messages.append(observation.user_message)
            query = ". ".join(messages)
            ranking = [
                item.parent_asin
                for item in sparse.search(query, 200, FIELD_WEIGHTS).items
            ]
            ranking = quality.rerank(ranking, weight=0.2, rerank_k=50)
            pairs.extend((query, identifier) for identifier in ranking[:50])
            query_count += 1
            turn = observation.turn
            question = QUESTION_ORDER[turn - 1] if turn <= len(QUESTION_ORDER) else None
            next_observation = environment.step(
                {
                    "message": "neural score trajectory",
                    "ask_attribute": question,
                    "recommendations": [],
                }
            )
            if next_observation is not None:
                observation = next_observation
    return pairs, {
        "trajectory_queries": query_count,
        "trajectory_candidate_rows": len(pairs),
        "unique_query_candidate_rows": len(
            {(query_hash(query), identifier) for query, identifier in pairs}
        ),
    }


def load_or_generate_cache(
    pairs: list[tuple[str, str]],
    identity: dict[str, object],
    catalog_path: Path,
) -> tuple[NeuralScoreCache, dict[str, object]]:
    expected = {(query_hash(query), identifier) for query, identifier in pairs}
    cache: NeuralScoreCache | None = None
    if CACHE_PATH.exists():
        try:
            candidate = NeuralScoreCache(CACHE_PATH, identity)
            if set(candidate.scores) == expected:
                cache = candidate
        except (ValueError, KeyError, json.JSONDecodeError, StopIteration):
            cache = None
    generated = cache is None
    initialization_seconds = score_seconds = 0.0
    score_calls = scored_pairs = 0
    if cache is None:
        scorer = PinnedCrossEncoderScorer(
            catalog_path, ROOT / "artifacts/cache/cross_encoder"
        )
        write_score_cache(CACHE_PATH, identity, pairs, scorer)
        initialization_seconds = scorer.initialization_seconds
        score_seconds = scorer.score_seconds
        score_calls = scorer.score_calls
        scored_pairs = scorer.scored_pairs
        cache = NeuralScoreCache(CACHE_PATH, identity)
    return cache, {
        "path": str(CACHE_PATH.relative_to(ROOT)),
        "generated_this_run": generated,
        "identity": identity,
        "row_count": len(cache.scores),
        "sha256": sha256_file(CACHE_PATH),
        "bytes": CACHE_PATH.stat().st_size,
        "generation_initialization_seconds": round(initialization_seconds, 6),
        "generation_score_seconds": round(score_seconds, 6),
        "generation_score_calls": score_calls,
        "generation_scored_pairs": scored_pairs,
        "complete_for_all_frozen_trajectory_top50_pairs": set(cache.scores) == expected,
    }


def neural_groups(
    groups: dict[str, list[RankingGroup]], feature_store: NeuralGBDTFeatureStore
) -> dict[str, list[RankingGroup]]:
    return {
        sample_id: [
            replace(
                group,
                matrix=feature_store.matrix(
                    group.query, group.candidates, NEURAL_METADATA_FEATURES
                ),
            )
            for group in values
        ]
        for sample_id, values in groups.items()
    }


def ranking_dataset(
    groups: dict[str, list[RankingGroup]], sample_ids: set[str]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    selected = [
        group for sample_id in sorted(sample_ids) for group in groups.get(sample_id, [])
    ]
    if not selected:
        raise ValueError("ranking dataset cannot be empty")
    return (
        np.vstack([group.matrix for group in selected]),
        np.concatenate(
            [np.asarray(group.labels, dtype=np.int64) for group in selected]
        ),
        [len(group.labels) for group in selected],
    )


def train_model(
    groups: dict[str, list[RankingGroup]],
    training_ids: set[str],
    validation_ids: set[str],
    config: dict,
) -> tuple[LambdaMARTModel, int]:
    selected = fit_lambdamart(
        *ranking_dataset(groups, training_ids),
        candidate_id=str(config["candidate_id"]),
        feature_names=NEURAL_METADATA_FEATURES,
        max_depth=int(config["max_depth"]),
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        max_rounds=int(config["max_rounds"]),
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        validation=ranking_dataset(groups, validation_ids),
        seed=SEED,
    )
    refit = fit_lambdamart(
        *ranking_dataset(groups, training_ids | validation_ids),
        candidate_id=str(config["candidate_id"]),
        feature_names=NEURAL_METADATA_FEATURES,
        max_depth=int(config["max_depth"]),
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        max_rounds=selected.best_iteration,
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        validation=None,
        seed=SEED,
    )
    return refit, selected.best_iteration


def scenario_comparison(candidate: dict, control: dict) -> dict[str, dict[str, float]]:
    names = set(candidate["scenario_metrics"]) | set(control["scenario_metrics"])
    result = {}
    for name in sorted(names):
        left = candidate["scenario_metrics"][name]
        right = control["scenario_metrics"][name]
        result[name] = {
            "hit_rate_at_10_delta": round(
                float(left["hit_rate_at_10"]) - float(right["hit_rate_at_10"]), 6
            ),
            "mrr_delta": round(float(left["mrr"]) - float(right["mrr"]), 6),
            "mttc_delta": round(float(left["mttc"]) - float(right["mttc"]), 6),
        }
    return result


def standalone_ce_reference() -> dict[str, object]:
    report = json.loads(CROSS_ENCODER_REPORT_PATH.read_text(encoding="utf-8"))
    return {
        "source_commit": "071eda9",
        "source_report_sha256": sha256_file(CROSS_ENCODER_REPORT_PATH),
        "candidate": "nested selected zero-shot cross-encoder",
        "metrics": report["nested_selected_metrics"],
        "paired_vs_linear_champion": report["nested_selected_vs_linear_champion"],
        "decision": report["decision"],
    }


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False:
        raise RuntimeError("interaction manifest does not preserve holdout firewall")
    if (
        manifest["cross_encoder"]["model"] != MODEL_NAME
        or manifest["cross_encoder"]["model_revision"] != MODEL_REVISION
        or manifest["cross_encoder"]["passage_schema_version"] != PASSAGE_SCHEMA_VERSION
    ):
        raise ValueError("manifest and pinned cross-encoder implementation differ")

    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    sparse = SparseIndex(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    base_features = GBDTFeatureStore(catalog_path, quality=quality.quality)

    pairs, trajectory = trajectory_pairs(samples, categories, products, sparse, quality)
    identity = score_cache_identity(sha256_file(catalog_path))
    cache, cache_report = load_or_generate_cache(pairs, identity, catalog_path)
    feature_store = NeuralGBDTFeatureStore(base_features, cache=cache)
    base_groups, collection = collect_groups(
        samples, categories, products, sparse, quality, base_features
    )
    groups = neural_groups(base_groups, feature_store)
    if feature_store.missing_count:
        raise RuntimeError("predeclared trajectory score cache is incomplete")
    print(json.dumps({**trajectory, **collection}, sort_keys=True), flush=True)

    audited = json.loads(AUDITED_REPORT_PATH.read_text(encoding="utf-8"))
    control = audited["variants"]["shallow_metadata_depth3"]
    control_sessions = control["oof_sessions"]
    if {str(item["sample_id"]) for item in control_sessions} != adaptive_ids:
        raise ValueError("audited GBDT control is not the complete adaptive OOF set")

    candidate_config = manifest["candidate"]
    oof_sessions: list[dict] = []
    folds = []
    importance = {name: 0 for name in NEURAL_METADATA_FEATURES}
    runtime_model: LambdaMARTModel | None = None
    for outer_index, outer_ids in enumerate(outer_folds):
        inner_index = (outer_index + 1) % len(outer_folds)
        inner_validation_ids = outer_folds[inner_index]
        inner_training_ids = adaptive_ids - outer_ids - inner_validation_ids
        model, selected_rounds = train_model(
            groups, inner_training_ids, inner_validation_ids, candidate_config
        )
        if outer_index == 0:
            runtime_model = model
        for name, count in model.split_importance().items():
            importance[name] += count
        result = evaluate(
            build_agent(quality, LambdaMARTReranker(feature_store, model)),
            [samples[sample_id] for sample_id in sorted(outer_ids)],
            catalog_ids,
            categories,
            products,
        )
        fold_control = control["folds"][outer_index]["outer_metrics"]
        fold_candidate = summarized_metrics(result["sessions"])
        oof_sessions.extend(result["sessions"])
        folds.append(
            {
                "outer_fold": outer_index,
                "inner_training_ids": sorted(inner_training_ids),
                "inner_validation_ids": sorted(inner_validation_ids),
                "outer_training_ids": sorted(adaptive_ids - outer_ids),
                "outer_validation_ids": sorted(outer_ids),
                "inner_selected_rounds": selected_rounds,
                "refit_training_groups": model.training_groups,
                "refit_training_rows": model.training_rows,
                "outer_metrics": fold_candidate,
                "audited_control_outer_metrics": fold_control,
                "technical_score_delta": round(
                    float(fold_candidate["recommended_technical_score"])
                    - float(fold_control["recommended_technical_score"]),
                    6,
                ),
            }
        )
        print(
            f"neural interaction fold={outer_index} rounds={selected_rounds} "
            f"score={fold_candidate['recommended_technical_score']}",
            flush=True,
        )

    if feature_store.missing_count:
        raise RuntimeError("complete OOF evaluation encountered a missing neural score")
    candidate_metrics = summarized_metrics(oof_sessions)
    paired = paired_evidence(oof_sessions, control_sessions)
    scenario_deltas = scenario_comparison(candidate_metrics, control["oof_metrics"])
    fold_deltas = [float(item["technical_score_delta"]) for item in folds]

    assert runtime_model is not None
    RUNTIME_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    runtime_model.save(RUNTIME_MODEL_PATH)
    performance_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.measure_neural_rank_runtime",
            str(RUNTIME_MODEL_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    performance = json.loads(performance_result.stdout.strip().splitlines()[-1])
    budgets = {
        "cold_start_seconds": 30.0,
        "warm_turn_p95_ms": 500.0,
        "peak_process_memory_mb": 4096.0,
        "total_model_asset_mb": 500.0,
    }
    performance_passed = (
        float(performance["cold_start_seconds"]) <= budgets["cold_start_seconds"]
        and float(performance["warm_turn_p95_ms"]) <= budgets["warm_turn_p95_ms"]
        and float(performance["peak_process_memory_mb"])
        <= budgets["peak_process_memory_mb"]
        and float(performance["total_model_asset_mb"])
        <= budgets["total_model_asset_mb"]
        and int(performance["failure_count"]) == 0
        and int(performance["external_calls_per_turn"]) == 0
        and int(performance["missing_score_count"]) == 0
    )

    standalone = standalone_ce_reference()
    linear_score = float(
        audited["controls"]["two_feature_linear_champion"]["oof_metrics"][
            "recommended_technical_score"
        ]
    )
    gbdt_gain = (
        float(control["oof_metrics"]["recommended_technical_score"]) - linear_score
    )
    standalone_ce_gain = float(
        standalone["paired_vs_linear_champion"]["mean_paired_delta"]  # type: ignore[index]
    )
    combination_gain = (
        float(candidate_metrics["recommended_technical_score"]) - linear_score
    )
    interaction_gain = combination_gain - gbdt_gain - standalone_ce_gain

    promotion_checks = {
        "mean_paired_delta_at_least_0_005": float(
            paired["mean_paired_session_reward_delta"]
        )
        >= 0.005,
        "bootstrap_lower_at_least_minus_0_002": float(
            paired["paired_bootstrap_95_interval"][0]
        )
        >= -0.002,
        "hit_at_10_non_regression": float(candidate_metrics["hit_rate_at_10"])
        >= float(control["oof_metrics"]["hit_rate_at_10"]),
        "at_least_four_positive_folds": sum(delta > 0.0 for delta in fold_deltas) >= 4,
        "no_fold_below_minus_0_005": min(fold_deltas) >= -0.005,
        "no_scenario_hit_regression_below_minus_0_025": min(
            value["hit_rate_at_10_delta"] for value in scenario_deltas.values()
        )
        >= -0.025,
        "cross_encoder_score_used": importance["cross_encoder_score"] > 0,
        "cache_complete_without_missingness": feature_store.missing_count == 0,
        "runtime_and_asset_budgets": performance_passed,
    }
    promoted = all(promotion_checks.values())
    decision = "PROMOTE_TO_INTEGRATION" if promoted else "PARK_INTERACTION"
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "predeclaration_commit": "1c0072b",
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "evidence_label": "outer-fold/out-of-fold",
        "split": manifest["split"],
        "split_sha256": sha256_file(nested_path),
        "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": sha256_file(catalog_path),
        "holdout_accessed": False,
        "f3_available": False,
        "seed": SEED,
        "code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "ghostlab/retrieval/gbdt.py",
                ROOT / "ghostlab/retrieval/neural_rank.py",
                ROOT / "scripts/run_gbdt_reranker.py",
                ROOT / "scripts/run_neural_rank_interaction.py",
                ROOT / "scripts/measure_neural_rank_runtime.py",
            )
        },
        "fixed_configuration": candidate_config,
        "cross_encoder": manifest["cross_encoder"],
        "trajectory_collection": {**trajectory, **collection},
        "score_cache": cache_report,
        "control": {
            "candidate_id": "shallow_metadata_depth3",
            "audited_report_path": str(AUDITED_REPORT_PATH.relative_to(ROOT)),
            "audited_report_sha256": sha256_file(AUDITED_REPORT_PATH),
            "oof_metrics": control["oof_metrics"],
            "stability": control["stability"],
        },
        "candidate": {
            "candidate_id": candidate_config["candidate_id"],
            "oof_metrics": candidate_metrics,
            "oof_sessions": sorted(
                oof_sessions, key=lambda item: str(item["sample_id"])
            ),
            "folds": folds,
            "stability": stability(folds),
            "paired_vs_audited_gbdt": paired,
            "scenario_deltas_vs_audited_gbdt": scenario_deltas,
            "feature_importance_split_count_across_outer_models": dict(
                sorted(importance.items(), key=lambda item: (-item[1], item[0]))
            ),
        },
        "backward_ablations": {
            "remove_cross_encoder_score": {
                "candidate": "audited shallow_metadata_depth3",
                "metrics": control["oof_metrics"],
                "paired_candidate_minus_ablation": paired,
            },
            "remove_gbdt": standalone,
        },
        "interaction_analysis": {
            "baseline": "two_feature_linear_champion",
            "baseline_score": round(linear_score, 6),
            "gbdt_gain": round(gbdt_gain, 6),
            "standalone_cross_encoder_gain": round(standalone_ce_gain, 6),
            "combination_gain": round(combination_gain, 6),
            "interaction_gain": round(interaction_gain, 6),
            "formula": "gain(GBDT+CE) - gain(GBDT) - gain(CE)",
        },
        "performance": {
            **performance,
            "budgets": budgets,
            "passed": performance_passed,
        },
        "selection": {
            "promotion_rule": manifest["promotion_rule"],
            "checks": promotion_checks,
            "decision": decision,
            "deployable_model_written": False,
            "rationale": (
                "The single predeclared interaction cleared every quality, stability, semantic-use, and cost gate; advance the minimal implementation for integration refit."
                if promoted
                else "The single predeclared interaction failed at least one promotion gate; preserve evidence and do not create a deployable neural model artifact."
            ),
        },
        "fit_audit": {
            "outer_model_count": len(folds),
            "inner_selection_fit_count": len(folds),
            "outer_training_refit_count": len(folds),
            "all_learned_tree_fits_inside_folds": True,
            "all_development_refit_performed": False,
            "runtime_measurement_model": "outer fold 0 model; cost reference only",
        },
        "failure_status": None,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "within_wall_clock_limit": time.perf_counter() - started
        <= float(manifest["wall_clock_limit_seconds"]),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "control": control["oof_metrics"],
                "candidate": candidate_metrics,
                "paired": paired,
                "fold_deltas": fold_deltas,
                "interaction": report["interaction_analysis"],
                "performance": report["performance"],
                "selection": report["selection"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
