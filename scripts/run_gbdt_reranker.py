from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
)
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.research.replay import ReplayEnvironment, session_reward
from ghostlab.retrieval.gbdt import (
    FEATURE_SETS,
    METADATA_FEATURES,
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
    fit_lambdamart,
)
from ghostlab.retrieval.learned import (
    CandidateFeatureStore,
    LearnedLinearReranker,
    PairwiseExample,
    fit_pairwise_linear,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.experimental import CandidateReranker, ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/gbdt_reranker_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/gbdt_reranker_v1.json"
MODEL_PATH = ROOT / "artifacts/models/gbdt_reranker_v1.json"
FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
QUESTION_ORDER = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)
LINEAR_FEATURES = ("feature_overlap", "catalog_quality")
SEED = 20260826


@dataclass(frozen=True)
class RankingGroup:
    sample_id: str
    turn: int
    query: str
    candidates: tuple[str, ...]
    labels: tuple[int, ...]
    matrix: NDArray[np.float64]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_groups(
    samples: dict[str, dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    quality: CatalogQualityReranker,
    features: GBDTFeatureStore,
) -> tuple[dict[str, list[RankingGroup]], dict[str, int]]:
    by_sample: dict[str, list[RankingGroup]] = defaultdict(list)
    trajectory_queries = positive_queries = candidate_rows = 0
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        messages: list[str] = []
        target = str(sample["ground_truth"]["parent_asin"])
        while not environment.done:
            messages.append(observation.user_message)
            query = ". ".join(messages)
            ranking = [
                item.parent_asin
                for item in sparse.search(query, 200, FIELD_WEIGHTS).items
            ]
            ranking = quality.rerank(ranking, weight=0.2, rerank_k=50)
            head = tuple(ranking[:50])
            trajectory_queries += 1
            if target in head:
                positive_queries += 1
                labels = tuple(int(identifier == target) for identifier in head)
                by_sample[sample_id].append(
                    RankingGroup(
                        sample_id=sample_id,
                        turn=observation.turn,
                        query=query,
                        candidates=head,
                        labels=labels,
                        matrix=features.matrix(query, head, METADATA_FEATURES),
                    )
                )
                candidate_rows += len(head)
            turn = observation.turn
            question = QUESTION_ORDER[turn - 1] if turn <= len(QUESTION_ORDER) else None
            next_observation = environment.step(
                {
                    "message": "training trajectory",
                    "ask_attribute": question,
                    "recommendations": [],
                }
            )
            if next_observation is not None:
                observation = next_observation
    return dict(by_sample), {
        "trajectory_queries": trajectory_queries,
        "queries_with_target_in_top50": positive_queries,
        "samples_with_ranking_groups": len(by_sample),
        "ranking_groups": sum(len(values) for values in by_sample.values()),
        "candidate_rows": candidate_rows,
    }


def ranking_dataset(
    groups: dict[str, list[RankingGroup]],
    sample_ids: set[str],
    feature_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[int]]:
    indices = [METADATA_FEATURES.index(name) for name in feature_names]
    selected = [
        group for sample_id in sorted(sample_ids) for group in groups.get(sample_id, [])
    ]
    if not selected:
        raise ValueError("ranking dataset cannot be empty")
    return (
        np.vstack([group.matrix[:, indices] for group in selected]),
        np.concatenate(
            [np.asarray(group.labels, dtype=np.int64) for group in selected]
        ),
        [len(group.labels) for group in selected],
    )


def linear_examples(
    groups: dict[str, list[RankingGroup]], feature_store: CandidateFeatureStore
) -> dict[str, list[PairwiseExample]]:
    result: dict[str, list[PairwiseExample]] = defaultdict(list)
    for sample_id, sample_groups in groups.items():
        for group in sample_groups:
            target_index = group.labels.index(1)
            target = group.candidates[target_index]
            count = len(group.candidates)
            target_base = 1.0 if count == 1 else 1.0 - target_index / max(1, count - 1)
            target_features = feature_store.features(group.query, target)
            negatives = list(
                dict.fromkeys([*group.candidates[:20], *group.candidates[20:50:5]])
            )
            for identifier in negatives:
                if identifier == target:
                    continue
                negative_index = group.candidates.index(identifier)
                negative_base = (
                    1.0 if count == 1 else 1.0 - negative_index / max(1, count - 1)
                )
                negative_features = feature_store.features(group.query, identifier)
                result[sample_id].append(
                    PairwiseExample(
                        base_margin=target_base - negative_base,
                        feature_delta=tuple(
                            left - right
                            for left, right in zip(
                                target_features, negative_features, strict=True
                            )
                        ),
                    )
                )
    return dict(result)


def build_agent(
    quality: CatalogQualityReranker,
    reranker: CandidateReranker | None,
) -> ExperimentalAgent:
    return ExperimentalAgent(
        ROOT / "data/catalog.jsonl",
        state_variant="raw_history",
        question_variant="sequence",
        question_order=QUESTION_ORDER,
        negative_evidence=False,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        quality_prior=quality,
        learned_reranker=reranker,
    )


def summarized_metrics(sessions: list[dict]) -> dict[str, object]:
    overall = metric_summary(sessions)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "recommended_technical_score": round(
            statistics.fmean(session_reward(session) for session in sessions), 6
        ),
        "scenario_metrics": {
            name: metric_summary(values) for name, values in sorted(grouped.items())
        },
    }


def fold_metrics(sessions: list[dict], fold_ids: set[str]) -> dict[str, object]:
    return summarized_metrics(
        [session for session in sessions if str(session["sample_id"]) in fold_ids]
    )


def paired_evidence(candidate: list[dict], control: list[dict]) -> dict[str, object]:
    control_rewards = {
        str(session["sample_id"]): session_reward(session) for session in control
    }
    deltas = [
        session_reward(session) - control_rewards[str(session["sample_id"])]
        for session in candidate
    ]
    lower, upper = bootstrap_mean_interval(deltas, resamples=10000, seed=SEED)
    tolerance = 1e-12
    return {
        "mean_paired_session_reward_delta": round(statistics.fmean(deltas), 6),
        "paired_bootstrap_95_interval": [round(lower, 6), round(upper, 6)],
        "paired_randomization_p_value": round(
            paired_randomization_pvalue(deltas, resamples=10000, seed=SEED), 6
        ),
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
        "resamples": 10000,
    }


def stability(folds: list[dict]) -> dict[str, object]:
    scores = [
        float(fold["outer_metrics"]["recommended_technical_score"]) for fold in folds
    ]
    return {
        "fold_scores": scores,
        "mean": round(statistics.fmean(scores), 6),
        "standard_deviation": round(statistics.pstdev(scores), 6),
        "worst_fold": round(min(scores), 6),
    }


def variant_config(manifest: dict, candidate_id: str) -> dict:
    return next(
        value
        for value in manifest["candidates"]
        if value["candidate_id"] == candidate_id
    )


def train_model(
    groups: dict[str, list[RankingGroup]],
    training_ids: set[str],
    validation_ids: set[str],
    config: dict,
) -> tuple[LambdaMARTModel, int]:
    feature_names = FEATURE_SETS[str(config["feature_set"])]
    training = ranking_dataset(groups, training_ids, feature_names)
    validation = ranking_dataset(groups, validation_ids, feature_names)
    selected = fit_lambdamart(
        *training,
        candidate_id=str(config["candidate_id"]),
        feature_names=feature_names,
        max_depth=int(config["max_depth"]),
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        max_rounds=int(config["max_rounds"]),
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        validation=validation,
        seed=SEED,
    )
    refit = fit_lambdamart(
        *ranking_dataset(groups, training_ids | validation_ids, feature_names),
        candidate_id=str(config["candidate_id"]),
        feature_names=feature_names,
        max_depth=int(config["max_depth"]),
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        max_rounds=selected.best_iteration,
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        validation=None,
        seed=SEED,
    )
    return refit, selected.best_iteration


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False:
        raise RuntimeError("experiment manifest does not preserve holdout firewall")
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
    gbdt_features = GBDTFeatureStore(catalog_path, quality=quality.quality)
    groups, collection = collect_groups(
        samples, categories, products, sparse, quality, gbdt_features
    )
    print(json.dumps(collection, sort_keys=True), flush=True)

    fixed_result = evaluate(
        build_agent(quality, None),
        [samples[sample_id] for sample_id in sorted(samples)],
        catalog_ids,
        categories,
        products,
    )
    controls: dict[str, dict] = {
        "fixed_field_quality": {
            "evidence_label": "outer-fold/out-of-fold",
            "oof_metrics": summarized_metrics(fixed_result["sessions"]),
            "oof_sessions": fixed_result["sessions"],
            "folds": [
                {
                    "outer_fold": index,
                    "outer_metrics": fold_metrics(fixed_result["sessions"], fold),
                }
                for index, fold in enumerate(outer_folds)
            ],
        }
    }

    linear_store = CandidateFeatureStore(
        catalog_path,
        enabled_features=LINEAR_FEATURES,
        quality=quality.quality,
    )
    examples = linear_examples(groups, linear_store)
    linear_sessions: list[dict] = []
    linear_folds = []
    for outer_index, outer_ids in enumerate(outer_folds):
        training_ids = adaptive_ids - outer_ids
        training_examples = [
            example
            for sample_id in sorted(training_ids)
            for example in examples.get(sample_id, [])
        ]
        model = fit_pairwise_linear(training_examples, enabled_features=LINEAR_FEATURES)
        result = evaluate(
            build_agent(quality, LearnedLinearReranker(linear_store, model)),
            [samples[sample_id] for sample_id in sorted(outer_ids)],
            catalog_ids,
            categories,
            products,
        )
        linear_sessions.extend(result["sessions"])
        linear_folds.append(
            {
                "outer_fold": outer_index,
                "outer_training_ids": sorted(training_ids),
                "outer_validation_ids": sorted(outer_ids),
                "training_pairs": len(training_examples),
                "model": {"weights": list(model.weights), "l2": model.l2},
                "outer_metrics": summarized_metrics(result["sessions"]),
            }
        )
        print(
            f"linear fold={outer_index} score={result['recommended_technical_score']}",
            flush=True,
        )
    controls["two_feature_linear_champion"] = {
        "evidence_label": "outer-fold/out-of-fold",
        "oof_metrics": summarized_metrics(linear_sessions),
        "oof_sessions": linear_sessions,
        "folds": linear_folds,
        "stability": stability(linear_folds),
        "paired_vs_fixed": paired_evidence(linear_sessions, fixed_result["sessions"]),
    }

    variants: dict[str, dict] = {}
    for config in manifest["candidates"]:
        candidate_id = str(config["candidate_id"])
        oof_sessions: list[dict] = []
        folds = []
        importance = {name: 0 for name in FEATURE_SETS[str(config["feature_set"])]}
        for outer_index, outer_ids in enumerate(outer_folds):
            inner_index = (outer_index + 1) % len(outer_folds)
            inner_validation_ids = outer_folds[inner_index]
            inner_training_ids = adaptive_ids - outer_ids - inner_validation_ids
            model, selected_rounds = train_model(
                groups,
                inner_training_ids,
                inner_validation_ids,
                config,
            )
            for name, count in model.split_importance().items():
                importance[name] += count
            result = evaluate(
                build_agent(
                    quality,
                    LambdaMARTReranker(gbdt_features, model),
                ),
                [samples[sample_id] for sample_id in sorted(outer_ids)],
                catalog_ids,
                categories,
                products,
            )
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
                    "outer_metrics": summarized_metrics(result["sessions"]),
                }
            )
            print(
                f"{candidate_id} fold={outer_index} rounds={selected_rounds} "
                f"score={result['recommended_technical_score']}",
                flush=True,
            )
        variants[candidate_id] = {
            "evidence_label": "outer-fold/out-of-fold",
            "configuration": config,
            "oof_metrics": summarized_metrics(oof_sessions),
            "oof_sessions": oof_sessions,
            "folds": folds,
            "stability": stability(folds),
            "paired_vs_two_feature_linear": paired_evidence(
                oof_sessions, linear_sessions
            ),
            "paired_vs_fixed": paired_evidence(oof_sessions, fixed_result["sessions"]),
            "feature_importance_split_count_across_outer_models": dict(
                sorted(importance.items(), key=lambda item: (-item[1], item[0]))
            ),
        }

    nonlinear_ids = [
        str(config["candidate_id"])
        for config in manifest["candidates"]
        if str(config["feature_set"]) != "rank_only"
    ]
    best_score = max(
        float(variants[candidate_id]["oof_metrics"]["recommended_technical_score"])
        for candidate_id in nonlinear_ids
    )
    selected_id = next(
        candidate_id
        for candidate_id in nonlinear_ids
        if float(variants[candidate_id]["oof_metrics"]["recommended_technical_score"])
        >= best_score - 0.005
    )
    selected_config = variant_config(manifest, selected_id)
    refit_validation_ids = outer_folds[0]
    refit_training_ids = adaptive_ids - refit_validation_ids
    deployable_model, selected_rounds = train_model(
        groups, refit_training_ids, refit_validation_ids, selected_config
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    deployable_model.save(MODEL_PATH)
    full_result = evaluate(
        build_agent(
            quality,
            LambdaMARTReranker(gbdt_features, deployable_model),
        ),
        [samples[sample_id] for sample_id in sorted(samples)],
        catalog_ids,
        categories,
        products,
    )
    performance_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.measure_gbdt_runtime",
            str(MODEL_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    performance = json.loads(performance_result.stdout.strip().splitlines()[-1])
    selected_variant = variants[selected_id]
    selected_metrics = selected_variant["oof_metrics"]
    linear_metrics = controls["two_feature_linear_champion"]["oof_metrics"]
    paired = selected_variant["paired_vs_two_feature_linear"]
    performance_passed = (
        performance["cold_start_seconds"] <= 30.0
        and performance["warm_turn_p95_ms"] <= 500.0
        and performance["peak_process_memory_mb"] <= 4096.0
        and performance["model_asset_mb"] <= 500.0
        and performance["failure_count"] == 0
        and performance["external_calls_per_turn"] == 0
    )
    promotion_passed = (
        float(selected_metrics["recommended_technical_score"])
        > float(linear_metrics["recommended_technical_score"])
        and float(selected_metrics["hit_rate_at_10"])
        >= float(linear_metrics["hit_rate_at_10"])
        and float(paired["paired_bootstrap_95_interval"][0]) >= -0.005
        and performance_passed
    )
    decision = "PROMOTE" if promotion_passed else "PARK_STANDALONE"
    rationale = (
        "The selected nonlinear candidate passed every predeclared evidence and runtime gate."
        if promotion_passed
        else "The selected nonlinear candidate failed at least one predeclared promotion gate; preserve its grouped OOF evidence and retain the two-feature linear champion."
    )
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "policy_control": manifest["parent_policy_id"],
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "evidence_label": "outer-fold/out-of-fold",
        "split": "nested_v1",
        "split_sha256": sha256_file(nested_path),
        "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": sha256_file(catalog_path),
        "holdout_accessed": False,
        "seed": SEED,
        "dependency": {
            "training": "scikit-learn decision-tree builder from frozen uv.lock",
            "runtime": "NumPy-only serialized tree traversal",
            "lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
        "code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "ghostlab/retrieval/gbdt.py",
                ROOT / "ghostlab/runtime/experimental.py",
                ROOT / "scripts/run_gbdt_reranker.py",
                ROOT / "scripts/measure_gbdt_runtime.py",
            )
        },
        "failure_status": None,
        "collection": collection,
        "controls": controls,
        "variants": variants,
        "selection": {
            "selected_candidate_id": selected_id,
            "selection_rule": manifest["promotion_rule"],
            "within_best_tie_band": 0.005,
            "decision": decision,
            "rationale": rationale,
        },
        "all_development_refit": {
            "evidence_label": "all-development refit",
            "inner_training_ids": sorted(refit_training_ids),
            "inner_validation_ids": sorted(refit_validation_ids),
            "inner_selected_rounds": selected_rounds,
            "model_path": str(MODEL_PATH.relative_to(ROOT)),
            "model_sha256": sha256_file(MODEL_PATH),
            "model": {
                "candidate_id": deployable_model.candidate_id,
                "feature_names": list(deployable_model.feature_names),
                "best_iteration": deployable_model.best_iteration,
                "training_groups": deployable_model.training_groups,
                "training_rows": deployable_model.training_rows,
                "feature_importance_split_count": dict(
                    sorted(
                        deployable_model.split_importance().items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ),
            },
            "metrics": {
                key: full_result[key]
                for key in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "recommended_technical_score",
                    "scenario_metrics",
                )
            },
        },
        "performance": {
            **performance,
            "budgets": {
                "cold_start_seconds": 30.0,
                "warm_turn_p95_ms": 500.0,
                "peak_process_memory_mb": 4096.0,
                "model_asset_mb": 500.0,
            },
            "passed": performance_passed,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "controls": {
                    key: value["oof_metrics"] for key, value in controls.items()
                },
                "variants": {
                    key: value["oof_metrics"] for key, value in variants.items()
                },
                "selection": report["selection"],
                "performance": report["performance"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
