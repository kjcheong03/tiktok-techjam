from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.retrieval.ensemble import (
    ModelRankEnsembleReranker,
    RankEnsembleAsset,
    fit_rank_stack_weights,
)
from ghostlab.retrieval.gbdt import (
    FEATURE_SETS,
    METADATA_FEATURES,
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
    fit_lambdamart,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.reward_lambdamart import fit_reward_lambdamart
from ghostlab.retrieval.sparse import SparseIndex
from scripts.run_gbdt_reranker import (
    RankingGroup,
    build_agent,
    collect_groups,
    paired_evidence,
    stability,
    summarized_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/w2_reward_lambdamart_v1.json"
ENSEMBLE_MANIFEST_PATH = ROOT / "configs/experiments/w2_rank_ensemble_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/w2_ranking_v1.json"
MODEL_DIR = ROOT / "artifacts/models/w2_ranking_v1"
SEED = 20260826


def ranking_dataset_with_turns(
    groups: dict[str, list[RankingGroup]],
    sample_ids: set[str],
    feature_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[int], list[int]]:
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
        [group.turn for group in selected],
    )


def train_objective(
    groups: dict[str, list[RankingGroup]],
    training_ids: set[str],
    validation_ids: set[str],
    objective_config: dict,
    capacity: dict,
) -> tuple[LambdaMARTModel, LambdaMARTModel]:
    names = FEATURE_SETS["metadata"]
    training = ranking_dataset_with_turns(groups, training_ids, names)
    validation = ranking_dataset_with_turns(groups, validation_ids, names)
    common = {
        "candidate_id": str(objective_config["candidate_id"]),
        "feature_names": names,
        "max_depth": int(capacity["max_depth"]),
        "num_leaves": int(capacity["num_leaves"]),
        "learning_rate": float(capacity["learning_rate"]),
        "max_rounds": int(capacity["max_rounds"]),
        "early_stopping_rounds": int(capacity["early_stopping_rounds"]),
        "seed": int(capacity["seed"]),
    }
    objective = str(objective_config["objective"])
    if objective == "ndcg_at_10":
        selected = fit_lambdamart(
            training[0],
            training[1],
            training[2],
            validation=(validation[0], validation[1], validation[2]),
            **common,
        )
        refit_data = ranking_dataset_with_turns(
            groups, training_ids | validation_ids, names
        )
        refit = fit_lambdamart(
            refit_data[0],
            refit_data[1],
            refit_data[2],
            validation=None,
            **{**common, "max_rounds": selected.best_iteration},
        )
        return selected, refit
    reward_common = {
        **common,
        "objective": objective,
        "min_samples_leaf": int(capacity["min_samples_leaf"]),
        "l2_leaf": float(capacity["l2_leaf"]),
    }
    selected = fit_reward_lambdamart(
        *training,
        validation=validation,
        **reward_common,
    )
    refit_data = ranking_dataset_with_turns(
        groups, training_ids | validation_ids, names
    )
    refit = fit_reward_lambdamart(
        *refit_data,
        validation=None,
        **{**reward_common, "max_rounds": selected.best_iteration},
    )
    return selected, refit


def model_scores(
    models: list[LambdaMARTModel],
    dataset: tuple[NDArray[np.float64], NDArray[np.int64], list[int], list[int]],
) -> NDArray[np.float64]:
    return np.vstack([model.predict(dataset[0]) for model in models])


def target_rank_movement(candidate: list[dict], control: list[dict]) -> dict[str, object]:
    control_by_id = {str(item["sample_id"]): item for item in control}
    crossings = rank_one_gains = rank_one_losses = improved = worsened = 0
    reciprocal_deltas: list[float] = []
    for item in candidate:
        base = control_by_id[str(item["sample_id"])]
        candidate_hit = bool(item["hit"])
        base_hit = bool(base["hit"])
        crossings += int(candidate_hit and not base_hit)
        rank_one_gains += int(item["best_rank"] == 1 and base["best_rank"] != 1)
        rank_one_losses += int(item["best_rank"] != 1 and base["best_rank"] == 1)
        delta = float(item["reciprocal_rank"]) - float(base["reciprocal_rank"])
        reciprocal_deltas.append(delta)
        improved += int(delta > 1e-12)
        worsened += int(delta < -1e-12)
    return {
        "new_session_hits": crossings,
        "new_rank_1_sessions": rank_one_gains,
        "lost_rank_1_sessions": rank_one_losses,
        "reciprocal_rank_improvements": improved,
        "reciprocal_rank_regressions": worsened,
        "mean_reciprocal_rank_delta": round(statistics.fmean(reciprocal_deltas), 6),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_diagnostic(
    rerankers: dict[str, object],
    groups: dict[str, list[RankingGroup]],
) -> dict[str, dict[str, float]]:
    queries = [
        group
        for sample_id in sorted(groups)
        for group in groups[sample_id]
    ][:100]
    result: dict[str, dict[str, float]] = {}
    for name, reranker in rerankers.items():
        started = time.perf_counter()
        for group in queries:
            reranker.rerank(group.query, list(group.candidates), rerank_k=50)
        elapsed = time.perf_counter() - started
        result[name] = {
            "queries": len(queries),
            "total_seconds": round(elapsed, 6),
            "milliseconds_per_query": round(1000.0 * elapsed / len(queries), 6),
        }
    return result


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ensemble_manifest = json.loads(ENSEMBLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False or manifest[
        "protected_holdout_access"
    ] != "forbidden":
        raise RuntimeError("ranking campaign does not preserve the F3 firewall")
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    features = GBDTFeatureStore(catalog_path, quality=quality.quality)
    groups, collection = collect_groups(
        samples,
        categories,
        products,
        SparseIndex(catalog_path),
        quality,
        features,
    )
    print(json.dumps(collection, sort_keys=True), flush=True)

    objective_configs = list(manifest["objectives"])
    capacity = dict(manifest["fixed_capacity"])
    ensemble_heads = list(ensemble_manifest["heads"])
    oof_sessions: dict[str, list[dict]] = defaultdict(list)
    fold_records: dict[str, list[dict]] = defaultdict(list)
    selected_weight_records: list[dict] = []

    for outer_index, outer_ids in enumerate(outer_folds):
        inner_index = (outer_index + 1) % len(outer_folds)
        inner_validation_ids = outer_folds[inner_index]
        inner_training_ids = adaptive_ids - outer_ids - inner_validation_ids
        selected_models: dict[str, LambdaMARTModel] = {}
        refit_models: dict[str, LambdaMARTModel] = {}
        fold_samples = [samples[sample_id] for sample_id in sorted(outer_ids)]
        for config in objective_configs:
            candidate_id = str(config["candidate_id"])
            selected, refit = train_objective(
                groups, inner_training_ids, inner_validation_ids, config, capacity
            )
            selected_models[candidate_id] = selected
            refit_models[candidate_id] = refit
            result = evaluate(
                build_agent(quality, LambdaMARTReranker(features, refit)),
                fold_samples,
                catalog_ids,
                categories,
                products,
            )
            oof_sessions[candidate_id].extend(result["sessions"])
            fold_records[candidate_id].append(
                {
                    "outer_fold": outer_index,
                    "inner_training_samples": len(inner_training_ids),
                    "inner_validation_samples": len(inner_validation_ids),
                    "outer_validation_samples": len(outer_ids),
                    "inner_selected_rounds": selected.best_iteration,
                    "outer_metrics": summarized_metrics(result["sessions"]),
                }
            )
            print(
                f"{candidate_id} fold={outer_index} rounds={selected.best_iteration} "
                f"score={result['recommended_technical_score']}",
                flush=True,
            )

        head_selected = [selected_models[name] for name in ensemble_heads]
        head_refit = tuple(refit_models[name] for name in ensemble_heads)
        validation_data = ranking_dataset_with_turns(
            groups, inner_validation_ids, FEATURE_SETS["metadata"]
        )
        stack = fit_rank_stack_weights(
            model_scores(head_selected, validation_data),
            validation_data[1],
            validation_data[2],
            validation_data[3],
            grid_step=float(ensemble_manifest["fold_local_candidate"]["grid_step"]),
        )
        selected_weight_records.append(
            {
                "outer_fold": outer_index,
                "weights": list(stack.weights),
                "inner_validation_reward": round(stack.inner_validation_reward, 6),
            }
        )
        ensemble_candidates = {
            "equal_standardized_scores": ModelRankEnsembleReranker(
                features, head_refit, method="standardized_score"
            ),
            "equal_mean_rank": ModelRankEnsembleReranker(
                features, head_refit, method="mean_rank"
            ),
            "fold_local_rank_stack": ModelRankEnsembleReranker(
                features,
                head_refit,
                method="standardized_score",
                weights=stack.weights,
            ),
        }
        for candidate_id, reranker in ensemble_candidates.items():
            result = evaluate(
                build_agent(quality, reranker),
                fold_samples,
                catalog_ids,
                categories,
                products,
            )
            oof_sessions[candidate_id].extend(result["sessions"])
            fold_records[candidate_id].append(
                {
                    "outer_fold": outer_index,
                    "outer_validation_samples": len(outer_ids),
                    "weights": (
                        list(stack.weights)
                        if candidate_id == "fold_local_rank_stack"
                        else [1.0 / len(head_refit)] * len(head_refit)
                    ),
                    "outer_metrics": summarized_metrics(result["sessions"]),
                }
            )
            print(
                f"{candidate_id} fold={outer_index} "
                f"score={result['recommended_technical_score']}",
                flush=True,
            )

    control_id = "ndcg_at_10_control"
    control_sessions = oof_sessions[control_id]
    candidates: dict[str, dict] = {}
    for candidate_id, sessions in sorted(oof_sessions.items()):
        candidates[candidate_id] = {
            "evidence_label": "stitched outer-fold OOF",
            "oof_metrics": summarized_metrics(sessions),
            "folds": fold_records[candidate_id],
            "stability": stability(fold_records[candidate_id]),
            "paired_vs_ndcg_control": (
                None
                if candidate_id == control_id
                else paired_evidence(sessions, control_sessions)
            ),
            "target_rank_movement_vs_ndcg": (
                None
                if candidate_id == control_id
                else target_rank_movement(sessions, control_sessions)
            ),
        }

    ranked = sorted(
        candidates,
        key=lambda name: (
            -float(candidates[name]["oof_metrics"]["recommended_technical_score"]),
            name,
        ),
    )
    best_id = ranked[0]

    # Produce compact development-refit assets without consulting F3.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    deploy_training = adaptive_ids - outer_folds[0]
    deploy_validation = outer_folds[0]
    final_models: dict[str, LambdaMARTModel] = {}
    asset_records: dict[str, dict[str, object]] = {}
    for config in objective_configs:
        candidate_id = str(config["candidate_id"])
        selected, model = train_objective(
            groups, deploy_training, deploy_validation, config, capacity
        )
        path = MODEL_DIR / f"{candidate_id}.json"
        model.save(path)
        final_models[candidate_id] = model
        asset_records[candidate_id] = {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "selected_rounds": selected.best_iteration,
            "objective": str(config["objective"]),
            "technique_id": str(config["technique_id"]),
        }

    deploy_validation_data = ranking_dataset_with_turns(
        groups, deploy_validation, FEATURE_SETS["metadata"]
    )
    deploy_selected_models = []
    for name in ensemble_heads:
        config = next(
            item for item in objective_configs if item["candidate_id"] == name
        )
        selected, _ = train_objective(
            groups, deploy_training, deploy_validation, config, capacity
        )
        deploy_selected_models.append(selected)
    deploy_stack = fit_rank_stack_weights(
        model_scores(deploy_selected_models, deploy_validation_data),
        deploy_validation_data[1],
        deploy_validation_data[2],
        deploy_validation_data[3],
        grid_step=float(ensemble_manifest["fold_local_candidate"]["grid_step"]),
    )
    relative_models = tuple(
        str((MODEL_DIR / f"{name}.json").relative_to(ROOT))
        for name in ensemble_heads
    )
    equal_asset = RankEnsembleAsset(
        technique_id="ranking.fold_ensemble.v1",
        aggregation="standardized_score",
        model_assets=relative_models,
        weights=(1.0 / len(ensemble_heads),) * len(ensemble_heads),
    )
    stack_asset = RankEnsembleAsset(
        technique_id="fusion.rank_stack.v1",
        aggregation="standardized_score",
        model_assets=relative_models,
        weights=deploy_stack.weights,
    )
    equal_path = MODEL_DIR / "fold_ensemble.json"
    stack_path = MODEL_DIR / "rank_stack.json"
    equal_asset.save(equal_path)
    stack_asset.save(stack_path)
    asset_records["fold_ensemble"] = {
        "relative_path": str(equal_path.relative_to(ROOT)),
        "sha256": sha256_file(equal_path),
        "bytes": equal_path.stat().st_size,
    }
    asset_records["rank_stack"] = {
        "relative_path": str(stack_path.relative_to(ROOT)),
        "sha256": sha256_file(stack_path),
        "bytes": stack_path.stat().st_size,
        "weights": list(deploy_stack.weights),
    }

    runtime = runtime_diagnostic(
        {
            "ndcg_single": LambdaMARTReranker(features, final_models[control_id]),
            "reward_single": LambdaMARTReranker(
                features, final_models["reward_lambdamart_v1"]
            ),
            "fold_ensemble": ModelRankEnsembleReranker.from_asset(
                features, equal_asset, project_root=ROOT
            ),
            "rank_stack": ModelRankEnsembleReranker.from_asset(
                features, stack_asset, project_root=ROOT
            ),
        },
        groups,
    )

    report = {
        "schema_version": 1,
        "experiment_id": "w2_ranking_v1",
        "evidence_label": "nested grouped outer-fold OOF on adaptive 150 only",
        "protected_holdout_accessed": False,
        "manifests": {
            "reward": str(MANIFEST_PATH.relative_to(ROOT)),
            "ensemble": str(ENSEMBLE_MANIFEST_PATH.relative_to(ROOT)),
        },
        "collection": collection,
        "matched_contract": {
            "feature_set": "metadata",
            "candidate_depth": int(manifest["rerank_k"]),
            "capacity": capacity,
            "same_candidate_set": True,
            "runtime_features_observable": True,
            "targets_turn_outcomes_training_labels_only": True,
        },
        "candidates": candidates,
        "ranking": ranked,
        "oof_leader": {
            "candidate_id": best_id,
            "metrics": candidates[best_id]["oof_metrics"],
            "promotion_status": "research_only_not_promoted",
        },
        "fold_local_stack_weights": selected_weight_records,
        "development_refit_assets": asset_records,
        "runtime": runtime,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "caveats": [
            "The matched ranking subsystem uses the frozen Wave 1 raw-history/fixed-question trajectory, not the complete 0.878963 guarded policy.",
            "The all-development assets are deployment artifacts, not OOF evidence.",
            "No F3/protected-50 data was imported, enumerated, read, or scored.",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["oof_leader"], sort_keys=True), flush=True)
    print(f"report={REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
