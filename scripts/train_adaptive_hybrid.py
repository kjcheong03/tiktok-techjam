from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index
from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.retrieval.gbdt import GBDTFeatureStore, fit_lambdamart
from ghostlab.retrieval.union_features import UNION_FEATURES
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.training.adaptive_datasets import (
    AdaptiveTrainingCorpus,
    fold_manifest,
    load_adaptive_training_corpus,
)
from ghostlab.training.adaptive_hybrid import (
    AdaptiveRankingGroup,
    collect_adaptive_ranking_groups,
    collection_config,
    evaluate_group_ordering,
    ranking_dataset,
    sha256_file,
)
from ghostlab.training.adaptive_lineage import (
    load_lineage_manifest,
    manifest_outer_folds,
    subset_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


@dataclass(frozen=True)
class ModelTrainingResult:
    model: Any
    selected: bool
    eligible_sample_ids: frozenset[str]
    folds: tuple[dict[str, object], ...]
    selected_rounds: tuple[int, ...]
    control: dict[str, float | int]
    candidate: dict[str, float | int]
    grouped: dict[str, dict[str, dict[str, float | int]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the adaptive union ranker with lineage-safe nested folds and "
            "emit a hash-bound config. The overload safe ranker is intentionally "
            "deterministic so cutoff turns do not invoke a fitted full ranker."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="repeat for every training JSONL; defaults to the 200+1000+1000 set",
    )
    parser.add_argument("--config", default="configs/adaptive_hybrid_1a_3b_v1.json")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--policy-id", default="adaptive_hybrid_1a_3b_1650_final_v1")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate datasets, folds and output paths without collecting or fitting",
    )
    parser.add_argument(
        "--split-output", default="configs/splits/adaptive_1650_group_nested_v1.json"
    )
    parser.add_argument(
        "--union-model-output",
        default="artifacts/models/adaptive_union_gbdt_1650_final_v1.json",
    )
    parser.add_argument(
        "--union-receipt-output",
        default="artifacts/models/adaptive_union_gbdt_1650_final_v1.fit_receipt.json",
    )
    parser.add_argument(
        "--output-config",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1.json",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/reports/adaptive_hybrid_training_1650_final_v1.json",
    )
    return parser


def _input_path(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _output_path(raw: str) -> Path:
    path = Path(raw)
    resolved = path.absolute() if path.is_absolute() else (ROOT / path).absolute()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"output must stay inside project root: {raw}") from error
    return resolved


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ids_sha256(sample_ids: set[str]) -> str:
    encoded = "\n".join(sorted(sample_ids)).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fit_ranker(
    groups: dict[str, list[AdaptiveRankingGroup]],
    sample_ids: set[str],
    *,
    validation_ids: set[str] | None,
    candidate_id: str,
    route: str | None,
    overloaded: bool,
    max_rounds: int,
    seed: int,
):
    training = ranking_dataset(groups, sample_ids, route=route, overloaded=overloaded)
    validation = (
        ranking_dataset(groups, validation_ids, route=route, overloaded=overloaded)
        if validation_ids is not None
        else None
    )
    return fit_lambdamart(
        *training,
        candidate_id=candidate_id,
        feature_names=UNION_FEATURES,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        max_rounds=max_rounds,
        early_stopping_rounds=12,
        validation=validation,
        seed=seed,
    )


def _weighted_metrics(
    pairs: list[tuple[dict[str, float | int], dict[str, float | int]]],
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    total = sum(int(control["groups"]) for control, _ in pairs)
    if total == 0:
        raise ValueError("cross-validation produced no ranking groups")

    def aggregate(index: int) -> dict[str, float | int]:
        selected = [pair[index] for pair in pairs]
        return {
            "groups": total,
            "hit_rate_at_10": sum(
                float(item["hit_rate_at_10"]) * int(item["groups"]) for item in selected
            )
            / total,
            "mrr": sum(float(item["mrr"]) * int(item["groups"]) for item in selected)
            / total,
        }

    return aggregate(0), aggregate(1)


def _cross_validate_and_fit(
    groups: dict[str, list[AdaptiveRankingGroup]],
    folds: tuple[tuple[str, ...], ...],
    all_ids: set[str],
    *,
    candidate_id: str,
    route: str | None,
    overloaded: bool,
    max_rounds: int,
    seed: int,
) -> ModelTrainingResult:
    eligible_ids = frozenset(
        sample_id
        for sample_id, items in groups.items()
        if any(
            (route is None or item.route == route) and item.overloaded is overloaded
            for item in items
        )
    )
    if not eligible_ids:
        raise ValueError(f"no eligible ranking sessions for {candidate_id}")
    fold_records: list[dict[str, object]] = []
    metric_pairs: list[tuple[dict[str, float | int], dict[str, float | int]]] = []
    selected_rounds: list[int] = []
    grouped_pairs: dict[
        tuple[str, str],
        list[tuple[dict[str, float | int], dict[str, float | int]]],
    ] = {}
    for outer_index, outer_fold in enumerate(folds):
        validation_ids = set(outer_fold)
        selection_ids = set(folds[(outer_index + 1) % len(folds)])
        training_ids = all_ids - validation_ids - selection_ids
        model = _fit_ranker(
            groups,
            training_ids,
            validation_ids=selection_ids,
            candidate_id=candidate_id,
            route=route,
            overloaded=overloaded,
            max_rounds=max_rounds,
            seed=seed + outer_index,
        )
        control = evaluate_group_ordering(
            groups,
            validation_ids,
            None,
            route=route,
            overloaded=overloaded,
        )
        candidate = evaluate_group_ordering(
            groups,
            validation_ids,
            model,
            route=route,
            overloaded=overloaded,
        )
        metric_pairs.append((control, candidate))
        outer_groups = [
            group
            for sample_id in validation_ids
            for group in groups.get(sample_id, ())
            if (route is None or group.route == route)
            and group.overloaded is overloaded
        ]
        dimensions = {
            "route": sorted({group.route for group in outer_groups}),
            "source": sorted({group.source for group in outer_groups}),
            "scenario": sorted({group.scenario_type for group in outer_groups}),
        }
        for dimension, values in dimensions.items():
            for value in values:
                filters: dict[str, str] = {}
                if dimension == "route":
                    filters["route"] = value
                elif dimension == "source":
                    filters["source"] = value
                else:
                    filters["scenario_type"] = value
                grouped_control = evaluate_group_ordering(
                    groups,
                    validation_ids,
                    None,
                    overloaded=overloaded,
                    **filters,
                )
                grouped_candidate = evaluate_group_ordering(
                    groups,
                    validation_ids,
                    model,
                    overloaded=overloaded,
                    **filters,
                )
                grouped_pairs.setdefault((dimension, value), []).append(
                    (grouped_control, grouped_candidate)
                )
        selected_rounds.append(model.best_iteration)
        fold_records.append(
            {
                "outer_fold": outer_index,
                "training_sample_count": len(training_ids),
                "selection_sample_count": len(selection_ids),
                "validation_sample_count": len(validation_ids),
                "training_sample_ids_sha256": _ids_sha256(training_ids),
                "selection_sample_ids_sha256": _ids_sha256(selection_ids),
                "validation_sample_ids_sha256": _ids_sha256(validation_ids),
                "best_iteration": model.best_iteration,
                "control": control,
                "candidate": candidate,
            }
        )
        print(
            f"model={candidate_id} fold={outer_index} "
            f"groups={candidate['groups']} rounds={model.best_iteration} "
            f"mrr={float(candidate['mrr']):.6f}",
            flush=True,
        )
    control, candidate = _weighted_metrics(metric_pairs)
    grouped: dict[str, dict[str, dict[str, float | int]]] = {}
    grouped_regressions: list[str] = []
    for (dimension, value), pairs in sorted(grouped_pairs.items()):
        group_control, group_candidate = _weighted_metrics(pairs)
        grouped.setdefault(dimension, {})[value] = {
            "groups": int(group_control["groups"]),
            "control_hit_rate_at_10": float(group_control["hit_rate_at_10"]),
            "candidate_hit_rate_at_10": float(group_candidate["hit_rate_at_10"]),
            "delta_hit_rate_at_10": float(group_candidate["hit_rate_at_10"])
            - float(group_control["hit_rate_at_10"]),
            "control_mrr": float(group_control["mrr"]),
            "candidate_mrr": float(group_candidate["mrr"]),
            "delta_mrr": float(group_candidate["mrr"]) - float(group_control["mrr"]),
        }
        if int(group_control["groups"]) >= 10 and (
            float(group_candidate["hit_rate_at_10"])
            < float(group_control["hit_rate_at_10"]) - 0.02
            or float(group_candidate["mrr"]) < float(group_control["mrr"]) - 0.02
        ):
            grouped_regressions.append(f"{dimension}:{value}")
    selected = (
        float(candidate["hit_rate_at_10"]) >= float(control["hit_rate_at_10"])
        and float(candidate["mrr"]) > float(control["mrr"])
        and not grouped_regressions
    )
    final_rounds = max(1, round(statistics.median(selected_rounds)))
    final_model = _fit_ranker(
        groups,
        all_ids,
        validation_ids=None,
        candidate_id=candidate_id,
        route=route,
        overloaded=overloaded,
        max_rounds=final_rounds,
        seed=seed,
    )
    return ModelTrainingResult(
        model=final_model,
        selected=selected,
        eligible_sample_ids=eligible_ids,
        folds=tuple(fold_records),
        selected_rounds=tuple(selected_rounds),
        control=control,
        candidate=candidate,
        grouped=grouped,
    )


def _receipt(
    *,
    asset_id: str,
    model_hash: str,
    model_path: Path,
    result: ModelTrainingResult,
    corpus: AdaptiveTrainingCorpus,
    split_path: Path,
    catalog_path: Path,
    baseline: AdaptiveHybridConfig,
    collection_hash: str,
    deployment_hash: str,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "asset_id": asset_id,
        "model_path": _relative(model_path),
        "model_sha256": model_hash,
        "replayed_corpus_sample_count": len(corpus.samples),
        "eligible_ranking_sample_count": len(result.eligible_sample_ids),
        "eligible_ranking_sample_ids_sha256": _ids_sha256(
            set(result.eligible_sample_ids)
        ),
        "dataset_sources": [source.__dict__ for source in corpus.sources],
        "feature_schema": list(UNION_FEATURES),
        "seed": seed,
        "selected_rounds": list(result.selected_rounds),
        "final_rounds": result.model.best_iteration,
        "split_manifest_path": _relative(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "catalog_sha256": sha256_file(catalog_path),
        "candidate_generation_config_sha256": collection_hash,
        "training_invocation_config_sha256": baseline.canonical_hash(),
        "deployment_config_sha256": deployment_hash,
        "selected_by_oof": result.selected,
        "holdout_accessed": False,
        "independent_template_consumed_for_training": any(
            "independent_template" in source.path for source in corpus.sources
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.max_rounds <= 0:
        raise ValueError("max rounds must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.perf_counter()
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    complete_corpus = load_adaptive_training_corpus(ROOT, datasets)
    lineage_manifest_path = _input_path(args.lineage_manifest)
    lineage_manifest = load_lineage_manifest(lineage_manifest_path, complete_corpus)
    corpus = subset_corpus(complete_corpus, lineage_manifest, "development")
    folds = manifest_outer_folds(lineage_manifest)
    if len(folds) != args.folds:
        raise ValueError("requested fold count does not match the lineage manifest")
    if len(corpus.samples) != 1650 and args.datasets is None:
        raise ValueError(
            "default adaptive development corpus must contain 1650 samples, "
            f"got {len(corpus.samples)}"
        )

    config_path = _input_path(args.config)
    catalog_path = _input_path(args.catalog)
    split_path = _output_path(args.split_output)
    union_model_path = _output_path(args.union_model_output)
    union_receipt_path = _output_path(args.union_receipt_output)
    output_config_path = _output_path(args.output_config)
    report_path = _output_path(args.report_output)

    baseline = load_adaptive_hybrid_config(config_path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"missing catalog: {catalog_path}")
    AdaptiveArchitectureAudit.validate(baseline)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "sample_count": len(corpus.samples),
                    "dataset_sources": [source.__dict__ for source in corpus.sources],
                    "fold_count": len(folds),
                    "fold_sample_counts": [len(fold) for fold in folds],
                    "output_config": _relative(output_config_path),
                    "union_model": _relative(union_model_path),
                    "overload_safe_ranker": "deterministic_bounded_scorer",
                    "independent_template_consumed_for_training": any(
                        "independent_template" in source.path
                        for source in corpus.sources
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    split_payload = fold_manifest(corpus, folds, seed=args.seed)
    split_payload.update(
        {
            "partition": "development",
            "lineage_manifest_path": _relative(lineage_manifest_path),
            "lineage_manifest_sha256": sha256_file(lineage_manifest_path),
            "group_safe": True,
        }
    )
    _write_json(split_path, split_payload)
    _, categories, products = catalog_index(catalog_path)
    groups, collection = collect_adaptive_ranking_groups(
        samples=corpus.samples,
        origins=corpus.origins,
        categories=categories,
        products=products,
        catalog_path=catalog_path,
        config=baseline,
        project_root=ROOT,
        features=GBDTFeatureStore(catalog_path),
        disable_browsing_safe=True,
    )
    print(json.dumps(collection, sort_keys=True), flush=True)

    union = _cross_validate_and_fit(
        groups,
        folds,
        corpus.sample_ids,
        candidate_id="adaptive_union_gbdt_1650_final_v1",
        route=None,
        overloaded=False,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    union_model_path.parent.mkdir(parents=True, exist_ok=True)
    union.model.save(union_model_path)
    union_hash = sha256_file(union_model_path)

    union_config = baseline.union_ranker
    if union.selected:
        union_config = union_config.model_copy(
            update={
                "backend": "gbdt",
                "model_path": _relative(union_model_path),
                "model_sha256": union_hash,
            }
        )
    # A genuine overload cutoff must not invoke the normal learned union path.
    # Keep its bounded safe scorer deterministic and separately observable.
    browsing_config = baseline.browsing.model_copy(
        update={
            "safe_ranker_backend": "deterministic",
            "safe_ranker_model_path": None,
            "safe_ranker_model_sha256": None,
        }
    )
    deployment = baseline.model_copy(
        update={
            "policy_id": args.policy_id,
            "union_ranker": union_config,
            "browsing": browsing_config,
        }
    )
    AdaptiveArchitectureAudit.validate(deployment)
    _write_json(output_config_path, deployment.model_dump(mode="json"))
    deployment_hash = deployment.canonical_hash()

    collection_hash = collection_config(
        baseline, disable_browsing_safe=True
    ).canonical_hash()
    union_receipt = _receipt(
        asset_id="adaptive_union_gbdt_1650_final_v1",
        model_hash=union_hash,
        model_path=union_model_path,
        result=union,
        corpus=corpus,
        split_path=split_path,
        catalog_path=catalog_path,
        baseline=baseline,
        collection_hash=collection_hash,
        deployment_hash=deployment_hash,
        seed=args.seed,
    )
    _write_json(union_receipt_path, union_receipt)

    def result_payload(result: ModelTrainingResult) -> dict[str, object]:
        return {
            "selected_for_output_config": result.selected,
            "replayed_corpus_sample_count": len(corpus.samples),
            "eligible_ranking_sample_count": len(result.eligible_sample_ids),
            "selected_rounds": list(result.selected_rounds),
            "oof_control": result.control,
            "oof_candidate": result.candidate,
            "oof_delta": {
                "hit_rate_at_10": float(result.candidate["hit_rate_at_10"])
                - float(result.control["hit_rate_at_10"]),
                "mrr": float(result.candidate["mrr"]) - float(result.control["mrr"]),
            },
            "folds": list(result.folds),
            "grouped_oof": result.grouped,
        }

    report = {
        "schema_version": 1,
        "experiment_id": args.policy_id,
        "sample_count": len(corpus.samples),
        "dataset_sources": [source.__dict__ for source in corpus.sources],
        "fold_count": len(folds),
        "collection": collection,
        "union": result_payload(union),
        "overload_safe_ranker": {
            "backend": "deterministic",
            "trained_asset": False,
            "normal_union_bypassed_on_overload": True,
        },
        "profile_awareness": {
            "runtime_profile_context_replayed": True,
            "conflict_safe_runtime_profile_stage_preserved": True,
            "profile_feature_present_in_union_schema": "profile_term_match"
            in UNION_FEATURES,
            "profile_channels_runtime_gated": True,
        },
        "output_config": _relative(output_config_path),
        "output_config_sha256": deployment_hash,
        "union_model": _relative(union_model_path),
        "union_model_sha256": union_hash,
        "holdout_accessed": False,
        "lineage_manifest": _relative(lineage_manifest_path),
        "lineage_manifest_sha256": sha256_file(lineage_manifest_path),
        "partition": "development",
        "group_safe_outer_folds": True,
        "independent_template_consumed_for_training": any(
            "independent_template" in source.path for source in corpus.sources
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
