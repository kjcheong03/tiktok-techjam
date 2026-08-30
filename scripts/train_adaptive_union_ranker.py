from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.retrieval.gbdt import METADATA_FEATURES, GBDTFeatureStore, fit_lambdamart
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.training.adaptive_hybrid import (
    collect_adaptive_ranking_groups,
    collection_config,
    evaluate_group_ordering,
    ranking_dataset,
    sha256_file,
    train_adaptive_union_model,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/adaptive_hybrid_1a_3b_v1.json"
SPLIT_PATH = ROOT / "configs/splits/nested_v1.json"
CATALOG_PATH = ROOT / "data/catalog.jsonl"
MODEL_PATH = ROOT / "artifacts/models/adaptive_union_gbdt_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/adaptive_union_gbdt_v1.json"
RECEIPT_PATH = ROOT / "artifacts/models/adaptive_union_gbdt_v1.fit_receipt.json"
SEED = 20260830


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    config = load_adaptive_hybrid_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    adaptive_ids = {str(item) for item in split["adaptive_sample_ids"]}
    outer_folds = [{str(item) for item in fold} for fold in split["outer_folds"]]
    samples = {
        str(item["sample_id"]): item
        for item in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(item["sample_id"]) in adaptive_ids
    }
    _, categories, products = catalog_index(CATALOG_PATH)
    features = GBDTFeatureStore(CATALOG_PATH)
    groups, collection = collect_adaptive_ranking_groups(
        samples=samples,
        categories=categories,
        products=products,
        catalog_path=CATALOG_PATH,
        config=config,
        project_root=ROOT,
        features=features,
    )
    print(json.dumps(collection, sort_keys=True), flush=True)

    fold_results = []
    selected_rounds = []
    weighted_control_hit = weighted_candidate_hit = 0.0
    weighted_control_mrr = weighted_candidate_mrr = 0.0
    total_groups = 0
    for outer_index, outer_ids in enumerate(outer_folds):
        inner_validation = outer_folds[(outer_index + 1) % len(outer_folds)]
        inner_training = adaptive_ids - outer_ids - inner_validation
        model = train_adaptive_union_model(
            groups,
            inner_training,
            inner_validation,
            max_rounds=100,
            seed=SEED + outer_index,
        )
        control = evaluate_group_ordering(groups, outer_ids, None, overloaded=False)
        candidate = evaluate_group_ordering(
            groups, outer_ids, model, overloaded=False
        )
        group_count = int(control["groups"])
        total_groups += group_count
        weighted_control_hit += float(control["hit_rate_at_10"]) * group_count
        weighted_candidate_hit += float(candidate["hit_rate_at_10"]) * group_count
        weighted_control_mrr += float(control["mrr"]) * group_count
        weighted_candidate_mrr += float(candidate["mrr"]) * group_count
        selected_rounds.append(model.best_iteration)
        fold_results.append(
            {
                "outer_fold": outer_index,
                "training_ids": sorted(inner_training),
                "selection_ids": sorted(inner_validation),
                "validation_ids": sorted(outer_ids),
                "best_iteration": model.best_iteration,
                "control": control,
                "candidate": candidate,
            }
        )
        print(
            f"fold={outer_index} rounds={model.best_iteration} "
            f"mrr={float(candidate['mrr']):.6f}",
            flush=True,
        )

    oof_control = {
        "groups": total_groups,
        "hit_rate_at_10": weighted_control_hit / total_groups,
        "mrr": weighted_control_mrr / total_groups,
    }
    oof_candidate = {
        "groups": total_groups,
        "hit_rate_at_10": weighted_candidate_hit / total_groups,
        "mrr": weighted_candidate_mrr / total_groups,
    }
    selected = (
        oof_candidate["hit_rate_at_10"] >= oof_control["hit_rate_at_10"]
        and oof_candidate["mrr"] > oof_control["mrr"]
    )
    final_rounds = max(1, round(statistics.median(selected_rounds)))
    final_model = fit_lambdamart(
        *ranking_dataset(groups, adaptive_ids, overloaded=False),
        candidate_id="adaptive_union_gbdt_v1",
        feature_names=METADATA_FEATURES,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        max_rounds=final_rounds,
        early_stopping_rounds=12,
        validation=None,
        seed=SEED,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(MODEL_PATH)
    model_sha256 = sha256_file(MODEL_PATH)
    receipt = {
        "schema_version": 1,
        "asset_id": "adaptive_union_gbdt_v1",
        "training_ids": sorted(adaptive_ids),
        "outer_folds": [sorted(fold) for fold in outer_folds],
        "feature_schema": list(METADATA_FEATURES),
        "seed": SEED,
        "selected_rounds": selected_rounds,
        "final_rounds": final_rounds,
        "catalog_sha256": sha256_file(CATALOG_PATH),
        "candidate_generation_config_sha256": collection_config(
            config
        ).canonical_hash(),
        "training_invocation_config_sha256": config.canonical_hash(),
        "split_sha256": sha256_file(SPLIT_PATH),
        "model_sha256": model_sha256,
        "holdout_accessed": False,
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "experiment_id": "adaptive_union_gbdt_v1",
        "holdout_accessed": False,
        "collection": collection,
        "folds": fold_results,
        "oof_control": oof_control,
        "oof_candidate": oof_candidate,
        "oof_delta": {
            "hit_rate_at_10": (
                oof_candidate["hit_rate_at_10"] - oof_control["hit_rate_at_10"]
            ),
            "mrr": oof_candidate["mrr"] - oof_control["mrr"],
        },
        "selected_for_adaptive_config": selected,
        "selection_rule": (
            "OOF Hit@10 non-regression and strict OOF MRR improvement on exact "
            "merged candidate pools"
        ),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": model_sha256,
        "fit_receipt_path": str(RECEIPT_PATH.relative_to(ROOT)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "oof_control": oof_control,
                "oof_candidate": oof_candidate,
                "model_sha256": model_sha256,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
