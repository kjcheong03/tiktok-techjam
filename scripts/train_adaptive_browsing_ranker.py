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
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/adaptive_hybrid_1a_3b_v1.json"
SPLIT_PATH = ROOT / "configs/splits/nested_v1.json"
CATALOG_PATH = ROOT / "data/catalog.jsonl"
MODEL_PATH = ROOT / "artifacts/models/adaptive_browsing_gbdt_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/adaptive_browsing_gbdt_v1.json"
RECEIPT_PATH = ROOT / "artifacts/models/adaptive_browsing_gbdt_v1.fit_receipt.json"
SEED = 20260831


def _dataset(groups: dict, sample_ids: set[str]):
    return ranking_dataset(
        groups, sample_ids, route="browsing", overloaded=True
    )


def _fit(
    groups: dict,
    training_ids: set[str],
    validation_ids: set[str],
    *,
    max_rounds: int,
    seed: int,
):
    return fit_lambdamart(
        *_dataset(groups, training_ids),
        candidate_id="adaptive_browsing_gbdt_v1",
        feature_names=METADATA_FEATURES,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        max_rounds=max_rounds,
        early_stopping_rounds=12,
        validation=_dataset(groups, validation_ids),
        seed=seed,
    )


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
    groups, collection = collect_adaptive_ranking_groups(
        samples=samples,
        categories=categories,
        products=products,
        catalog_path=CATALOG_PATH,
        config=config,
        project_root=ROOT,
        features=GBDTFeatureStore(CATALOG_PATH),
        disable_browsing_safe=True,
    )
    browsing_groups = sum(
        group.route == "browsing" and group.overloaded
        for items in groups.values()
        for group in items
    )
    if browsing_groups == 0:
        raise ValueError("no target-containing overloaded Browsing pools collected")
    print(
        json.dumps({**collection, "eligible_browsing_groups": browsing_groups}),
        flush=True,
    )

    folds = []
    selected_rounds = []
    total = 0
    control_hit = candidate_hit = control_mrr = candidate_mrr = 0.0
    for outer_index, outer_ids in enumerate(outer_folds):
        selection_ids = outer_folds[(outer_index + 1) % len(outer_folds)]
        training_ids = adaptive_ids - outer_ids - selection_ids
        model = _fit(
            groups,
            training_ids,
            selection_ids,
            max_rounds=100,
            seed=SEED + outer_index,
        )
        control = evaluate_group_ordering(
            groups,
            outer_ids,
            None,
            route="browsing",
            overloaded=True,
        )
        candidate = evaluate_group_ordering(
            groups,
            outer_ids,
            model,
            route="browsing",
            overloaded=True,
        )
        count = int(control["groups"])
        total += count
        control_hit += float(control["hit_rate_at_10"]) * count
        candidate_hit += float(candidate["hit_rate_at_10"]) * count
        control_mrr += float(control["mrr"]) * count
        candidate_mrr += float(candidate["mrr"]) * count
        selected_rounds.append(model.best_iteration)
        folds.append(
            {
                "outer_fold": outer_index,
                "best_iteration": model.best_iteration,
                "control": control,
                "candidate": candidate,
            }
        )
        print(
            f"fold={outer_index} groups={count} rounds={model.best_iteration} "
            f"mrr={float(candidate['mrr']):.6f}",
            flush=True,
        )

    oof_control = {
        "groups": total,
        "hit_rate_at_10": control_hit / total,
        "mrr": control_mrr / total,
    }
    oof_candidate = {
        "groups": total,
        "hit_rate_at_10": candidate_hit / total,
        "mrr": candidate_mrr / total,
    }
    selected = (
        oof_candidate["hit_rate_at_10"] >= oof_control["hit_rate_at_10"]
        and oof_candidate["mrr"] > oof_control["mrr"]
    )
    final_rounds = max(1, round(statistics.median(selected_rounds)))
    final_model = fit_lambdamart(
        *_dataset(groups, adaptive_ids),
        candidate_id="adaptive_browsing_gbdt_v1",
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
    model_hash = sha256_file(MODEL_PATH)
    receipt = {
        "schema_version": 1,
        "asset_id": "adaptive_browsing_gbdt_v1",
        "training_ids": sorted(adaptive_ids),
        "feature_schema": list(METADATA_FEATURES),
        "seed": SEED,
        "selected_rounds": selected_rounds,
        "final_rounds": final_rounds,
        "catalog_sha256": sha256_file(CATALOG_PATH),
        "candidate_generation_config_sha256": collection_config(
            config, disable_browsing_safe=True
        ).canonical_hash(),
        "training_invocation_config_sha256": config.canonical_hash(),
        "split_sha256": sha256_file(SPLIT_PATH),
        "model_sha256": model_hash,
        "holdout_accessed": False,
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "experiment_id": "adaptive_browsing_gbdt_v1",
        "holdout_accessed": False,
        "collection": collection,
        "folds": folds,
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
            "overloaded Browsing dense pools"
        ),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": model_hash,
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
                "model_sha256": model_hash,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
