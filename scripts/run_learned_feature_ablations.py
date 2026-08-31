from __future__ import annotations

import json
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.retrieval.learned import (
    FEATURE_NAMES,
    CandidateFeatureStore,
    fit_pairwise_linear,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from scripts.run_learned_reranker import (
    agent,
    collect_examples,
    summarized_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SETS = {
    "full": FEATURE_NAMES,
    "stable_core": (
        "title_overlap",
        "category_overlap",
        "feature_overlap",
        "catalog_quality",
    ),
    "feature_quality": ("feature_overlap", "catalog_quality"),
    "feature_only": ("feature_overlap",),
    "quality_only": ("catalog_quality",),
}


def main() -> None:
    started = time.perf_counter()
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    feature_store = CandidateFeatureStore(ROOT / "data/catalog.jsonl")
    examples, collection = collect_examples(
        samples,
        categories,
        products,
        SparseIndex(ROOT / "data/catalog.jsonl"),
        CatalogQualityReranker(ROOT / "data/catalog.jsonl"),
        feature_store,
    )
    print(json.dumps(collection, sort_keys=True), flush=True)
    oof_sessions: dict[str, list[dict]] = {name: [] for name in FEATURE_SETS}
    folds = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        training_examples = [
            example for sample_id in training for example in examples.get(sample_id, [])
        ]
        fold = {"outer_fold": fold_index, "variants": {}}
        for name, feature_names in FEATURE_SETS.items():
            model = fit_pairwise_linear(
                training_examples, enabled_features=feature_names
            )
            result = evaluate(
                agent(feature_store, model),
                [samples[sample_id] for sample_id in sorted(outer)],
                catalog_ids,
                categories,
                products,
            )
            oof_sessions[name].extend(result["sessions"])
            fold["variants"][name] = {
                "weights": list(model.weights),
                "outer_score": result["recommended_technical_score"],
            }
            print(
                f"fold {fold_index} {name} {result['recommended_technical_score']}",
                flush=True,
            )
        folds.append(fold)

    metrics = {
        name: summarized_metrics(sessions) for name, sessions in oof_sessions.items()
    }
    best_score = max(
        float(result["recommended_technical_score"]) for result in metrics.values()
    )
    score_band = {
        name
        for name, result in metrics.items()
        if float(result["recommended_technical_score"]) >= best_score - 0.01
    }
    best_hit_rate = max(float(metrics[name]["hit_rate_at_10"]) for name in score_band)
    selected = min(
        (
            name
            for name in score_band
            if float(metrics[name]["hit_rate_at_10"]) == best_hit_rate
        ),
        key=lambda name: (
            len(FEATURE_SETS[name]),
            -float(metrics[name]["recommended_technical_score"]),
            name,
        ),
    )
    all_examples = [
        example for sample_id in sorted(examples) for example in examples[sample_id]
    ]
    selected_model = fit_pairwise_linear(
        all_examples, enabled_features=FEATURE_SETS[selected]
    )
    full_result = evaluate(
        agent(feature_store, selected_model),
        [samples[sample_id] for sample_id in sorted(samples)],
        catalog_ids,
        categories,
        products,
    )
    report = {
        "phase": 20,
        "gate": "nested_learned_feature_ablation",
        "split": "nested_v1_oof",
        "holdout_accessed": False,
        "collection": collection,
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "oof_metrics": metrics,
        "oof_sessions": oof_sessions,
        "folds": folds,
        "selection_rule": (
            "among variants within 0.01 of best OOF score, retain the best "
            "Hit Rate@10, then select the fewest features"
        ),
        "selected": selected,
        "selected_full_model": {
            "weights": list(selected_model.weights),
            "training_pairs": selected_model.training_pairs,
            "l2": selected_model.l2,
        },
        "selected_full_training_metrics": {
            key: full_result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
                "scenario_metrics",
            )
        },
        "selected_full_training_sessions": full_result["sessions"],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    output = ROOT / "artifacts/reports/phase20_learned_features.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "oof_metrics": metrics,
                "selected": selected,
                "selected_full_model": report["selected_full_model"],
                "selected_full_training_metrics": report[
                    "selected_full_training_metrics"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
