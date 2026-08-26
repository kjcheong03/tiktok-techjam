from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0.025, 0.05, 0.1, 0.2)
BASE_NAMES = (
    "raw_best_features_4",
    "raw_best_category_8",
    "title2_category8_feature4",
)


def normalize_options(values: dict[str, object]) -> dict[str, object]:
    options = dict(values)
    if isinstance(options.get("question_order"), list):
        options["question_order"] = tuple(options["question_order"])
    if isinstance(options.get("sparse_weights"), list):
        options["sparse_weights"] = tuple(options["sparse_weights"])
    return options


def main() -> None:
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    ]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    phase16 = json.loads(
        (ROOT / "artifacts/reports/phase16_field_weights.json").read_text()
    )["variants"]
    phase17 = json.loads(
        (ROOT / "artifacts/reports/phase17_field_interactions.json").read_text()
    )["variants"]
    controls = {
        "raw_best_features_4": phase16["raw_best_features_4"],
        "raw_best_category_8": phase16["raw_best_category_8"],
        "title2_category8_feature4": phase17["title2_category8_feature4"],
    }
    results = dict(controls)
    for base in BASE_NAMES:
        base_options = normalize_options(controls[base]["options"])
        for weight in WEIGHTS:
            name = f"{base}_quality_{weight:g}"
            options = {**base_options, "quality_prior_weight": weight}
            started = time.perf_counter()
            result = evaluate(
                ExperimentalAgent(ROOT / "data/catalog.jsonl", **options),  # type: ignore[arg-type]
                samples,
                catalog_ids,
                categories,
                products,
            )
            results[name] = {
                "options": options,
                "complexity": int(controls[base]["complexity"]) + 1,
                "metrics": {
                    key: result[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "recommended_technical_score",
                        "scenario_metrics",
                    )
                },
                "sessions": result["sessions"],
                "wall_seconds": round(time.perf_counter() - started, 6),
            }
            print(name, result["recommended_technical_score"], flush=True)

    reward_maps = {
        name: {
            str(session["sample_id"]): session_reward(session)
            for session in result["sessions"]
        }
        for name, result in results.items()
    }
    selections = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        scores = {
            name: statistics.fmean(rewards[key] for key in training)
            for name, rewards in reward_maps.items()
        }
        best = max(scores.values())
        selected = min(
            (name for name, score in scores.items() if score >= best - 0.01),
            key=lambda name: (
                results[name]["complexity"],
                -scores[name],
                name,
            ),
        )
        selections.append(
            {
                "outer_fold": fold_index,
                "selected": selected,
                "training_reward": round(scores[selected], 6),
                "outer_reward": round(
                    statistics.fmean(reward_maps[selected][key] for key in outer), 6
                ),
            }
        )
    ranked = sorted(
        results,
        key=lambda name: (
            -results[name]["metrics"]["recommended_technical_score"],
            results[name]["complexity"],
            name,
        ),
    )
    report = {
        "phase": 18,
        "gate": "catalog_quality_prior_interactions",
        "split": "nested_v1",
        "sample_count": len(samples),
        "holdout_accessed": False,
        "weights": list(WEIGHTS),
        "fold_selections": selections,
        "nested_mean_outer_reward": round(
            statistics.fmean(item["outer_reward"] for item in selections), 6
        ),
        "top": [
            {
                "name": name,
                "score": results[name]["metrics"]["recommended_technical_score"],
            }
            for name in ranked
        ],
        "variants": results,
    }
    output = ROOT / "artifacts/reports/phase18_quality_priors.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "top": report["top"],
                "fold_selections": selections,
                "nested_mean_outer_reward": report["nested_mean_outer_reward"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
