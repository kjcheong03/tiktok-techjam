from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
RAW_OPTIONS: dict[str, object] = {
    "state_variant": "raw_history",
    "question_variant": "sequence",
    "question_order": (
        "other",
        "other",
        "use_case",
        "other",
        "size",
        "other",
        "other",
        "size",
    ),
    "negative_evidence": False,
    "retrieval_route": "keyword",
}
INTERACTIONS = {
    "feature4_category6": (6.0, 6.0, 4.0, 2.5, 1.5, 1.0),
    "feature4_category8": (6.0, 8.0, 4.0, 2.5, 1.5, 1.0),
    "title2_feature4": (2.0, 4.0, 4.0, 2.5, 1.5, 1.0),
    "title2_category8": (2.0, 8.0, 2.5, 2.5, 1.5, 1.0),
    "title2_category8_feature4": (2.0, 8.0, 4.0, 2.5, 1.5, 1.0),
    "title4_category8_feature4": (4.0, 8.0, 4.0, 2.5, 1.5, 1.0),
    "feature4_without_description": (6.0, 4.0, 4.0, 2.5, 1.5, 0.0),
    "category8_without_description": (6.0, 8.0, 2.5, 2.5, 1.5, 0.0),
}
CONTROL_NAMES = (
    "raw_best_organizer",
    "raw_best_title_2",
    "raw_best_category_8",
    "raw_best_features_4",
)


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
    results = {name: phase16[name] for name in CONTROL_NAMES}
    for name, weights in INTERACTIONS.items():
        options = {**RAW_OPTIONS, "sparse_weights": weights}
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
            "complexity": 7,
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
        "phase": 17,
        "gate": "evidence_backed_field_interactions",
        "split": "nested_v1",
        "sample_count": len(samples),
        "holdout_accessed": False,
        "candidate_count": len(results),
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
    output = ROOT / "artifacts/reports/phase17_field_interactions.json"
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
