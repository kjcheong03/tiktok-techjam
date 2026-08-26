from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0.05, 0.1, 0.2, 0.4)
BASES: dict[str, tuple[dict[str, object], int]] = {
    "anchor": (
        {
            "state_variant": "multi",
            "question_variant": "other_always",
            "retrieval_route": "keyword",
        },
        1,
    ),
    "simple_sequence": (
        {
            "state_variant": "multi",
            "question_variant": "sequence",
            "question_order": ("feature", "other", "other", "other"),
            "retrieval_route": "keyword",
        },
        3,
    ),
    "raw_best": (
        {
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
        },
        5,
    ),
}


def main() -> None:
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    ]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    variants = {
        **{
            name: (options, complexity) for name, (options, complexity) in BASES.items()
        },
        **{
            f"{name}_profile_{weight:g}": (
                {**options, "profile_prior_weight": weight},
                complexity + 1,
            )
            for name, (options, complexity) in BASES.items()
            for weight in WEIGHTS
        },
    }
    results = {}
    for name, (options, complexity) in variants.items():
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
            "complexity": complexity,
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

    report = {
        "phase": 15,
        "gate": "profile_prior_weight_ablation",
        "split": "nested_v1",
        "sample_count": len(samples),
        "holdout_accessed": False,
        "weights": list(WEIGHTS),
        "fold_selections": selections,
        "nested_mean_outer_reward": round(
            statistics.fmean(item["outer_reward"] for item in selections), 6
        ),
        "variants": results,
    }
    output = ROOT / "artifacts/reports/phase15_profile_priors.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fold_selections": selections,
                "nested_mean_outer_reward": report["nested_mean_outer_reward"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
