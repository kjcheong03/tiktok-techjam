from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
RAW_SEQUENCE = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)
SIMPLE_SEQUENCE = ("feature", "other", "other", "other")
COMMON: dict[str, object] = {
    "retrieval_route": "keyword",
    "sparse_weights": FIELD_WEIGHTS,
    "quality_prior_weight": 0.2,
}
CANDIDATES: dict[str, tuple[dict[str, object], int]] = {
    "raw_sequence": (
        {
            **COMMON,
            "state_variant": "raw_history",
            "question_variant": "sequence",
            "question_order": RAW_SEQUENCE,
            "negative_evidence": False,
        },
        4,
    ),
    "raw_other_always": (
        {
            **COMMON,
            "state_variant": "raw_history",
            "question_variant": "other_always",
            "negative_evidence": False,
        },
        2,
    ),
    "raw_no_question": (
        {
            **COMMON,
            "state_variant": "raw_history",
            "question_variant": "none",
            "negative_evidence": False,
        },
        1,
    ),
    "raw_adaptive": (
        {
            **COMMON,
            "state_variant": "raw_history",
            "question_variant": "adaptive",
            "negative_evidence": False,
        },
        5,
    ),
    "multi_sequence": (
        {
            **COMMON,
            "state_variant": "multi",
            "question_variant": "sequence",
            "question_order": SIMPLE_SEQUENCE,
        },
        3,
    ),
    "multi_other_always": (
        {
            **COMMON,
            "state_variant": "multi",
            "question_variant": "other_always",
        },
        2,
    ),
    "multi_adaptive": (
        {
            **COMMON,
            "state_variant": "multi",
            "question_variant": "adaptive",
        },
        5,
    ),
    "compressed_sequence": (
        {
            **COMMON,
            "state_variant": "compressed",
            "question_variant": "sequence",
            "question_order": SIMPLE_SEQUENCE,
        },
        4,
    ),
    "current_other_always": (
        {
            **COMMON,
            "state_variant": "current",
            "question_variant": "other_always",
        },
        1,
    ),
}


def main() -> None:
    started = time.perf_counter()
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    ]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    results = {}
    for name, (options, complexity) in CANDIDATES.items():
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
            name: statistics.fmean(rewards[sample_id] for sample_id in training)
            for name, rewards in reward_maps.items()
        }
        best = max(scores.values())
        selected = min(
            (name for name, score in scores.items() if score >= best - 0.01),
            key=lambda name: (CANDIDATES[name][1], -scores[name], name),
        )
        selections.append(
            {
                "outer_fold": fold_index,
                "selected": selected,
                "training_reward": round(scores[selected], 6),
                "outer_reward": round(
                    statistics.fmean(
                        reward_maps[selected][sample_id] for sample_id in outer
                    ),
                    6,
                ),
            }
        )

    ranked = sorted(
        results,
        key=lambda name: (
            -results[name]["metrics"]["recommended_technical_score"],
            CANDIDATES[name][1],
            name,
        ),
    )
    report = {
        "phase": 21,
        "gate": "bounded_cross_phase_state_question_recheck",
        "split": "nested_v1",
        "holdout_accessed": False,
        "candidate_count": len(CANDIDATES),
        "top": [
            {
                "name": name,
                "score": results[name]["metrics"]["recommended_technical_score"],
            }
            for name in ranked
        ],
        "fold_selections": selections,
        "nested_mean_outer_reward": round(
            statistics.fmean(item["outer_reward"] for item in selections), 6
        ),
        "variants": results,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    output = ROOT / "artifacts/reports/phase21_cross_phase_rechecks.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "top": report["top"],
                "fold_selections": selections,
                "nested_mean_outer_reward": report["nested_mean_outer_reward"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
