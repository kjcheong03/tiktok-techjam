from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.policy.adaptive_questions import AdaptiveQuestionPolicy
from ghostlab.research.replay import session_reward
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "ghostlab/runtime/experimental.py",
    "ghostlab/policy/adaptive_questions.py",
    "ghostlab/policy/signals.py",
    "ghostlab/retrieval/filters.py",
    "ghostlab/retrieval/sparse.py",
    "ghostlab/state/memory.py",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants() -> dict[str, tuple[dict[str, object], int]]:
    default = AdaptiveQuestionPolicy()
    observed_priority = AdaptiveQuestionPolicy(
        priority=(
            "use_case",
            "size",
            "budget",
            "color",
            "feature",
            "material",
            "style",
        )
    )
    common: dict[str, object] = {
        "state_variant": "multi",
        "retrieval_route": "keyword",
    }
    return {
        "anchor_multi_other": (
            {**common, "question_variant": "other_always"},
            1,
        ),
        "standard_simple_sequence": (
            {
                **common,
                "question_variant": "sequence",
                "question_order": ("feature", "other", "other", "other"),
            },
            3,
        ),
        "standard_raw_best": (
            {
                **common,
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
            },
            5,
        ),
        "adaptive_multi": (
            {**common, "question_variant": "adaptive", "adaptive_policy": default},
            4,
        ),
        "adaptive_raw": (
            {
                **common,
                "state_variant": "raw_history",
                "question_variant": "adaptive",
                "adaptive_policy": default,
            },
            5,
        ),
        "adaptive_raw_no_negative_parser": (
            {
                **common,
                "state_variant": "raw_history",
                "question_variant": "adaptive",
                "adaptive_policy": default,
                "negative_evidence": False,
            },
            5,
        ),
        "adaptive_compressed": (
            {
                **common,
                "state_variant": "compressed",
                "question_variant": "adaptive",
                "adaptive_policy": default,
            },
            5,
        ),
        "adaptive_initial_other_1": (
            {
                **common,
                "question_variant": "adaptive",
                "adaptive_policy": AdaptiveQuestionPolicy(initial_other_turns=1),
            },
            5,
        ),
        "adaptive_initial_other_3": (
            {
                **common,
                "question_variant": "adaptive",
                "adaptive_policy": AdaptiveQuestionPolicy(initial_other_turns=3),
            },
            5,
        ),
        "adaptive_no_discovery_refresh": (
            {
                **common,
                "question_variant": "adaptive",
                "adaptive_policy": AdaptiveQuestionPolicy(other_refresh_interval=0),
            },
            4,
        ),
        "adaptive_observed_priority": (
            {
                **common,
                "question_variant": "adaptive",
                "adaptive_policy": observed_priority,
            },
            5,
        ),
        "adaptive_multi_filter": (
            {
                **common,
                "question_variant": "adaptive",
                "adaptive_policy": default,
                "structured_filter": True,
            },
            5,
        ),
        "adaptive_raw_filter": (
            {
                **common,
                "state_variant": "raw_history",
                "question_variant": "adaptive",
                "adaptive_policy": default,
                "structured_filter": True,
            },
            6,
        ),
        "adaptive_raw_observed_filter": (
            {
                **common,
                "state_variant": "raw_history",
                "question_variant": "adaptive",
                "adaptive_policy": observed_priority,
                "negative_evidence": False,
                "structured_filter": True,
            },
            7,
        ),
        "standard_raw_best_filter": (
            {
                **common,
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
                "structured_filter": True,
            },
            6,
        ),
    }


def serializable_options(options: dict[str, object]) -> dict[str, object]:
    return {
        key: asdict(value) if isinstance(value, AdaptiveQuestionPolicy) else value
        for key, value in options.items()
    }


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [sample for sample in samples if str(sample["sample_id"]) in adaptive_ids]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    results = {}
    for name, (options, complexity) in variants().items():
        started = time.perf_counter()
        agent = ExperimentalAgent(ROOT / "data/catalog.jsonl", **options)  # type: ignore[arg-type]
        result = evaluate(agent, samples, catalog_ids, categories, products)
        results[name] = {
            "options": serializable_options(options),
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
            "question_reasons": dict(
                sorted(
                    Counter(
                        str(item["reason"]) for item in agent.question_trace
                    ).items()
                )
            ),
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
    fold_selections = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        training_scores = {
            name: statistics.fmean(rewards[key] for key in training)
            for name, rewards in reward_maps.items()
        }
        best = max(training_scores.values())
        selected = min(
            (name for name, score in training_scores.items() if score >= best - 0.01),
            key=lambda name: (
                results[name]["complexity"],
                -training_scores[name],
                name,
            ),
        )
        fold_selections.append(
            {
                "outer_fold": fold_index,
                "selected": selected,
                "training_reward": round(training_scores[selected], 6),
                "outer_reward": round(
                    statistics.fmean(reward_maps[selected][key] for key in outer), 6
                ),
            }
        )

    report = {
        "phase": 13,
        "gate": "targeted_adaptive_and_filter_interactions",
        "split": "nested_v1",
        "sample_count": len(samples),
        "candidate_count": len(results),
        "holdout_accessed": False,
        "source_hashes": {path: file_hash(ROOT / path) for path in SOURCE_FILES},
        "data_hash": file_hash(ROOT / "data/public_set.jsonl"),
        "fold_selections": fold_selections,
        "nested_mean_outer_reward": round(
            statistics.fmean(item["outer_reward"] for item in fold_selections), 6
        ),
        "variants": results,
    }
    output = ROOT / "artifacts/reports/phase13_targeted_adaptive.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fold_selections": fold_selections,
                "nested_mean_outer_reward": report["nested_mean_outer_reward"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
