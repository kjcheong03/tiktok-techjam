from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "current_fixed": {"state_variant": "current", "question_variant": "fixed"},
    "raw_history_fixed": {"state_variant": "raw_history", "question_variant": "fixed"},
    "single_missing": {
        "state_variant": "single",
        "question_variant": "missing_priority",
    },
    "multi_missing": {"state_variant": "multi", "question_variant": "missing_priority"},
    "compressed_missing": {
        "state_variant": "compressed",
        "question_variant": "missing_priority",
    },
    "multi_no_negative": {
        "state_variant": "multi",
        "question_variant": "missing_priority",
        "negative_evidence": False,
    },
    "multi_no_override": {
        "state_variant": "multi",
        "question_variant": "missing_priority",
        "override_invalidation": False,
    },
    "multi_no_question": {"state_variant": "multi", "question_variant": "none"},
    "multi_fixed": {"state_variant": "multi", "question_variant": "fixed"},
    "multi_feature_first": {
        "state_variant": "multi",
        "question_variant": "feature_first",
    },
    "multi_uncertainty": {
        "state_variant": "multi",
        "question_variant": "uncertainty",
    },
    "multi_other_always": {
        "state_variant": "multi",
        "question_variant": "other_always",
    },
}


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    adaptive_ids = set(split["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in adaptive_ids]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    results: dict[str, object] = {}
    for name, options in VARIANTS.items():
        agent = ExperimentalAgent(ROOT / "data/catalog.jsonl", **options)  # type: ignore[arg-type]
        result = evaluate(agent, samples, catalog_ids, categories, products)
        question_counts = Counter(
            str(item["ask_attribute"])
            for item in agent.question_trace
            if item["ask_attribute"] is not None
        )
        results[name] = {
            **{key: value for key, value in result.items() if key != "sessions"},
            "question_counts": dict(sorted(question_counts.items())),
            "sessions": result["sessions"],
        }
        print(name, result["recommended_technical_score"], flush=True)
    report = {
        "phase": "4-5",
        "split": "adaptive_v1",
        "sample_count": len(samples),
        "variants": results,
    }
    output = ROOT / "artifacts/reports/phase4_5_ablations.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            key: value
            for key, value in result.items()
            if key in {"hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"}
        }
        for name, result in results.items()
    }
    (ROOT / "artifacts/reports/phase4_5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
