from __future__ import annotations

import json
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "weighted_90_10_fixed": {
        "state_variant": "multi",
        "question_variant": "fixed",
        "retrieval_route": "weighted",
        "sparse_weight": 0.9,
        "dense_weight": 0.1,
    },
    "weighted_85_15_fixed": {
        "state_variant": "multi",
        "question_variant": "fixed",
        "retrieval_route": "weighted",
        "sparse_weight": 0.85,
        "dense_weight": 0.15,
    },
    "keyword_fixed_linear": {
        "state_variant": "multi",
        "question_variant": "fixed",
        "retrieval_route": "keyword",
        "reranker": "linear",
    },
    "raw_history_other": {
        "state_variant": "raw_history",
        "question_variant": "other_always",
        "retrieval_route": "keyword",
    },
    "rrf_multi_other": {
        "state_variant": "multi",
        "question_variant": "other_always",
        "retrieval_route": "rrf",
    },
    "weighted_90_10_multi_other": {
        "state_variant": "multi",
        "question_variant": "other_always",
        "retrieval_route": "weighted",
        "sparse_weight": 0.9,
        "dense_weight": 0.1,
    },
}


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    adaptive_ids = set(split["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in adaptive_ids]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    results = {}
    for name, options in VARIANTS.items():
        started = time.perf_counter()
        result = evaluate(
            ExperimentalAgent(ROOT / "data/catalog.jsonl", **options),  # type: ignore[arg-type]
            samples,
            catalog_ids,
            categories,
            products,
        )
        results[name] = {
            **result,
            "wall_seconds": round(time.perf_counter() - started, 6),
        }
        print(name, result["recommended_technical_score"], flush=True)
    report = {
        "phase": 11,
        "split": "adaptive_v1_exploratory",
        "sample_count": len(samples),
        "variants": results,
    }
    output = ROOT / "artifacts/reports/phase11_optional.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            key: value
            for key, value in result.items()
            if key
            in {
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
                "wall_seconds",
            }
        }
        for name, result in results.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
