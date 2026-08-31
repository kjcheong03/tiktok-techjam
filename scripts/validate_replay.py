from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import evaluate_replay
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    allowed = set(split["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in allowed]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    options = {"state_variant": "multi", "question_variant": "fixed"}
    official = evaluate(
        ExperimentalAgent(ROOT / "data/catalog.jsonl", **options),
        samples,
        catalog_ids,
        categories,
        products,
    )
    replay = evaluate_replay(
        ExperimentalAgent(ROOT / "data/catalog.jsonl", **options),
        samples,
        categories,
        products,
    )
    parity_keys = (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "scenario_metrics",
        "sessions",
    )
    mismatches = {
        key: [official[key], replay[key]]
        for key in parity_keys
        if official[key] != replay[key]
    }
    report = {
        "phase": 6,
        "split": "adaptive_v1",
        "sample_count": len(samples),
        "passed": not mismatches,
        "mismatches": mismatches,
        "metrics": {key: replay[key] for key in parity_keys[:-1]},
    }
    output = ROOT / "artifacts/reports/phase6_replay_parity.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit("replay parity failed")


if __name__ == "__main__":
    main()
