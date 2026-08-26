from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.research.replay import evaluate_replay
from ghostlab.research.technique_suite import (
    PROJECT_ROOT,
    build_suite_agent,
    load_suite_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one validated unified technique preset"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl"
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_suite_config(args.config)
    config_json = config.model_dump_json()
    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    agent = build_suite_agent(config, args.catalog)
    result = evaluate_replay(agent, samples, categories, products)
    report = {
        "experiment_id": config.experiment_id,
        "configuration_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "config": config.model_dump(mode="json"),
        "sample_count": len(samples),
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
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "configuration_sha256": report["configuration_sha256"],
                "sample_count": report["sample_count"],
                "metrics": report["metrics"],
                "output": None if args.output is None else str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
