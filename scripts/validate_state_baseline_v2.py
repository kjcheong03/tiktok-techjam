from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.research.replay import evaluate_replay, paired_delta, session_reward
from ghostlab.research.technique_suite import (
    PROJECT_ROOT,
    build_suite_agent,
    load_suite_config,
)

REFERENCE_SCORES = {
    "state_baseline_v2_raw_control.json": 0.662395,
    "state_baseline_v2_fixed.json": 0.782154,
    "state_baseline_v2_other.json": 0.837003,
}
REFERENCE_SESSION_HASHES = {
    "state_baseline_v2_raw_control.json": (
        "c6bfe3a049295472c8231942f47b026f3200bf4d0c2d8eeddee3904c9c658ec0"
    ),
    "state_baseline_v2_fixed.json": (
        "8656d3497e1d7fccd657fd611dc40febbf4fec928fc057077e885f339ef657a4"
    ),
    "state_baseline_v2_other.json": (
        "88af16b45d89e06dfdbe4cf718732eec5b34742d86992fd90b6a59bbcbda1bf6"
    ),
}
INTEGRATION_INPUTS = (
    "ghostlab/state/baseline_v2.py",
    "ghostlab/state/query.py",
    "ghostlab/state/v2_view.py",
    "ghostlab/runtime/unified_experimental.py",
    "ghostlab/research/technique_suite.py",
    "configs/suites/state_baseline_v2_raw_control.json",
    "configs/suites/state_baseline_v2_fixed.json",
    "configs/suites/state_baseline_v2_other.json",
    "configs/suites/state_baseline_v2_ranked.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_hash(sessions: list[dict]) -> str:
    canonical = json.dumps(
        sorted(sessions, key=lambda item: str(item["sample_id"])),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def fold_scores(sessions: list[dict], folds: list[list[str]]) -> list[float]:
    by_id = {str(item["sample_id"]): item for item in sessions}
    return [
        round(
            statistics.fmean(session_reward(by_id[sample_id]) for sample_id in fold), 6
        )
        for fold in folds
    ]


def evaluate(
    config_path: Path, samples: list[dict], categories: dict, products: dict
) -> dict:
    config = load_suite_config(config_path)
    result = evaluate_replay(
        build_suite_agent(config, PROJECT_ROOT / "data/catalog.jsonl"),
        samples,
        categories,
        products,
    )
    return {
        "experiment_id": config.experiment_id,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "configuration_sha256": hashlib.sha256(
            config.model_dump_json().encode()
        ).hexdigest(),
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
        "session_sha256": session_hash(result["sessions"]),
        "sessions": result["sessions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the native State Baseline V2 integration on adaptive_v1 only"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/reports/state_baseline_v2_integration.json",
    )
    args = parser.parse_args()

    adaptive_path = PROJECT_ROOT / "configs/splits/adaptive_v1.json"
    nested_path = PROJECT_ROOT / "configs/splits/nested_v1.json"
    dataset_path = PROJECT_ROOT / "data/public_set.jsonl"
    catalog_path = PROJECT_ROOT / "data/catalog.jsonl"
    adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    allowed = {str(value) for value in adaptive["sample_ids"]}
    samples = [
        row for row in load_jsonl(dataset_path) if str(row["sample_id"]) in allowed
    ]
    if len(samples) != 150:
        raise RuntimeError("State Baseline V2 validation requires exactly adaptive_v1")
    if set(nested["adaptive_sample_ids"]) != allowed:
        raise RuntimeError("nested folds do not partition adaptive_v1")

    _, categories, products = catalog_index(catalog_path)
    config_names = [
        *REFERENCE_SCORES,
        "state_baseline_v2_ranked.json",
    ]
    results = {
        name: evaluate(
            PROJECT_ROOT / "configs/suites" / name,
            samples,
            categories,
            products,
        )
        for name in config_names
    }

    parity: dict[str, dict] = {}
    for name, expected in REFERENCE_SCORES.items():
        observed = results[name]["metrics"]["recommended_technical_score"]
        parity[name] = {
            "expected": expected,
            "observed": observed,
            "expected_session_sha256": REFERENCE_SESSION_HASHES[name],
            "observed_session_sha256": results[name]["session_sha256"],
            "exact": (
                observed == expected
                and results[name]["session_sha256"] == REFERENCE_SESSION_HASHES[name]
            ),
        }
        if not parity[name]["exact"]:
            raise RuntimeError(f"technical parity failed for {name}")

    control = results["state_baseline_v2_raw_control.json"]
    comparisons: dict[str, dict] = {}
    for name in config_names[1:]:
        candidate = results[name]
        deltas = paired_delta(candidate["sessions"], control["sessions"])
        comparisons[name] = {
            "against": "state_baseline_v2_raw_control.json",
            "mean_session_reward_delta": round(statistics.fmean(deltas), 6),
            "bootstrap_95_interval": [
                round(value, 6)
                for value in bootstrap_mean_interval(deltas, resamples=10_000)
            ],
            "paired_randomization_pvalue": round(
                paired_randomization_pvalue(deltas, resamples=10_000), 6
            ),
            "positive_outer_folds": sum(
                candidate_score > control_score
                for candidate_score, control_score in zip(
                    fold_scores(candidate["sessions"], nested["outer_folds"]),
                    fold_scores(control["sessions"], nested["outer_folds"]),
                    strict=True,
                )
            ),
            "outer_fold_scores": fold_scores(
                candidate["sessions"], nested["outer_folds"]
            ),
        }

    report = {
        "schema_version": 1,
        "evaluation_label": "fixed adaptive_v1 mechanism and interaction validation",
        "sample_count": len(samples),
        "adaptive_split": str(adaptive_path.relative_to(PROJECT_ROOT)),
        "nested_split": str(nested_path.relative_to(PROJECT_ROOT)),
        "protected_holdout_accessed": False,
        "data_sha256": sha256_file(dataset_path),
        "catalog_sha256": sha256_file(catalog_path),
        "integration_input_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in INTEGRATION_INPUTS
        },
        "parity": parity,
        "comparisons": comparisons,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protected_holdout_accessed": False,
                "parity": parity,
                "scores": {
                    name: value["metrics"]["recommended_technical_score"]
                    for name, value in results.items()
                },
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
