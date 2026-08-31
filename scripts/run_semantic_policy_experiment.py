from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from scripts.compare_local_llm_rankers import (
    DEFAULT_DATASETS,
    MODELS,
    _canonical_sha256,
    _float_value,
    _int_value,
    _run_worker,
    lineage_safe_sample_ids,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json"
DEFAULT_MANIFEST = "data/splits/adaptive_hybrid_lineage_75_25_v1.json"
MODEL_ID = "smollm2-1.7b-instruct"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trial_matrix() -> tuple[dict[str, object], ...]:
    return (
        {
            "trial_id": "browsing_control_no_llm",
            "scenario_type": "browsing",
            "activation_mode": "never",
            "weight": 0.05,
        },
        *(
            {
                "trial_id": f"browsing_all_weight_{weight:.2f}",
                "scenario_type": "browsing",
                "activation_mode": "browsing_all",
                "weight": weight,
            }
            for weight in (0.05, 0.10, 0.15)
        ),
        *(
            {
                "trial_id": f"browsing_gated_margin_{margin:.2f}_weight_{weight:.2f}",
                "scenario_type": "browsing",
                "activation_mode": "browsing_ambiguous",
                "maximum_margin": margin,
                "minimum_entropy": 0.85,
                "weight": weight,
            }
            for margin, weight in ((0.02, 0.05), (0.02, 0.10), (0.05, 0.10))
        ),
        {
            "trial_id": "buying_control_no_llm",
            "scenario_type": "buying",
            "activation_mode": "never",
            "weight": 0.05,
        },
        *(
            {
                "trial_id": f"buying_all_weight_{weight:.2f}",
                "scenario_type": "buying",
                "activation_mode": "buying_all",
                "weight": weight,
            }
            for weight in (0.05, 0.10)
        ),
        *(
            {
                "trial_id": f"buying_semantic_constraints_weight_{weight:.2f}",
                "scenario_type": "buying",
                "activation_mode": "buying_semantic_constraints",
                "weight": weight,
            }
            for weight in (0.05, 0.10)
        ),
    )


def _delta(result: dict[str, object], control: dict[str, object]) -> dict[str, float]:
    return {
        key: _float_value(result[key]) - _float_value(control[key])
        for key in (
            "recommended_technical_score",
            "hit_rate_at_10",
            "mrr",
            "mttc",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated low-weight and activation-gating semantic experiment"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--lineage-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--max-samples-per-scenario", type=int, default=60)
    parser.add_argument("--trial-timeout-seconds", type=int, default=1800)
    parser.add_argument("--component-timeout-ms", type=int, default=120000)
    parser.add_argument(
        "--output",
        default="artifacts/reports/semantic_policy_low_weight_experiment.json",
    )
    args = parser.parse_args()
    if args.max_samples_per_scenario <= 0:
        raise ValueError("max samples per scenario must be positive")

    config_path = ROOT / args.config
    before_hash = _file_sha256(config_path)
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    corpus = load_adaptive_training_corpus(ROOT, datasets)
    manifest_path = ROOT / args.lineage_manifest
    manifest = load_lineage_manifest(manifest_path, corpus)
    development = subset_corpus(corpus, manifest, "development")
    fold_ids = {
        scenario: lineage_safe_sample_ids(
            development,
            manifest,
            args.max_samples_per_scenario,
            scenario_type=scenario,
        )
        for scenario in ("browsing", "buying")
    }
    definition = MODELS[MODEL_ID]
    trials = _trial_matrix()
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="semantic-policy-study-") as value:
        temporary = Path(value)
        for index, trial in enumerate(trials, start=1):
            scenario = str(trial["scenario_type"])
            spec: dict[str, Any] = {
                "config": args.config,
                "datasets": list(datasets),
                "lineage_manifest": args.lineage_manifest,
                "fold_sample_ids": [list(fold) for fold in fold_ids[scenario]],
                "component_timeout_ms": args.component_timeout_ms,
                "model_id": MODEL_ID,
                "definition": definition,
                "depth": 10,
                "maximum_margin": 0.02,
                "minimum_entropy": 0.85,
                **trial,
            }
            print(
                f"START semantic-policy trial {index}/{len(trials)} "
                f"id={trial['trial_id']}",
                flush=True,
            )
            result = _run_worker(
                spec,
                temporary=temporary,
                timeout_seconds=args.trial_timeout_seconds,
            )
            result["trial_id"] = trial["trial_id"]
            result["scenario_type"] = scenario
            results.append(result)
            print(
                f"DONE semantic-policy trial {index}/{len(trials)} "
                f"status={result['status']}",
                flush=True,
            )

    complete = [item for item in results if item.get("status") == "complete"]
    controls = {
        scenario: next(
            item
            for item in complete
            if item["trial_id"] == f"{scenario}_control_no_llm"
        )
        for scenario in ("browsing", "buying")
    }
    comparisons: list[dict[str, object]] = []
    for item in complete:
        scenario = str(item["scenario_type"])
        control = controls[scenario]
        is_control = item is control
        deltas = _delta(item, control)
        removal_delta = _int_value(item["confirmed_target_removal_count"]) - _int_value(
            control["confirmed_target_removal_count"]
        )
        safe = (
            _int_value(item["output_constraint_violations"]) == 0
            and removal_delta <= 0
            and _int_value(item["target_demoted_from_top10"]) == 0
        )
        helpful = (
            not is_control
            and safe
            and _int_value(item["semantic_activations"]) > 0
            and deltas["recommended_technical_score"] > 0.0
            and deltas["mrr"] >= 0.0
            and deltas["hit_rate_at_10"] >= 0.0
        )
        comparisons.append(
            {
                "trial_id": item["trial_id"],
                "scenario_type": scenario,
                "control": is_control,
                "safe": safe,
                "helpful": helpful,
                "confirmed_target_removal_delta": removal_delta,
                "deltas_vs_no_llm": deltas,
            }
        )

    paired = {}
    for scenario in ("browsing", "buying"):
        scenario_results = [
            item for item in complete if item["scenario_type"] == scenario
        ]
        paired[scenario] = {
            "ordered_sessions": len(
                {str(item["ordered_session_ids_sha256"]) for item in scenario_results}
            )
            == 1,
            "candidate_pools": len(
                {str(item["candidate_pool_sha256"]) for item in scenario_results}
            )
            == 1,
        }

    after_hash = _file_sha256(config_path)
    payload = {
        "schema_version": 1,
        "evaluation_scope": "lineage_safe_development_semantic_policy_experiment",
        "partition": "development",
        "holdout_accessed": False,
        "datasets": list(datasets),
        "lineage_manifest": args.lineage_manifest,
        "lineage_manifest_sha256": manifest.manifest_sha256,
        "model_id": MODEL_ID,
        "depth": 10,
        "trial_count": len(trials),
        "completed_trial_count": len(complete),
        "fold_sample_counts": {
            scenario: [len(fold) for fold in folds]
            for scenario, folds in fold_ids.items()
        },
        "paired_evidence": paired,
        "selected_runtime_config": args.config,
        "selected_runtime_config_sha256_before": before_hash,
        "selected_runtime_config_sha256_after": after_hash,
        "selected_runtime_config_unchanged": before_hash == after_hash,
        "results": results,
        "comparisons": comparisons,
        "helpful_trials": [
            item["trial_id"] for item in comparisons if bool(item["helpful"])
        ],
        "experiment_does_not_promote_or_modify_runtime": True,
        "ordered_sessions_sha256": {
            scenario: _canonical_sha256(tuple(value for fold in folds for value in fold))
            for scenario, folds in fold_ids.items()
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
