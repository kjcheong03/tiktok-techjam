from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from baseline.official_reference import Agent as OfficialKeywordAgent
from evaluator.local_evaluator import catalog_index, metric_summary
from ghostlab.research.replay import evaluate_shared
from ghostlab.research.technique_suite import build_suite_agent, load_suite_config
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_metrics(
    sessions: list[dict[str, Any]], origins: dict[str, str]
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[origins[str(session["sample_id"])]].append(session)
    return {source: metric_summary(rows) for source, rows in sorted(grouped.items())}


def _report(
    *,
    system_id: str,
    result: dict[str, Any],
    origins: dict[str, str],
    elapsed_seconds: float,
    provenance: dict[str, object],
    partition: str,
    holdout_accessed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "system_id": system_id,
        "role": "explanatory_baseline",
        "champion_eligible": False,
        "partition": partition,
        "holdout_accessed": holdout_accessed,
        "sample_count": len(result["sessions"]),
        "metrics": {
            key: result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
                "scenario_metrics",
            )
        },
        "source_metrics": _source_metrics(result["sessions"], origins),
        "evaluation_contract": result["evaluation_contract"],
        "evaluation_seconds": round(elapsed_seconds, 6),
        "provenance": provenance,
        "sessions": result["sessions"],
    }


def evaluate_reference_systems(
    *,
    samples: list[dict[str, Any]],
    origins: dict[str, str],
    catalog_path: Path,
    state_config_path: Path,
    partition: str,
    holdout_accessed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate frozen A/B references on one identical ordered sample population."""

    report_a = evaluate_reference_a(
        samples=samples,
        origins=origins,
        catalog_path=catalog_path,
        partition=partition,
        holdout_accessed=holdout_accessed,
    )
    _, categories, products = catalog_index(catalog_path)
    state_config = load_suite_config(state_config_path)
    started = time.perf_counter()
    result_b = evaluate_shared(
        build_suite_agent(state_config, catalog_path),
        samples,
        categories,
        products,
        catalog_path=catalog_path,
    )
    report_b = _report(
        system_id="B_state_baseline_v2_tagged_best",
        result=result_b,
        origins=origins,
        elapsed_seconds=time.perf_counter() - started,
        provenance={
            "teammate_source": "origin/feat/state-baseline-v2@7b78dd4",
            "teammate_tag": "state-baseline-v2-best",
            "native_config": str(state_config_path.relative_to(ROOT)),
            "native_exact_parity_evidence": (
                "artifacts/reports/state_baseline_v2_integration.json"
            ),
            "configuration": "coverage_adaptive_state_with_history + fixed_other",
            "diagnostic_policy_warning": (
                "fixed_other is simulator-sensitive and is an explanatory baseline, "
                "not a production champion"
            ),
        },
        partition=partition,
        holdout_accessed=holdout_accessed,
    )
    if [row["sample_id"] for row in report_a["sessions"]] != [
        row["sample_id"] for row in report_b["sessions"]
    ]:
        raise RuntimeError("A and B did not evaluate identical ordered sample IDs")
    return report_a, report_b


def evaluate_reference_a(
    *,
    samples: list[dict[str, Any]],
    origins: dict[str, str],
    catalog_path: Path,
    partition: str,
    holdout_accessed: bool,
) -> dict[str, Any]:
    """Evaluate only the frozen organizer BM25 reference."""

    _, categories, products = catalog_index(catalog_path)
    started = time.perf_counter()
    result_a = evaluate_shared(
        OfficialKeywordAgent(catalog_path),
        samples,
        categories,
        products,
        catalog_path=catalog_path,
    )
    report_a = _report(
        system_id="A_official_stateless_bm25",
        result=result_a,
        origins=origins,
        elapsed_seconds=time.perf_counter() - started,
        provenance={
            "implementation": "baseline/official_reference.py",
            "description": "organizer stateless SQLite FTS5/BM25 reference",
        },
        partition=partition,
        holdout_accessed=holdout_accessed,
    )
    return report_a


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stateless organizer BM25 (A) on development; optionally "
            "evaluate the research-only State Baseline V2 reproduction"
        )
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument(
        "--state-config", default="configs/suites/state_baseline_v2_other.json"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--output-a",
        default="artifacts/reports/adaptive_baseline_a_development_1650.json",
    )
    parser.add_argument(
        "--output-b",
        default="artifacts/reports/adaptive_baseline_b_development_1650.json",
    )
    parser.add_argument(
        "--include-state-v2-reference",
        action="store_true",
        help="also evaluate research-only State Baseline V2 (not part of A/C/D)",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    complete = load_adaptive_training_corpus(ROOT, datasets)
    lineage = load_lineage_manifest(ROOT / args.lineage_manifest, complete)
    development = subset_corpus(complete, lineage, "development")
    samples = [development.samples[item] for item in sorted(development.samples)]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    selected_ids = {str(item["sample_id"]) for item in samples}
    origins = {
        sample_id: source
        for sample_id, source in development.origins.items()
        if sample_id in selected_ids
    }
    catalog_path = ROOT / args.catalog
    report_a = evaluate_reference_a(
        samples=samples,
        origins=origins,
        catalog_path=catalog_path,
        partition="development",
        holdout_accessed=False,
    )
    _write_json(ROOT / args.output_a, report_a)
    print(
        f"DONE A samples={len(samples)} "
        f"score={report_a['metrics']['recommended_technical_score']:.6f}",
        flush=True,
    )
    if args.include_state_v2_reference:
        _, report_b = evaluate_reference_systems(
            samples=samples,
            origins=origins,
            catalog_path=catalog_path,
            state_config_path=ROOT / args.state_config,
            partition="development",
            holdout_accessed=False,
        )
        _write_json(ROOT / args.output_b, report_b)
        print(
            f"DONE research-only B samples={len(samples)} "
            f"score={report_b['metrics']['recommended_technical_score']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
