from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import load_jsonl
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.selected import sha256_file
from scripts.evaluate_adaptive_reference_baselines import evaluate_reference_a

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "data/public_set.jsonl"
DEFAULT_CATALOG = "data/catalog.jsonl"
DEFAULT_CONTROL = "configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json"
DEFAULT_FINALIST = (
    "configs/finalists/adaptive_hybrid_1650/"
    "rank_1_warm-start-d4e040a07e6d-translated-v2-sem-w0p10-d10-f1-"
    "add-fusion_rrf-714a27f0249c.json"
)
DEFAULT_E5_EMBEDDINGS = (
    "artifacts/cache/dense/e5_small_v2-da979b05a68a-eb21af922f8c.npy"
)
DEFAULT_E5_SIDECAR = (
    "artifacts/cache/dense/e5_small_v2-da979b05a68a-eb21af922f8c.json"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ordered_ids(report: dict[str, Any]) -> list[str]:
    sessions = report.get("sessions")
    if not isinstance(sessions, list):
        raise TypeError("evaluation report has no session rows")
    return [str(row["sample_id"]) for row in sessions]


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    source = report.get("metrics", report)
    if not isinstance(source, dict):
        raise TypeError("evaluation metrics must be an object")
    mttc = float(source["mttc"])
    efficiency = float(source.get("efficiency", max(0.0, (11.0 - mttc) / 10.0)))
    score = float(source["recommended_technical_score"])
    return {
        "hit_rate_at_10": float(source["hit_rate_at_10"]),
        "mrr": float(source["mrr"]),
        "mttc": mttc,
        "normalized_efficiency": efficiency,
        "technical_score": score,
    }


def _delta(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    return {
        key: candidate[key] - control[key]
        for key in (
            "hit_rate_at_10",
            "mrr",
            "normalized_efficiency",
            "technical_score",
        )
    } | {"mttc": candidate["mttc"] - control["mttc"]}


def build_public_comparison(
    *,
    reference_a: dict[str, Any],
    control: dict[str, Any],
    finalist: dict[str, Any],
    dataset_path: str,
    dataset_sha256: str,
    catalog_path: str,
    catalog_sha256: str,
    report_paths: dict[str, str],
    report_hashes: dict[str, str],
    control_config_path: str,
    control_config_file_sha256: str,
    control_config_canonical_sha256: str,
    finalist_config_path: str,
    finalist_config_file_sha256: str,
    finalist_config_canonical_sha256: str,
    local_assets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports = (reference_a, control, finalist)
    ordered_ids = [_ordered_ids(report) for report in reports]
    if len(ordered_ids[0]) != 200:
        raise ValueError(
            f"official public benchmark requires exactly 200 sessions, got {len(ordered_ids[0])}"
        )
    if len(set(ordered_ids[0])) != 200:
        raise ValueError("official public benchmark session IDs must be unique")
    if any(ids != ordered_ids[0] for ids in ordered_ids[1:]):
        raise ValueError("A/C/finalist reports use different ordered session IDs")
    contracts = [report.get("evaluation_contract") for report in reports]
    if not all(isinstance(contract, dict) for contract in contracts):
        raise TypeError("A/C/finalist reports must include an evaluation contract")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("A/C/finalist reports use different evaluator contracts")

    metrics_a, metrics_c, metrics_d = [_metrics(report) for report in reports]
    systems = [
        {
            "system_id": "A_official_stateless_bm25",
            "display_name": "Organizer BM25 Starter",
            "role": "reference",
            "metrics": metrics_a,
            "report_path": report_paths["A"],
            "report_sha256": report_hashes["A"],
        },
        {
            "system_id": "C_fixed_adaptive_architecture",
            "display_name": "Fixed Adaptive Architecture",
            "role": "control",
            "metrics": metrics_c,
            "report_path": report_paths["C"],
            "report_sha256": report_hashes["C"],
            "config_path": control_config_path,
            "config_file_sha256": control_config_file_sha256,
            "config_canonical_sha256": control_config_canonical_sha256,
        },
        {
            "system_id": "GhostLab_Champion",
            "display_name": "GhostLab Champion",
            "role": "champion",
            "metrics": metrics_d,
            "report_path": report_paths["D"],
            "report_sha256": report_hashes["D"],
            "config_path": finalist_config_path,
            "config_file_sha256": finalist_config_file_sha256,
            "config_canonical_sha256": finalist_config_canonical_sha256,
        },
    ]
    return {
        "schema_version": 1,
        "evaluation_scope": "official_public_200_benchmark",
        "sample_count": 200,
        "independent_holdout": False,
        "selection_or_training_overlap": True,
        "interpretation": (
            "Public organizer benchmark for reproducibility and demonstration; "
            "not an unseen generalization estimate."
        ),
        "comparison_contract": {
            "systems": ["A", "C", "GhostLab Champion"],
            "same_ordered_session_ids": True,
            "same_catalog": True,
            "same_evaluator_contract": True,
            "post_evaluation_tuning_allowed": False,
        },
        "inputs": {
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_sha256,
            "catalog_path": catalog_path,
            "catalog_sha256": catalog_sha256,
            "ordered_session_ids_sha256": hashlib.sha256(
                "\n".join(ordered_ids[0]).encode("utf-8")
            ).hexdigest(),
            "evaluation_contract": contracts[0],
            "local_assets": local_assets or {},
        },
        "systems": systems,
        "deltas": {
            "C_minus_A": _delta(metrics_c, metrics_a),
            "Champion_minus_C": _delta(metrics_d, metrics_c),
            "Champion_minus_A": _delta(metrics_d, metrics_a),
        },
    }


def _run_adaptive(
    *, config_path: str, output_path: str, dataset_path: str, catalog_path: str
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": ".",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    subprocess.run(
        (
            sys.executable,
            "scripts/run_adaptive_hybrid.py",
            "--catalog",
            catalog_path,
            "--dataset",
            dataset_path,
            "--partition",
            "all",
            "--config",
            config_path,
            "--output",
            output_path,
        ),
        cwd=ROOT,
        env=environment,
        check=True,
    )


def _build_benchmark_index(
    *,
    public_report_path: Path,
    final_selection_report_path: Path,
    final_selection_receipt_path: Path,
    adjudication_path: Path,
) -> dict[str, Any]:
    final_report = _load(final_selection_report_path)
    if int(final_report.get("sample_count", 0)) != 550:
        raise ValueError("final-selection evidence does not contain 550 sessions")
    system_ids = [str(item.get("system_id")) for item in final_report.get("systems", [])]
    if system_ids != [
        "A_official_stateless_bm25",
        "C_fixed_adaptive_architecture",
        "GhostLab_Challenger",
    ]:
        raise ValueError("final-selection evidence is not the frozen A/C/finalist comparison")
    return {
        "schema_version": 1,
        "comparison_contract": "A/C/GhostLab Champion",
        "evaluations": {
            "official_public_200": {
                "sample_count": 200,
                "report_path": public_report_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(public_report_path),
                "independent_holdout": False,
            },
            "final_selection_550": {
                "sample_count": 550,
                "report_path": final_selection_report_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(final_selection_report_path),
                "receipt_path": final_selection_receipt_path.relative_to(ROOT).as_posix(),
                "receipt_sha256": sha256_file(final_selection_receipt_path),
                "adjudication_path": adjudication_path.relative_to(ROOT).as_posix(),
                "adjudication_sha256": sha256_file(adjudication_path),
                "rerun_performed": False,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate A/C/GhostLab Champion on the official 200-session public set"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--control-config", default=DEFAULT_CONTROL)
    parser.add_argument("--finalist-config", default=DEFAULT_FINALIST)
    parser.add_argument("--e5-embeddings", default=DEFAULT_E5_EMBEDDINGS)
    parser.add_argument("--e5-sidecar", default=DEFAULT_E5_SIDECAR)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="rebuild comparison manifests from existing A/C/finalist run JSON",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/reports/adaptive_public_200_runs"
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_public_200.json"
    )
    parser.add_argument(
        "--benchmark-index",
        default="artifacts/reports/adaptive_ac_finalist_benchmark_index.json",
    )
    parser.add_argument(
        "--final-selection-report",
        default="artifacts/reports/adaptive_final_holdout.json",
    )
    parser.add_argument(
        "--final-selection-receipt",
        default="artifacts/reports/adaptive_final_holdout.access_receipt.json",
    )
    parser.add_argument(
        "--adjudication",
        default="configs/champion_adjudications/ghostlab_challenger_manual_v1.json",
    )
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    catalog = ROOT / args.catalog
    control_config = ROOT / args.control_config
    finalist_config = ROOT / args.finalist_config
    e5_embeddings = ROOT / args.e5_embeddings
    e5_sidecar = ROOT / args.e5_sidecar
    output_dir = ROOT / args.output_dir
    output_path = ROOT / args.output
    index_path = ROOT / args.benchmark_index
    samples = load_jsonl(dataset)
    if len(samples) != 200:
        raise ValueError(f"official public dataset must contain 200 sessions, got {len(samples)}")
    sample_ids = [str(item["sample_id"]) for item in samples]
    if len(set(sample_ids)) != 200:
        raise ValueError("official public dataset contains duplicate sample IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "A": output_dir / "reference_a.json",
        "C": output_dir / "control_c.json",
        "D": output_dir / "ghostlab_finalist.json",
    }
    if args.reuse_existing:
        missing = [path for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "cannot reuse missing public benchmark runs: "
                + ", ".join(str(path) for path in missing)
            )
        report_a = _load(paths["A"])
    else:
        print("Evaluating A — Organizer BM25 Starter on 200 sessions", flush=True)
        report_a = evaluate_reference_a(
            samples=samples,
            origins={sample_id: args.dataset for sample_id in sample_ids},
            catalog_path=catalog,
            partition="official_public_200",
            holdout_accessed=True,
        )
        _write(paths["A"], report_a)

        print("Evaluating C — Fixed Adaptive Architecture on 200 sessions", flush=True)
        _run_adaptive(
            config_path=args.control_config,
            output_path=paths["C"].relative_to(ROOT).as_posix(),
            dataset_path=args.dataset,
            catalog_path=args.catalog,
        )
        print("Evaluating GhostLab Champion on 200 sessions", flush=True)
        _run_adaptive(
            config_path=args.finalist_config,
            output_path=paths["D"].relative_to(ROOT).as_posix(),
            dataset_path=args.dataset,
            catalog_path=args.catalog,
        )

    if not e5_embeddings.is_file() or not e5_sidecar.is_file():
        raise FileNotFoundError("the pinned 50,000-product E5 cache is incomplete")

    comparison = build_public_comparison(
        reference_a=report_a,
        control=_load(paths["C"]),
        finalist=_load(paths["D"]),
        dataset_path=args.dataset,
        dataset_sha256=sha256_file(dataset),
        catalog_path=args.catalog,
        catalog_sha256=sha256_file(catalog),
        report_paths={key: path.relative_to(ROOT).as_posix() for key, path in paths.items()},
        report_hashes={key: sha256_file(path) for key, path in paths.items()},
        control_config_path=args.control_config,
        control_config_file_sha256=sha256_file(control_config),
        control_config_canonical_sha256=load_adaptive_hybrid_config(
            control_config
        ).canonical_hash(),
        finalist_config_path=args.finalist_config,
        finalist_config_file_sha256=sha256_file(finalist_config),
        finalist_config_canonical_sha256=load_adaptive_hybrid_config(
            finalist_config
        ).canonical_hash(),
        local_assets={
            "e5_product_embeddings": {
                "path": args.e5_embeddings,
                "sha256": sha256_file(e5_embeddings),
                "byte_size": e5_embeddings.stat().st_size,
            },
            "e5_cache_sidecar": {
                "path": args.e5_sidecar,
                "sha256": sha256_file(e5_sidecar),
                "byte_size": e5_sidecar.stat().st_size,
            },
        },
    )
    _write(output_path, comparison)
    index = _build_benchmark_index(
        public_report_path=output_path,
        final_selection_report_path=ROOT / args.final_selection_report,
        final_selection_receipt_path=ROOT / args.final_selection_receipt,
        adjudication_path=ROOT / args.adjudication,
    )
    _write(index_path, index)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
