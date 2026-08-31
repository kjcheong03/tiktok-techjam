from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from ghostlab.evaluation.statistics import bootstrap_mean_interval
from ghostlab.optimization.racing import lineage_cluster_means
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.selected import sha256_file
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from scripts.evaluate_adaptive_reference_baselines import evaluate_reference_a

ROOT = Path(__file__).resolve().parents[1]
SELECTION_TIE_BREAK_ORDER = [
    "recommended_technical_score:desc",
    "mrr:desc",
    "hit_rate_at_10:desc",
    "mttc:asc",
    "fallback_rate:asc",
    "development_rank:asc",
    "candidate_id:asc",
]
DEFAULT_DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def verify_frozen_holdout_inputs(
    frozen: dict[str, Any],
    *,
    control_config_path: Path,
    reference_a_path: Path,
    gates_path: Path,
) -> dict[str, str]:
    """Verify every comparison dependency before the holdout is opened."""

    dependencies = (
        (
            "control_config_path",
            "control_config_file_sha256",
            control_config_path,
        ),
        (
            "reference_a_implementation_path",
            "reference_a_implementation_sha256",
            reference_a_path,
        ),
        ("gates_path", "gates_sha256", gates_path),
    )
    verified: dict[str, str] = {}
    for path_key, hash_key, actual_path in dependencies:
        relative = actual_path.relative_to(ROOT).as_posix()
        if frozen.get(path_key) != relative:
            raise ValueError(f"frozen {path_key} does not match the requested input")
        actual_hash = sha256_file(actual_path)
        if frozen.get(hash_key) != actual_hash:
            raise ValueError(f"frozen {hash_key} changed before holdout")
        verified[hash_key] = actual_hash

    canonical = load_adaptive_hybrid_config(control_config_path).canonical_hash()
    if frozen.get("control_config_canonical_sha256") != canonical:
        raise ValueError("frozen control canonical config hash changed before holdout")
    verified["control_config_canonical_sha256"] = canonical
    return verified


def _run_evaluation(
    config: str, output: str, datasets: tuple[str, ...], manifest: str
) -> None:
    command = [
        sys.executable,
        "scripts/run_adaptive_hybrid.py",
        "--config",
        config,
        "--lineage-manifest",
        manifest,
        "--partition",
        "holdout",
        "--output",
        output,
    ]
    for dataset in datasets:
        command.extend(("--dataset", dataset))
    subprocess.run(command, cwd=ROOT, check=True)


def _metric(report: dict[str, Any], key: str) -> float:
    return float(report[key])


def _session_reward(session: dict[str, Any]) -> float:
    hit = float(bool(session["hit"]))
    reciprocal = float(session["reciprocal_rank"])
    first_hit = session.get("first_hit_turn")
    turn = float(first_hit) if isinstance(first_hit, int) else 11.0
    efficiency = max(0.0, min(1.0, (11.0 - turn) / 10.0))
    return 0.5 * hit + 0.3 * reciprocal + 0.2 * efficiency


def paired_cluster_statistics(
    challenger: dict[str, Any],
    control: dict[str, Any],
    group_by_sample: dict[str, str],
) -> dict[str, float | int]:
    challenger_rows = {str(row["sample_id"]): row for row in challenger["sessions"]}
    control_rows = {str(row["sample_id"]): row for row in control["sessions"]}
    identifiers = sorted(set(challenger_rows) & set(control_rows))
    if set(identifiers) != set(group_by_sample):
        raise ValueError("holdout reports do not cover the frozen holdout IDs exactly")
    deltas = [
        _session_reward(challenger_rows[item]) - _session_reward(control_rows[item])
        for item in identifiers
    ]
    cluster_values = lineage_cluster_means(
        deltas, [group_by_sample[item] for item in identifiers]
    )
    lower, upper = bootstrap_mean_interval(
        cluster_values, resamples=5000, confidence=0.95, seed=20260831
    )
    return {
        "sample_count": len(deltas),
        "lineage_cluster_count": len(cluster_values),
        "mean_paired_delta": statistics.fmean(cluster_values),
        "confidence": 0.95,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def verify_development_evidence(
    proposal: dict[str, Any],
    challenger_path: Path,
    manifest_hash: str,
    holdout_ids: frozenset[str],
) -> dict[str, object]:
    campaign_path = ROOT / str(proposal["campaign_report"])
    campaign = _load(campaign_path)
    if campaign.get("partition") != "development":
        raise ValueError("campaign evidence is not development-only")
    if campaign.get("lineage_manifest_sha256") != manifest_hash:
        raise ValueError("campaign evidence uses a different lineage manifest")
    config = load_adaptive_hybrid_config(challenger_path)
    receipt_checked = False
    if config.union_ranker.backend == "gbdt":
        assert config.union_ranker.model_path is not None
        model_path = ROOT / config.union_ranker.model_path
        receipt_path = model_path.with_name(f"{model_path.stem}.fit_receipt.json")
        receipt = _load(receipt_path)
        split_path = ROOT / str(receipt["split_manifest_path"])
        split = _load(split_path)
        development_ids = {
            str(sample_id)
            for fold in split.get("outer_folds", [])
            for sample_id in fold
        }
        if development_ids & holdout_ids:
            raise ValueError("fit receipt split contains holdout IDs")
        if receipt.get("holdout_accessed") is not False:
            raise ValueError("fit receipt does not prove holdout isolation")
        receipt_checked = True
    return {
        "campaign_partition": "development",
        "campaign_manifest_verified": True,
        "fit_receipt_checked": receipt_checked,
        "holdout_ids_in_development_evidence": 0,
    }


def _slice_gates(
    challenger: dict[str, Any],
    control: dict[str, Any],
    section: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    challenger_slices = challenger.get(section, {})
    control_slices = control.get(section, {})
    for name in sorted(set(challenger_slices) & set(control_slices)):
        candidate_value = float(challenger_slices[name]["hit_rate_at_10"])
        control_value = float(control_slices[name]["hit_rate_at_10"])
        rows.append(
            {
                "gate": f"{section}:{name}:hit_rate_at_10",
                "challenger": candidate_value,
                "control": control_value,
                "tolerance": tolerance,
                "passed": candidate_value >= control_value - tolerance,
            }
        )
    return rows


def compare_reports(
    challenger: dict[str, Any],
    control: dict[str, Any],
    gates: dict[str, Any],
    paired: dict[str, float | int] | None = None,
) -> list[dict[str, Any]]:
    buying_challenger = challenger.get("scenario_metrics", {}).get("buying", {})
    buying_control = control.get("scenario_metrics", {}).get("buying", {})
    count = max(1, int(challenger["adaptive_runtime"]["trace_count"]))
    control_count = max(1, int(control["adaptive_runtime"]["trace_count"]))
    rows: list[dict[str, Any]] = [
        {
            "gate": "combined_recommended_technical_score",
            "challenger": _metric(challenger, "recommended_technical_score"),
            "control": _metric(control, "recommended_technical_score"),
            "passed": _metric(challenger, "recommended_technical_score")
            >= _metric(control, "recommended_technical_score")
            + float(gates["combined_score_min_delta"]),
        },
        {
            "gate": "buying_hit_rate_at_10",
            "challenger": float(buying_challenger.get("hit_rate_at_10", 0.0)),
            "control": float(buying_control.get("hit_rate_at_10", 0.0)),
            "passed": float(buying_challenger.get("hit_rate_at_10", 0.0))
            >= float(buying_control.get("hit_rate_at_10", 0.0))
            - float(gates["buying_hit_at_10_max_regression"]),
        },
        {
            "gate": "buying_mrr",
            "challenger": float(buying_challenger.get("mrr", 0.0)),
            "control": float(buying_control.get("mrr", 0.0)),
            "passed": float(buying_challenger.get("mrr", 0.0))
            >= float(buying_control.get("mrr", 0.0))
            - float(gates["buying_mrr_max_regression"]),
        },
        {
            "gate": "mttc",
            "challenger": _metric(challenger, "mttc"),
            "control": _metric(control, "mttc"),
            "passed": _metric(challenger, "mttc")
            <= _metric(control, "mttc") + float(gates["mttc_max_regression"]),
        },
        {
            "gate": "fallback_rate",
            "challenger": challenger["adaptive_runtime"]["fallback_count"] / count,
            "control": control["adaptive_runtime"]["fallback_count"] / control_count,
            "passed": challenger["adaptive_runtime"]["fallback_count"] / count
            <= control["adaptive_runtime"]["fallback_count"] / control_count
            + float(gates["fallback_rate_max_regression"]),
        },
        {
            "gate": "zero_output_constraint_violations",
            "challenger": challenger["adaptive_runtime"][
                "output_constraint_violation_count"
            ],
            "passed": not gates["require_zero_output_constraint_violations"]
            or challenger["adaptive_runtime"]["output_constraint_violation_count"] == 0,
        },
        {
            "gate": "zero_overload_trace_violations",
            "challenger": challenger["adaptive_runtime"][
                "overload_cutoff_trace_violations"
            ],
            "passed": not gates["require_zero_overload_trace_violations"]
            or challenger["adaptive_runtime"]["overload_cutoff_trace_violations"] == 0,
        },
    ]
    rows.extend(
        _slice_gates(
            challenger,
            control,
            "scenario_metrics",
            float(gates["scenario_hit_at_10_max_regression"]),
        )
    )
    rows.extend(
        _slice_gates(
            challenger,
            control,
            "source_metrics",
            float(gates["source_hit_at_10_max_regression"]),
        )
    )
    rows.extend(
        _slice_gates(
            challenger,
            control,
            "route_metrics",
            float(gates["route_hit_at_10_max_regression"]),
        )
    )
    if paired is not None:
        rows.extend(
            (
                {
                    "gate": "paired_lineage_cluster_mean",
                    "challenger": paired["mean_paired_delta"],
                    "passed": float(paired["mean_paired_delta"])
                    >= float(gates["paired_cluster_mean_min"]),
                },
                {
                    "gate": "paired_lineage_cluster_ci_lower",
                    "challenger": paired["ci_lower"],
                    "passed": float(paired["ci_lower"])
                    >= float(gates["paired_cluster_ci_lower_min"]),
                },
            )
        )
    return rows


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    source = report.get("metrics", report)
    metrics = {
        key: float(source[key])
        for key in (
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "recommended_technical_score",
        )
    }
    efficiency = float(
        source.get(
            "efficiency",
            max(0.0, min(1.0, (11.0 - metrics["mttc"]) / 10.0)),
        )
    )
    metrics.update(
        {
            "efficiency": efficiency,
            "normalized_efficiency": efficiency,
            "technical_score": metrics["recommended_technical_score"],
        }
    )
    return metrics


def _system_entry(
    *,
    system_id: str,
    role: str,
    champion_eligible: bool,
    report: dict[str, Any],
    report_path: str,
    config_path: str | None = None,
    config_sha256: str | None = None,
    note: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    metric_source = report.get("metrics", report)
    entry: dict[str, Any] = {
        "system_id": system_id,
        "role": role,
        "champion_eligible": champion_eligible,
        "metrics": _summary_metrics(report),
        "scenario_metrics": metric_source.get("scenario_metrics", {}),
        "source_metrics": report.get("source_metrics", {}),
        "sessions": report.get("sessions", []),
        "report_path": report_path,
    }
    if config_path is not None:
        entry["config_path"] = config_path
    if config_sha256 is not None:
        entry["config_sha256"] = config_sha256
    if note is not None:
        entry["note"] = note
    if display_name is not None:
        entry["display_name"] = display_name
    return entry


def _selection_key(
    frozen: dict[str, Any], report: dict[str, Any]
) -> tuple[float | int | str, ...]:
    runtime = report["adaptive_runtime"]
    fallback_rate = int(runtime["fallback_count"]) / max(
        1, int(runtime["trace_count"])
    )
    return (
        -_metric(report, "recommended_technical_score"),
        -_metric(report, "mrr"),
        -_metric(report, "hit_rate_at_10"),
        _metric(report, "mttc"),
        fallback_rate,
        int(frozen["development_rank"]),
        str(frozen["candidate_id"]),
    )


def build_fair_holdout_report(
    *,
    frozen: list[dict[str, Any]],
    selection_rule: dict[str, Any],
    reference_a: dict[str, Any],
    control: dict[str, Any],
    challengers: list[dict[str, Any]],
    reference_a_path: str,
    control_path: str,
    challenger_paths: list[str],
    control_config: str,
    control_config_sha256: str,
    challenger_config_canonical_sha256: list[str],
    gate_results: list[list[dict[str, Any]]],
    paired: list[dict[str, float | int]],
    pairwise: dict[str, dict[str, float | int]],
    receipt_path: str,
    proposal_report_path: str | None = None,
    proposal_report_sha256: str | None = None,
    gates_sha256: str | None = None,
) -> dict[str, Any]:
    if selection_rule.get("tie_break_order") != SELECTION_TIE_BREAK_ORDER:
        raise ValueError("final-selection tie-break order differs from frozen contract")
    if not bool(selection_rule.get("no_post_selection_tuning")):
        raise ValueError("final-selection contract must prohibit post-selection tuning")
    challenger_count = len(frozen)
    if not (
        1 <= challenger_count <= 3
        and challenger_count
        == len(challengers)
        == len(challenger_paths)
        == len(challenger_config_canonical_sha256)
        == len(gate_results)
        == len(paired)
    ):
        raise ValueError("final selection requires between one and three frozen D systems")
    all_reports = (reference_a, control, *challengers)
    contracts = [report.get("evaluation_contract") for report in all_reports]
    if not all(isinstance(contract, dict) for contract in contracts):
        raise ValueError("A/C/D reports must include the shared evaluation contract")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("A/C/D reports do not use an identical evaluation contract")
    ordered_ids = [
        [str(row["sample_id"]) for row in report.get("sessions", [])]
        for report in all_reports
    ]
    if len(ordered_ids[0]) != 550 or any(
        identifiers != ordered_ids[0] for identifiers in ordered_ids[1:]
    ):
        raise ValueError(
            "A/C/D final-selection reports must contain the same 550 ordered session IDs"
        )
    systems = [
        _system_entry(
            system_id="A_official_stateless_bm25",
            role="reference_baseline",
            champion_eligible=False,
            report=reference_a,
            report_path=reference_a_path,
            note="Frozen organizer reference; excluded from champion selection.",
            display_name="Organizer BM25 Starter",
        ),
        _system_entry(
            system_id="C_fixed_adaptive_architecture",
            role="promotion_control",
            champion_eligible=True,
            report=control,
            report_path=control_path,
            config_path=control_config,
            config_sha256=control_config_sha256,
            display_name="Fixed Adaptive Architecture",
        ),
    ]
    comparisons: list[dict[str, Any]] = []
    for index, (
        item,
        challenger,
        report_path,
        canonical_hash,
        gates,
        paired_stats,
    ) in enumerate(
        zip(
            frozen,
            challengers,
            challenger_paths,
            challenger_config_canonical_sha256,
            gate_results,
            paired,
            strict=True,
        ),
        start=1,
    ):
        system_id = (
            "GhostLab_Challenger"
            if challenger_count == 1
            else f"GhostLab_Challenger_{index}"
        )
        display_name = (
            "GhostLab Challenger"
            if challenger_count == 1
            else f"GhostLab Challenger {index}"
        )
        passed = all(bool(row["passed"]) for row in gates)
        systems.append(
            _system_entry(
                system_id=system_id,
                role="promotion_challenger",
                champion_eligible=True,
                report=challenger,
                report_path=report_path,
                config_path=str(item["config_path"]),
                config_sha256=canonical_hash,
                display_name=display_name,
            )
        )
        comparisons.append(
            {
                "control_system_id": "C_fixed_adaptive_architecture",
                "challenger_system_id": system_id,
                "candidate_id": item["candidate_id"],
                "development_rank": item["development_rank"],
                "all_gates_passed": passed,
                "gates": gates,
                "paired_lineage_cluster_statistics": paired_stats,
            }
        )
    eligible = [
        (item, challenger, comparison, canonical_hash, report_path)
        for item, challenger, comparison, canonical_hash, report_path in zip(
            frozen,
            challengers,
            comparisons,
            challenger_config_canonical_sha256,
            challenger_paths,
            strict=True,
        )
        if comparison["all_gates_passed"]
    ]
    selected = min(
        eligible, key=lambda value: _selection_key(value[0], value[1]), default=None
    )
    promoted = selected is not None
    selected_system_id = (
        str(selected[2]["challenger_system_id"])
        if selected is not None
        else "C_fixed_adaptive_architecture"
    )
    selected_candidate_id = (
        str(selected[0]["candidate_id"]) if selected is not None else None
    )
    compatibility_challenger = (
        {
            "candidate_id": selected_candidate_id,
            "config_path": selected[0]["config_path"],
            "config_sha256": selected[3],
            "config_file_sha256": selected[0]["config_sha256"],
            "report_path": selected[4],
            "metrics": _summary_metrics(selected[1]),
        }
        if selected is not None
        else None
    )
    return {
        "schema_version": 3,
        "evaluation_scope": "one_time_final_selection_set",
        "evaluation_partition": "final_selection",
        "final_selection_accessed": True,
        "final_selection_sample_count": 550,
        "sample_count": 550,
        "evaluation_contract": contracts[0],
        "system_count": 2 + challenger_count,
        "reference_count": 1,
        "challenger_count": challenger_count,
        "control_count": 1,
        "frozen_candidate_ids": [item["candidate_id"] for item in frozen],
        "frozen_inputs": {
            "proposal_report_path": proposal_report_path,
            "proposal_report_sha256": proposal_report_sha256,
            "gates_sha256": gates_sha256,
        },
        "comparison_semantics": {
            "same_ground": True,
            "same_ordered_session_ids": True,
            "same_catalog": True,
            "same_evaluator_contract": True,
            "reference_only_systems": [
                "A_official_stateless_bm25",
            ],
            "champion_selection_scope": (
                f"C versus {challenger_count} independently gated D system(s)"
            ),
            "selection_set_is_unbiased_holdout": False,
            "private_evaluation_is_unseen_generalization_test": True,
            "no_post_selection_tuning": True,
        },
        "systems": systems,
        "pairwise_lineage_cluster_statistics": pairwise,
        "promotion_comparisons": comparisons,
        "selection_rule": selection_rule,
        "selected_system_id": selected_system_id,
        "selected_candidate_id": selected_candidate_id,
        "all_gates_passed": promoted,
        "decision": "PROMOTE" if promoted else "RETAIN_CONTROL",
        # Compatibility fields retained for the explicit activation command.
        "challenger": compatibility_challenger,
        "control": {
            "config_path": control_config,
            "config_sha256": control_config_sha256,
            "report_path": control_path,
            "metrics": _summary_metrics(control),
        },
        "gates": selected[2]["gates"] if selected is not None else [],
        "paired_lineage_cluster_statistics": (
            selected[2]["paired_lineage_cluster_statistics"]
            if selected is not None
            else None
        ),
        "receipt": receipt_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate A/C plus one to three frozen D systems exactly once on the "
            "550-session final selection set"
        )
    )
    parser.add_argument(
        "--proposal-report", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    parser.add_argument(
        "--control-config",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument(
        "--gates", default="configs/evaluation/adaptive_holdout_gates_v2.json"
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_final_holdout.json"
    )
    parser.add_argument(
        "--receipt",
        default="artifacts/reports/adaptive_final_holdout.access_receipt.json",
    )
    args = parser.parse_args()

    proposal_path = ROOT / args.proposal_report
    manifest_path = ROOT / args.lineage_manifest
    gates_path = ROOT / args.gates
    output_path = ROOT / args.output
    receipt_path = ROOT / args.receipt
    if receipt_path.exists() or output_path.exists():
        raise RuntimeError("holdout was already accessed; refusing a second evaluation")
    proposal = _load(proposal_path)
    frozen = proposal.get("frozen_proposals")
    if not isinstance(frozen, list) or not 1 <= len(frozen) <= 3 or not all(
        isinstance(item, dict) for item in frozen
    ):
        raise TypeError(
            "development report must contain between one and three frozen proposals"
        )
    candidate_ids = [str(item["candidate_id"]) for item in frozen]
    if len(set(candidate_ids)) != len(frozen):
        raise ValueError("frozen proposal candidate IDs must be unique")
    challenger_paths = [ROOT / str(item["config_path"]) for item in frozen]
    manifest_hash = sha256_file(manifest_path)
    for item, challenger_path in zip(frozen, challenger_paths, strict=True):
        if sha256_file(challenger_path) != item["config_sha256"]:
            raise ValueError(
                f"frozen challenger config hash changed: {item['candidate_id']}"
            )
        canonical = load_adaptive_hybrid_config(challenger_path).canonical_hash()
        if canonical != item.get("config_canonical_sha256"):
            raise ValueError(
                f"frozen challenger canonical hash changed: {item['candidate_id']}"
            )
        if item.get("lineage_manifest_sha256") != manifest_hash:
            raise ValueError("frozen proposal and final-selection manifest do not match")
    frozen_dependencies = proposal.get("frozen_dependencies")
    if not isinstance(frozen_dependencies, dict):
        raise TypeError("development report has no frozen comparison dependencies")
    frozen_dependencies = verify_frozen_holdout_inputs(
        frozen_dependencies,
        control_config_path=ROOT / args.control_config,
        reference_a_path=ROOT / "baseline/official_reference.py",
        gates_path=gates_path,
    )
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    corpus = load_adaptive_training_corpus(ROOT, datasets)
    manifest = load_lineage_manifest(manifest_path, corpus)
    if len(manifest.holdout_ids) != 550:
        raise ValueError("final holdout must contain exactly 550 sessions")
    development_evidence = [
        {
            "candidate_id": item["candidate_id"],
            **verify_development_evidence(
                proposal,
                challenger_path,
                manifest_hash,
                manifest.holdout_ids,
            ),
        }
        for item, challenger_path in zip(frozen, challenger_paths, strict=True)
    ]
    receipt = {
        "schema_version": 3,
        "status": "started",
        "protocol": "one_time_final_selection_set",
        "frozen_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "development_rank": item["development_rank"],
                "config_path": item["config_path"],
                "config_file_sha256": item["config_sha256"],
                "config_canonical_sha256": item["config_canonical_sha256"],
            }
            for item in frozen
        ],
        "control_config_sha256": frozen_dependencies[
            "control_config_file_sha256"
        ],
        "control_config_canonical_sha256": frozen_dependencies[
            "control_config_canonical_sha256"
        ],
        "reference_a_implementation_sha256": frozen_dependencies[
            "reference_a_implementation_sha256"
        ],
        "evaluated_systems": [
            "Organizer BM25 Starter",
            "Fixed Adaptive Architecture",
            *(
                ["GhostLab Challenger"]
                if len(frozen) == 1
                else [
                    f"GhostLab Challenger {index}"
                    for index in range(1, len(frozen) + 1)
                ]
            ),
        ],
        "lineage_manifest_sha256": sha256_file(manifest_path),
        "gates_sha256": frozen_dependencies["gates_sha256"],
        "proposal_report_path": proposal_path.relative_to(ROOT).as_posix(),
        "proposal_report_sha256": sha256_file(proposal_path),
        "holdout_ids_sha256": hashlib.sha256(
            "\n".join(sorted(manifest.holdout_ids)).encode()
        ).hexdigest(),
        "holdout_sample_count": 550,
        "development_evidence": development_evidence,
    }
    _write_json(receipt_path, receipt)
    run_dir = output_path.parent / "adaptive_final_holdout_runs"
    reference_a_output = run_dir / "reference_a.json"
    control_output = run_dir / "control.json"
    challenger_outputs = (
        [run_dir / "ghostlab_challenger.json"]
        if len(frozen) == 1
        else [
            run_dir / f"ghostlab_challenger_{index}_{item['candidate_id']}.json"
            for index, item in enumerate(frozen, start=1)
        ]
    )
    holdout_corpus = subset_corpus(corpus, manifest, "holdout")
    holdout_samples = [
        holdout_corpus.samples[sample_id]
        for sample_id in sorted(holdout_corpus.samples)
    ]
    reference_a = evaluate_reference_a(
        samples=holdout_samples,
        origins=holdout_corpus.origins,
        catalog_path=ROOT / "data/catalog.jsonl",
        partition="holdout",
        holdout_accessed=True,
    )
    _write_json(reference_a_output, reference_a)
    _run_evaluation(
        args.control_config,
        str(control_output.relative_to(ROOT)),
        datasets,
        args.lineage_manifest,
    )
    for challenger_path, challenger_output in zip(
        challenger_paths, challenger_outputs, strict=True
    ):
        _run_evaluation(
            str(challenger_path.relative_to(ROOT)),
            str(challenger_output.relative_to(ROOT)),
            datasets,
            args.lineage_manifest,
        )
    control = _load(control_output)
    challengers = [_load(path) for path in challenger_outputs]
    gates = _load(gates_path)
    if gates.get("evaluation_scope") != "one_time_final_selection_set":
        raise ValueError("gates are not scoped to the one-time final selection set")
    minimum_challengers = int(gates.get("minimum_challenger_count", 0))
    maximum_challengers = int(gates.get("maximum_challenger_count", 0))
    if not (
        minimum_challengers == 1
        and maximum_challengers == 3
        and minimum_challengers <= len(frozen) <= maximum_challengers
    ):
        raise ValueError(
            "gates must allow the frozen challenger count within the one-to-three range"
        )
    if gates.get("selection_tie_break_order") != SELECTION_TIE_BREAK_ORDER:
        raise ValueError("gates contain an unsupported final-selection tie-break order")
    if not bool(gates.get("no_post_selection_tuning")):
        raise ValueError("gates must prohibit post-selection tuning")
    if proposal.get("selection_rule", {}).get("tie_break_order") != gates.get(
        "selection_tie_break_order"
    ):
        raise ValueError("frozen proposal and gates use different tie-break orders")
    holdout_group_by_sample = {
        sample_id: manifest.group_by_sample[sample_id]
        for sample_id in manifest.holdout_ids
    }
    paired = [
        paired_cluster_statistics(challenger, control, holdout_group_by_sample)
        for challenger in challengers
    ]
    pairwise = {
        "C_minus_A": paired_cluster_statistics(
            control, reference_a, holdout_group_by_sample
        ),
        **{
            (
                "GhostLab_Challenger_minus_C"
                if len(frozen) == 1
                else f"GhostLab_Challenger_{index}_minus_C"
            ): stats
            for index, stats in enumerate(paired, start=1)
        },
    }
    gate_results = [
        compare_reports(challenger, control, gates, stats)
        for challenger, stats in zip(challengers, paired, strict=True)
    ]
    receipt_candidates = cast(
        list[dict[str, Any]], receipt["frozen_candidates"]
    )
    report = build_fair_holdout_report(
        frozen=frozen,
        selection_rule=dict(proposal["selection_rule"]),
        reference_a=reference_a,
        control=control,
        challengers=challengers,
        reference_a_path=str(reference_a_output.relative_to(ROOT)),
        control_path=str(control_output.relative_to(ROOT)),
        challenger_paths=[str(path.relative_to(ROOT)) for path in challenger_outputs],
        control_config=args.control_config,
        control_config_sha256=str(receipt["control_config_sha256"]),
        challenger_config_canonical_sha256=[
            str(item["config_canonical_sha256"])
            for item in receipt_candidates
        ],
        gate_results=gate_results,
        paired=paired,
        pairwise=pairwise,
        receipt_path=str(receipt_path.relative_to(ROOT)),
        proposal_report_path=str(proposal_path.relative_to(ROOT)),
        proposal_report_sha256=str(receipt["proposal_report_sha256"]),
        gates_sha256=str(receipt["gates_sha256"]),
    )
    _write_json(output_path, report)
    _write_json(
        receipt_path,
        {
            **receipt,
            "status": "complete",
            "decision": report["decision"],
            "selected_system_id": report["selected_system_id"],
            "selected_candidate_id": report["selected_candidate_id"],
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
