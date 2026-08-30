from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from ghostlab.evaluation.statistics import bootstrap_mean_interval
from ghostlab.optimization.racing import lineage_cluster_means
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.selected import sha256_file
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import load_lineage_manifest, subset_corpus
from scripts.evaluate_adaptive_reference_baselines import evaluate_reference_systems

ROOT = Path(__file__).resolve().parents[1]
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
            "gate": "zero_confirmed_target_removals",
            "challenger": challenger["target_survival_audit"][
                "confirmed_target_removal_count"
            ],
            "passed": not gates["require_zero_confirmed_target_removals"]
            or challenger["target_survival_audit"]["confirmed_target_removal_count"]
            == 0,
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
    return {
        key: float(source[key])
        for key in (
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "recommended_technical_score",
        )
    }


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
    return entry


def build_fair_holdout_report(
    *,
    frozen: dict[str, Any],
    reference_a: dict[str, Any],
    reference_b: dict[str, Any],
    control: dict[str, Any],
    challenger: dict[str, Any],
    reference_a_path: str,
    reference_b_path: str,
    control_path: str,
    challenger_path: str,
    control_config: str,
    control_config_sha256: str,
    challenger_config_canonical_sha256: str,
    gate_results: list[dict[str, Any]],
    paired: dict[str, float | int],
    pairwise: dict[str, dict[str, float | int]],
    receipt_path: str,
) -> dict[str, Any]:
    contracts = [
        report.get("evaluation_contract")
        for report in (reference_a, reference_b, control, challenger)
    ]
    if not all(isinstance(contract, dict) for contract in contracts):
        raise ValueError("A/B/C/D reports must include the shared evaluation contract")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("A/B/C/D reports do not use an identical evaluation contract")
    ordered_ids = [
        [str(row["sample_id"]) for row in report.get("sessions", [])]
        for report in (reference_a, reference_b, control, challenger)
    ]
    if len(ordered_ids[0]) != 550 or any(
        identifiers != ordered_ids[0] for identifiers in ordered_ids[1:]
    ):
        raise ValueError(
            "A/B/C/D holdout reports must contain the same 550 ordered session IDs"
        )
    passed = all(bool(item["passed"]) for item in gate_results)
    systems = [
        _system_entry(
            system_id="A_official_stateless_bm25",
            role="reference_baseline",
            champion_eligible=False,
            report=reference_a,
            report_path=reference_a_path,
            note="Frozen organizer reference; excluded from champion selection.",
        ),
        _system_entry(
            system_id="B_state_baseline_v2_tagged_best",
            role="reference_baseline",
            champion_eligible=False,
            report=reference_b,
            report_path=reference_b_path,
            note=(
                "Frozen exact-parity State V2 reference. fixed_other is "
                "simulator-sensitive and excluded from champion selection."
            ),
        ),
        _system_entry(
            system_id="C_fixed_adaptive_architecture",
            role="promotion_control",
            champion_eligible=True,
            report=control,
            report_path=control_path,
            config_path=control_config,
            config_sha256=control_config_sha256,
        ),
        _system_entry(
            system_id=f"D_{frozen['candidate_id']}",
            role="promotion_challenger",
            champion_eligible=True,
            report=challenger,
            report_path=challenger_path,
            config_path=str(frozen["config_path"]),
            config_sha256=challenger_config_canonical_sha256,
        ),
    ]
    return {
        "schema_version": 2,
        "evaluation_scope": "single_use_untouched_holdout",
        "evaluation_partition": "holdout",
        "holdout_accessed": True,
        "holdout_sample_count": 550,
        "sample_count": 550,
        "evaluation_contract": contracts[0],
        "system_count": 4,
        "reference_count": 2,
        "challenger_count": 1,
        "control_count": 1,
        "frozen_candidate_id": frozen["candidate_id"],
        "comparison_semantics": {
            "same_ground": True,
            "same_ordered_session_ids": True,
            "same_catalog": True,
            "same_evaluator_contract": True,
            "reference_only_systems": [
                "A_official_stateless_bm25",
                "B_state_baseline_v2_tagged_best",
            ],
            "champion_selection_scope": "C versus D only",
            "no_post_holdout_tuning": True,
        },
        "systems": systems,
        "pairwise_lineage_cluster_statistics": pairwise,
        "promotion_comparison": {
            "control_system_id": "C_fixed_adaptive_architecture",
            "challenger_system_id": f"D_{frozen['candidate_id']}",
            "gates": gate_results,
            "paired_lineage_cluster_statistics": paired,
        },
        "all_gates_passed": passed,
        "decision": "PROMOTE" if passed else "RETAIN_CONTROL",
        # Compatibility fields retained for the explicit activation command.
        "challenger": {
            "config_path": frozen["config_path"],
            "config_sha256": challenger_config_canonical_sha256,
            "config_file_sha256": frozen["config_sha256"],
            "report_path": challenger_path,
            "metrics": _summary_metrics(challenger),
        },
        "control": {
            "config_path": control_config,
            "config_sha256": control_config_sha256,
            "report_path": control_path,
            "metrics": _summary_metrics(control),
        },
        "gates": gate_results,
        "paired_lineage_cluster_statistics": paired,
        "receipt": receipt_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen A/B references plus one frozen C control and one frozen "
            "D challenger once on holdout; promotion remains C-versus-D only"
        )
    )
    parser.add_argument(
        "--proposal-report", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    parser.add_argument(
        "--control-config",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument(
        "--state-reference-config",
        default="configs/suites/state_baseline_v2_other.json",
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument(
        "--lineage-manifest",
        default="data/splits/adaptive_hybrid_lineage_75_25_v1.json",
    )
    parser.add_argument(
        "--gates", default="configs/evaluation/adaptive_holdout_gates_v1.json"
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
    frozen = proposal.get("frozen_proposal")
    if not isinstance(frozen, dict):
        raise TypeError("development report has no single frozen proposal")
    challenger_path = ROOT / str(frozen["config_path"])
    if sha256_file(challenger_path) != frozen["config_sha256"]:
        raise ValueError("frozen challenger config hash changed")
    if frozen.get("lineage_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("frozen proposal and holdout manifest do not match")
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    corpus = load_adaptive_training_corpus(ROOT, datasets)
    manifest = load_lineage_manifest(manifest_path, corpus)
    if len(manifest.holdout_ids) != 550:
        raise ValueError("final holdout must contain exactly 550 sessions")
    development_evidence = verify_development_evidence(
        proposal,
        challenger_path,
        sha256_file(manifest_path),
        manifest.holdout_ids,
    )
    receipt = {
        "schema_version": 2,
        "status": "started",
        "frozen_candidate_id": frozen["candidate_id"],
        "challenger_config_file_sha256": frozen["config_sha256"],
        "challenger_config_canonical_sha256": load_adaptive_hybrid_config(
            challenger_path
        ).canonical_hash(),
        "control_config_sha256": sha256_file(ROOT / args.control_config),
        "reference_a_implementation_sha256": sha256_file(
            ROOT / "baseline/official_reference.py"
        ),
        "reference_b_config_sha256": sha256_file(ROOT / args.state_reference_config),
        "lineage_manifest_sha256": sha256_file(manifest_path),
        "gates_sha256": sha256_file(gates_path),
        "holdout_ids_sha256": hashlib.sha256(
            "\n".join(sorted(manifest.holdout_ids)).encode()
        ).hexdigest(),
        "holdout_sample_count": 550,
        "development_evidence": development_evidence,
    }
    _write_json(receipt_path, receipt)
    run_dir = output_path.parent / "adaptive_final_holdout_runs"
    reference_a_output = run_dir / "reference_a.json"
    reference_b_output = run_dir / "reference_b.json"
    control_output = run_dir / "control.json"
    challenger_output = run_dir / "challenger.json"
    holdout_corpus = subset_corpus(corpus, manifest, "holdout")
    holdout_samples = [
        holdout_corpus.samples[sample_id]
        for sample_id in sorted(holdout_corpus.samples)
    ]
    reference_a, reference_b = evaluate_reference_systems(
        samples=holdout_samples,
        origins=holdout_corpus.origins,
        catalog_path=ROOT / "data/catalog.jsonl",
        state_config_path=ROOT / args.state_reference_config,
        partition="holdout",
        holdout_accessed=True,
    )
    _write_json(reference_a_output, reference_a)
    _write_json(reference_b_output, reference_b)
    _run_evaluation(
        args.control_config,
        str(control_output.relative_to(ROOT)),
        datasets,
        args.lineage_manifest,
    )
    _run_evaluation(
        str(challenger_path.relative_to(ROOT)),
        str(challenger_output.relative_to(ROOT)),
        datasets,
        args.lineage_manifest,
    )
    control = _load(control_output)
    challenger = _load(challenger_output)
    gates = _load(gates_path)
    holdout_group_by_sample = {
        sample_id: manifest.group_by_sample[sample_id]
        for sample_id in manifest.holdout_ids
    }
    paired = paired_cluster_statistics(challenger, control, holdout_group_by_sample)
    pairwise = {
        "B_minus_A": paired_cluster_statistics(
            reference_b, reference_a, holdout_group_by_sample
        ),
        "C_minus_B": paired_cluster_statistics(
            control, reference_b, holdout_group_by_sample
        ),
        "D_minus_C": paired,
    }
    gate_results = compare_reports(challenger, control, gates, paired)
    report = build_fair_holdout_report(
        frozen=frozen,
        reference_a=reference_a,
        reference_b=reference_b,
        control=control,
        challenger=challenger,
        reference_a_path=str(reference_a_output.relative_to(ROOT)),
        reference_b_path=str(reference_b_output.relative_to(ROOT)),
        control_path=str(control_output.relative_to(ROOT)),
        challenger_path=str(challenger_output.relative_to(ROOT)),
        control_config=args.control_config,
        control_config_sha256=str(receipt["control_config_sha256"]),
        challenger_config_canonical_sha256=str(
            receipt["challenger_config_canonical_sha256"]
        ),
        gate_results=gate_results,
        paired=paired,
        pairwise=pairwise,
        receipt_path=str(receipt_path.relative_to(ROOT)),
    )
    _write_json(output_path, report)
    _write_json(
        receipt_path, {**receipt, "status": "complete", "decision": report["decision"]}
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
