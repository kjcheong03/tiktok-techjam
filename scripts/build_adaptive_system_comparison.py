from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ghostlab.research.replay import session_reward

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("metrics", report)
    return {
        "hit_rate_at_10": float(source["hit_rate_at_10"]),
        "mrr": float(source["mrr"]),
        "mttc": float(source["mttc"]),
        "recommended_technical_score": float(source["recommended_technical_score"]),
    }


def _session_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sessions = report.get("sessions")
    if not isinstance(sessions, list):
        raise TypeError("comparison input has no per-session results")
    return {str(item["sample_id"]): item for item in sessions}


def _session_ids(report: dict[str, Any]) -> list[str]:
    sessions = report.get("sessions")
    if not isinstance(sessions, list):
        raise TypeError("comparison input has no per-session results")
    return [str(item["sample_id"]) for item in sessions]


def _paired_delta(candidate: dict[str, Any], control: dict[str, Any]) -> float:
    candidate_rows = _session_index(candidate)
    control_rows = _session_index(control)
    if set(candidate_rows) != set(control_rows):
        raise ValueError("A/C/finalist reports do not contain identical sample IDs")
    return statistics.fmean(
        session_reward(candidate_rows[sample_id])
        - session_reward(control_rows[sample_id])
        for sample_id in sorted(candidate_rows)
    )


def build_comparison(
    report_a: dict[str, Any],
    report_c: dict[str, Any],
    top_three: dict[str, Any] | None = None,
    finalist_evaluations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_ids = [_session_ids(report) for report in (report_a, report_c)]
    if any(identifiers != ordered_ids[0] for identifiers in ordered_ids[1:]):
        raise ValueError("A/C reports do not contain identical ordered sample IDs")
    if len(ordered_ids[0]) != 1650:
        raise ValueError(
            f"final comparison requires 1650 development IDs, got {len(ordered_ids[0])}"
        )
    contracts = [
        report.get("evaluation_contract") for report in (report_a, report_c)
    ]
    if not all(isinstance(contract, dict) for contract in contracts):
        raise ValueError("A/C reports must include the shared evaluation contract")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("A/C reports do not use an identical evaluation contract")

    systems: list[dict[str, Any]] = [
        {
            "system_id": "A_official_stateless_bm25",
            "role": "explanatory_baseline",
            "champion_eligible": False,
            "metrics": _metrics(report_a),
            "scenario_metrics": report_a.get("metrics", report_a).get(
                "scenario_metrics", {}
            ),
            "source_metrics": report_a.get("source_metrics", {}),
            "sessions": report_a["sessions"],
        },
        {
            "system_id": "C_fixed_adaptive_architecture",
            "role": "ghostlab_control",
            "champion_eligible": True,
            "metrics": _metrics(report_c),
            "scenario_metrics": report_c.get("scenario_metrics", {}),
            "source_metrics": report_c.get("source_metrics", {}),
            "sessions": report_c["sessions"],
        },
    ]
    finalists: list[dict[str, Any]] = []
    if top_three is not None:
        if finalist_evaluations is None:
            raise ValueError(
                "Top-3 comparison requires matched development finalist evaluations"
            )
        evaluated = {
            str(item["candidate_id"]): item
            for item in finalist_evaluations.get("evaluations", [])
        }
        for item in top_three.get("finalists", []):
            candidate_id = str(item["candidate_id"])
            matched = evaluated.get(candidate_id)
            if matched is None:
                raise ValueError(f"missing matched evaluation for {candidate_id}")
            if matched.get("config_path") != item.get("config_path") or matched.get(
                "config_sha256"
            ) != item.get("config_sha256"):
                raise ValueError(
                    f"matched evaluation config differs for {candidate_id}"
                )
            matched_report = matched.get("report")
            if not isinstance(matched_report, dict):
                raise TypeError(
                    f"matched evaluation report is invalid for {candidate_id}"
                )
            if _session_ids(matched_report) != ordered_ids[0]:
                raise ValueError(f"matched evaluation order differs for {candidate_id}")
            if matched_report.get("evaluation_contract") != contracts[0]:
                raise ValueError(
                    f"matched evaluation contract differs for {candidate_id}"
                )
            selection_metrics = item["metrics"]
            finalists.append(
                {
                    "system_id": f"D{item['rank']}_{item['candidate_id']}",
                    "candidate_id": candidate_id,
                    "role": "ghostlab_challenger",
                    "champion_eligible": bool(item["promotion_eligible"]),
                    "techniques": item.get("techniques", []),
                    "metrics": _metrics(matched_report),
                    "scenario_metrics": matched_report.get("scenario_metrics", {}),
                    "source_metrics": matched_report.get("source_metrics", {}),
                    "sessions": matched_report["sessions"],
                    "paired_reward_delta_vs_c": _paired_delta(matched_report, report_c),
                    "ghostlab_selection_metrics": selection_metrics,
                }
            )
    systems.extend(finalists)
    paired_deltas = {
        "C_minus_A": _paired_delta(report_c, report_a),
    }
    paired_deltas.update(
        {
            f"{item['system_id']}_minus_C": item["paired_reward_delta_vs_c"]
            for item in finalists
        }
    )
    return {
        "schema_version": 1,
        "evaluation_partition": "development",
        "sample_count": len(ordered_ids[0]),
        "holdout_accessed": False,
        "evaluation_contract": contracts[0],
        "comparison_semantics": {
            "same_ground": True,
            "same_ordered_session_ids": True,
            "same_catalog": True,
            "same_evaluator_contract": True,
            "A": "official stateless organizer BM25",
            "C": "fixed compulsory adaptive 1A-3B architecture before GhostLab search",
            "D": "GhostLab challengers built on C",
            "champion_selection_scope": "C versus D only",
            "A_purpose": "organizer reference baseline only",
            "finalist_purpose": (
                "development finalists; freeze every eligible finalist up to three "
                "before final selection"
            ),
        },
        "paired_reward_deltas": paired_deltas,
        "ghostlab_status": (
            "top_three_available" if finalists else "challengers_not_run"
        ),
        "systems": systems,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Adaptive system comparison",
        "",
        (
            "All rows use the same 1,650-session lineage-safe development partition. "
            "The 550-session final selection set was not accessed."
        ),
        "",
        "| System | Role | Champion eligible | Hit@10 | MRR | MTTC | Technical score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["systems"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['system_id']} | {item['role']} | "
            f"{'yes' if item['champion_eligible'] else 'no'} | "
            f"{metrics['hit_rate_at_10']:.6f} | {metrics['mrr']:.6f} | "
            f"{metrics['mttc']:.6f} | "
            f"{metrics['recommended_technical_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired development deltas",
            "",
            *[
                f"- `{name}`: {value:+.6f}"
                for name, value in report["paired_reward_deltas"].items()
            ],
            "",
            (
                "A explains the gain over the organizer starter but cannot become champion. GhostLab "
                "promotion compares the fixed C control only with D challengers."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the unified A/C and optional D1-D3 comparison"
    )
    parser.add_argument(
        "--report-a",
        default="artifacts/reports/adaptive_baseline_a_development_1650.json",
    )
    parser.add_argument(
        "--report-c", default="artifacts/reports/adaptive_hybrid_development_1650.json"
    )
    parser.add_argument(
        "--top-three", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    parser.add_argument(
        "--finalist-evaluations",
        default="artifacts/reports/adaptive_finalist_development_evaluations.json",
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_system_comparison_1650.json"
    )
    parser.add_argument(
        "--markdown",
        help="optional human-readable Markdown output; JSON is canonical",
    )
    args = parser.parse_args()
    top_three_path = ROOT / args.top_three
    finalist_evaluations_path = ROOT / args.finalist_evaluations
    report = build_comparison(
        _load(ROOT / args.report_a),
        _load(ROOT / args.report_c),
        _load(top_three_path) if top_three_path.is_file() else None,
        (
            _load(finalist_evaluations_path)
            if top_three_path.is_file() and finalist_evaluations_path.is_file()
            else None
        ),
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_output = None
    if args.markdown:
        markdown = ROOT / args.markdown
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(report), encoding="utf-8")
        markdown_output = args.markdown
    print(
        json.dumps(
            {"output": args.output, "markdown": markdown_output, **report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
