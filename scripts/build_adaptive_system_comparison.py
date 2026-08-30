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


def _paired_delta(candidate: dict[str, Any], control: dict[str, Any]) -> float:
    candidate_rows = _session_index(candidate)
    control_rows = _session_index(control)
    if set(candidate_rows) != set(control_rows):
        raise ValueError("A/B/C reports do not contain identical sample IDs")
    return statistics.fmean(
        session_reward(candidate_rows[sample_id])
        - session_reward(control_rows[sample_id])
        for sample_id in sorted(candidate_rows)
    )


def build_comparison(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    report_c: dict[str, Any],
    top_three: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indexes = [_session_index(report) for report in (report_a, report_b, report_c)]
    sample_ids = set(indexes[0])
    if any(set(index) != sample_ids for index in indexes[1:]):
        raise ValueError("A/B/C reports do not contain identical sample IDs")
    if len(sample_ids) != 1650:
        raise ValueError(
            f"final comparison requires 1650 development IDs, got {len(sample_ids)}"
        )

    systems: list[dict[str, Any]] = [
        {
            "system_id": "A_official_stateless_bm25",
            "role": "explanatory_baseline",
            "champion_eligible": False,
            "metrics": _metrics(report_a),
        },
        {
            "system_id": "B_state_baseline_v2_tagged_best",
            "role": "explanatory_baseline",
            "champion_eligible": False,
            "metrics": _metrics(report_b),
        },
        {
            "system_id": "C_fixed_adaptive_architecture",
            "role": "ghostlab_control",
            "champion_eligible": True,
            "metrics": _metrics(report_c),
        },
    ]
    finalists: list[dict[str, Any]] = []
    if top_three is not None:
        for item in top_three.get("finalists", []):
            raw = item["metrics"]
            finalists.append(
                {
                    "system_id": f"D{item['rank']}_{item['candidate_id']}",
                    "candidate_id": item["candidate_id"],
                    "role": "ghostlab_challenger",
                    "champion_eligible": bool(item["promotion_eligible"]),
                    "techniques": item.get("techniques", []),
                    "metrics": {
                        "hit_rate_at_10": float(raw["hit_rate_at_10"]),
                        "mrr": float(raw["mrr"]),
                        "mttc": float(raw["mttc"]),
                        "recommended_technical_score": float(raw["score"]),
                    },
                    "paired_reward_delta_vs_c": float(raw["mean_paired_delta"]),
                }
            )
    systems.extend(finalists)
    return {
        "schema_version": 1,
        "evaluation_partition": "development",
        "sample_count": len(sample_ids),
        "holdout_accessed": False,
        "comparison_semantics": {
            "A": "official stateless organizer BM25",
            "B": "tagged-best State Baseline V2 native exact-parity reproduction",
            "C": "fixed compulsory adaptive 1A-3B architecture before GhostLab search",
            "D": "GhostLab challengers built on C",
            "champion_selection_scope": "C versus D only",
            "A_and_B_purpose": "explanatory reference baselines only",
            "top_three_purpose": (
                "development finalists; freeze exactly one before one-time holdout"
            ),
        },
        "paired_reward_deltas": {
            "B_minus_A": _paired_delta(report_b, report_a),
            "C_minus_A": _paired_delta(report_c, report_a),
            "C_minus_B": _paired_delta(report_c, report_b),
        },
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
            "The untouched 550-session holdout was not accessed."
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
                "A and B explain system gains but cannot become champion. GhostLab "
                "promotion compares the fixed C control only with D challengers."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the unified A/B/C and optional D1-D3 comparison"
    )
    parser.add_argument(
        "--report-a",
        default="artifacts/reports/adaptive_baseline_a_development_1650.json",
    )
    parser.add_argument(
        "--report-b",
        default="artifacts/reports/adaptive_baseline_b_development_1650.json",
    )
    parser.add_argument(
        "--report-c", default="artifacts/reports/adaptive_hybrid_development_1650.json"
    )
    parser.add_argument(
        "--top-three", default="artifacts/reports/adaptive_hybrid_top3.json"
    )
    parser.add_argument(
        "--output", default="artifacts/reports/adaptive_system_comparison_1650.json"
    )
    parser.add_argument(
        "--markdown", default="artifacts/reports/adaptive_system_comparison_1650.md"
    )
    args = parser.parse_args()
    top_three_path = ROOT / args.top_three
    report = build_comparison(
        _load(ROOT / args.report_a),
        _load(ROOT / args.report_b),
        _load(ROOT / args.report_c),
        _load(top_three_path) if top_three_path.is_file() else None,
    )
    output = ROOT / args.output
    markdown = ROOT / args.markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"output": args.output, "markdown": args.markdown, **report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
