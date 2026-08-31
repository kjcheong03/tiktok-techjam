from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _system(report: dict[str, Any], system_id: str) -> dict[str, Any]:
    systems = report.get("systems")
    if not isinstance(systems, list):
        raise TypeError("final-selection report has no systems list")
    for item in systems:
        if isinstance(item, dict) and item.get("system_id") == system_id:
            return item
    raise ValueError(f"final-selection report is missing system {system_id}")


def build_summary(
    report: dict[str, Any],
    active: dict[str, Any],
    *,
    project_root: Path,
) -> str:
    if report.get("evaluation_scope") != "one_time_final_selection_set":
        raise ValueError("expected the one-time final-selection report")

    active_path_value = active.get("preset_path")
    active_sha = active.get("preset_sha256")
    if not isinstance(active_path_value, str) or not isinstance(active_sha, str):
        raise TypeError("active candidate is missing its preset path or SHA-256")
    active_path = project_root / active_path_value
    if not active_path.is_file():
        raise FileNotFoundError(f"active champion preset is missing: {active_path}")
    actual_sha = _sha256(active_path)
    if actual_sha != active_sha:
        raise ValueError(
            "active champion SHA-256 mismatch: "
            f"expected {active_sha}, calculated {actual_sha}"
        )

    rows = (
        ("A", "Organizer BM25", _system(report, "A_official_stateless_bm25")),
        (
            "C",
            "Fixed Adaptive Architecture",
            _system(report, "C_fixed_adaptive_architecture"),
        ),
        ("D", "GhostLab Champion", _system(report, "GhostLab_Challenger")),
    )
    champion = rows[-1][2]
    if champion.get("config_path") != active_path_value:
        raise ValueError(
            "active champion does not match D in the final-selection report"
        )

    sample_count = int(report.get("sample_count", 0))
    lines = [
        f"FINAL RESULTS — {sample_count} ONE-TIME FINAL-SELECTION SESSIONS",
        "",
        f"{'System':<35} {'Hit@10':>8} {'MRR':>8} {'MTTC':>8} {'Score':>9}",
        f"{'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9}",
    ]
    for label, name, item in rows:
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError(f"system {label} has no metrics")
        lines.append(
            f"{label + ' — ' + name:<35} "
            f"{float(metrics['hit_rate_at_10']):>8.4f} "
            f"{float(metrics['mrr']):>8.4f} "
            f"{float(metrics['mttc']):>8.4f} "
            f"{float(metrics['technical_score']):>9.4f}"
        )

    candidate_id = active_path.stem.removeprefix("rank_1_")
    lines.extend(
        [
            "",
            "Active champion: D — GhostLab Champion",
            f"Candidate: {candidate_id}",
            f"Configuration: {active_path_value}",
            f"Verified SHA-256: {active_sha}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the frozen A/C/D final-selection results and active champion"
    )
    parser.add_argument(
        "--report",
        default="artifacts/reports/adaptive_final_holdout.json",
        help="recorded one-time final-selection report",
    )
    parser.add_argument(
        "--active-candidate",
        default="configs/active_candidate.json",
        help="active champion pointer",
    )
    args = parser.parse_args()
    print(
        build_summary(
            _load_json(ROOT / args.report),
            _load_json(ROOT / args.active_candidate),
            project_root=ROOT,
        )
    )


if __name__ == "__main__":
    main()
