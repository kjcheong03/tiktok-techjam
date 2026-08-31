from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live progress and matched-score monitor for adaptive GhostLab"
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/campaigns/adaptive_hybrid_1650_v1/checkpoint.json",
    )
    parser.add_argument(
        "--log",
        default="artifacts/logs/adaptive_hybrid_warm_8h_resume.log",
    )
    parser.add_argument(
        "--pipeline-checkpoint",
        default="artifacts/campaigns/adaptive_hybrid_pipeline/checkpoint.json",
    )
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _budgets(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    command = _read_json(path).get("stages", {}).get("campaign", {}).get("command", [])
    flags = {
        "--candidate-limit": "f0",
        "--f1-candidates": "f1",
        "--f2-candidates": "f2",
    }
    result: dict[str, int] = {}
    for flag, phase in flags.items():
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                result[phase] = int(command[index + 1])
    return result


def _active_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.startswith("{") or '"event": "evaluation_' not in raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("event") in {
            "evaluation_started",
            "evaluation_finished",
            "evaluation_resumed",
        }:
            latest = event
    if latest and latest.get("event") == "evaluation_started":
        return latest
    return None


def _metric_line(label: str, record: dict[str, Any] | None) -> str:
    if record is None:
        return f"{label}: not evaluated in this phase"
    candidate = record.get("candidate", {})
    return (
        f"{label}: {candidate.get('candidate_id')}\n"
        f"  score={float(record.get('score', 0.0)):.6f}  "
        f"Hit@10={float(record.get('hit_rate_at_10', 0.0)):.4f}  "
        f"MRR={float(record.get('mrr', 0.0)):.4f}  "
        f"MTTC={float(record.get('mttc', 0.0)):.3f}  "
        f"p95={float(record.get('latency_p95_ms', 0.0)):.1f}ms"
    )


def render(
    checkpoint_path: Path,
    log_path: Path,
    pipeline_checkpoint_path: Path,
) -> str:
    checkpoint = _read_json(checkpoint_path)
    entries = checkpoint.get("evaluations", {})
    by_phase: dict[str, list[dict[str, Any]]] = {"f0": [], "f1": [], "f2": []}
    for key, record in entries.items():
        phase = str(key).split(":", 1)[0]
        if phase in by_phase:
            by_phase[phase].append(record)
    active = _active_event(log_path)
    if active is not None:
        phase = str(active.get("fidelity", "f0"))
    else:
        phase = next(
            (item for item in ("f2", "f1", "f0") if by_phase[item]),
            "f0",
        )
    records = by_phase[phase]
    control = next(
        (
            item
            for item in records
            if item.get("candidate", {}).get("generation") == "control"
        ),
        None,
    )
    challengers = [
        item
        for item in records
        if item.get("candidate", {}).get("generation") != "control"
    ]
    best = max(
        challengers, key=lambda item: float(item.get("score", 0.0)), default=None
    )
    budgets = _budgets(pipeline_checkpoint_path)
    counts = Counter(key.split(":", 1)[0] for key in entries)
    progress = "  ".join(
        f"{item.upper()}={counts[item]}/{budgets.get(item, '?')}"
        for item in ("f0", "f1", "f2")
    )
    active_text = (
        f"{active.get('candidate_id')} ({active.get('fidelity')}, "
        f"ordinal {active.get('ordinal')})"
        if active
        else "between evaluations"
    )
    lines = [
        time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        f"Campaign progress: {progress}  completed={len(entries)}",
        f"Current phase: {phase.upper()}",
        f"Active: {active_text}",
        "",
        _metric_line("C control", control),
        _metric_line("Highest challenger", best),
    ]
    if control is not None and best is not None:
        lines.append(
            "  delta_vs_C="
            f"{float(best.get('score', 0.0)) - float(control.get('score', 0.0)):+.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    checkpoint = ROOT / args.checkpoint
    log = ROOT / args.log
    pipeline_checkpoint = ROOT / args.pipeline_checkpoint
    while True:
        if not args.once:
            print("\033[2J\033[H", end="")
        try:
            print(render(checkpoint, log, pipeline_checkpoint), flush=True)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print(f"Waiting for campaign checkpoint: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
