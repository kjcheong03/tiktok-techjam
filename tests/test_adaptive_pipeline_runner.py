from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_adaptive_pipeline_plan_has_ordered_dependency_stages() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_adaptive_hybrid_pipeline.py",
            "--show-plan",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stages = payload["stages"]
    assert [stage["stage"] for stage in stages] == [
        "split",
        "fit",
        "diversity",
        "llm",
        "evaluate",
        "baselines",
        "validate",
        "campaign",
        "package",
        "finalists",
        "compare",
    ]
    assert all(stage["status"] in {"run", "skip_complete"} for stage in stages)
    assert all(stage["command"] and stage["outputs"] for stage in stages)


def test_adaptive_pipeline_can_stop_before_campaign() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_adaptive_hybrid_pipeline.py",
            "--through-stage",
            "validate",
            "--show-plan",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stages = json.loads(completed.stdout)["stages"]
    assert [stage["stage"] for stage in stages] == [
        "split",
        "fit",
        "diversity",
        "llm",
        "evaluate",
        "baselines",
        "validate",
    ]


def test_focused_warm_start_profile_builds_bounded_campaign() -> None:
    warm_start = (
        "configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_adaptive_hybrid_pipeline.py",
            "--from-stage",
            "campaign",
            "--through-stage",
            "campaign",
            "--campaign-search-profile",
            "focused_warm_start",
            "--campaign-warm-start",
            warm_start,
            "--show-plan",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    command = json.loads(completed.stdout)["stages"][0]["command"]

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--candidate-limit") == "36"
    assert value("--beam-width") == "8"
    assert value("--higher-order-rounds") == "1"
    assert value("--f1-candidates") == "6"
    assert value("--f2-candidates") == "5"
    assert value("--hpo-trials-per-structure") == "1"
    assert value("--warm-start") == warm_start


def test_focused_profile_requires_explicit_warm_start() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_adaptive_hybrid_pipeline.py",
            "--campaign-search-profile",
            "focused_warm_start",
            "--show-plan",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "requires --campaign-warm-start" in completed.stderr


def test_additive_warm_start_profile_is_monotonic_and_isolated() -> None:
    warm_start = (
        "configs/warm_starts/"
        "adaptive_d4e040a07e6d_to_1a_3b_f1_selected_v1.json"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_adaptive_hybrid_pipeline.py",
            "--from-stage",
            "campaign",
            "--through-stage",
            "package",
            "--campaign-search-profile",
            "additive_warm_start",
            "--campaign-warm-start",
            warm_start,
            "--show-plan",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stages = json.loads(completed.stdout)["stages"]
    campaign = stages[0]
    command = campaign["command"]

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--search-mode") == "additive_warm_start"
    assert value("--candidate-limit") == "14"
    assert value("--higher-order-rounds") == "2"
    assert value("--max-additive-techniques") == "3"
    assert value("--warm-start") == warm_start
    assert "--freeze-warm-semantic" in command
    assert command.count("--additive-technique") == 6
    assert "additive_warm_start" in value("--checkpoint")
    assert "additive_warm_start" in value("--output")
    assert "additive_warm_start" in stages[1]["outputs"][0]
