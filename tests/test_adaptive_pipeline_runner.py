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
        "fit",
        "llm",
        "evaluate",
        "validate",
        "campaign",
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
        "fit",
        "llm",
        "evaluate",
        "validate",
    ]
