from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.policy.models import RuntimeConfig
from ghostlab.research.replay import paired_delta
from ghostlab.runtime.agent import GhostLabRuntime

ROOT = Path(__file__).resolve().parents[1]
GUARDED_DIR = ROOT / "artifacts/guarded"
ACCESS_LOG = GUARDED_DIR / "access_log.jsonl"


def append_access(payload: dict) -> None:
    GUARDED_DIR.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    if ACCESS_LOG.exists() and ACCESS_LOG.read_text(encoding="utf-8").strip():
        raise SystemExit(
            "guarded holdout has already been accessed; refusing a second run"
        )
    analysis_path = ROOT / "configs/validation/primary_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    compiled_path = ROOT / "configs/compiled_policy.json"
    compiled_config = RuntimeConfig.model_validate_json(
        compiled_path.read_text(encoding="utf-8")
    )
    if analysis["candidate_policy_id"] != compiled_config.policy_id:
        raise SystemExit("predeclared candidate does not match compiled primary")
    compiled_hash = hashlib.sha256(compiled_path.read_bytes()).hexdigest()
    if compiled_hash != analysis["compiled_policy_sha256"]:
        raise SystemExit("compiled policy hash does not match predeclared analysis")
    started = datetime.now(UTC).isoformat()
    append_access(
        {
            "status": "started",
            "at": started,
            "candidate_policy_id": analysis["candidate_policy_id"],
            "primary_metric": analysis["primary_metric"],
        }
    )
    guarded = json.loads((GUARDED_DIR / "f3_v1.json").read_text(encoding="utf-8"))
    holdout_ids = set(guarded["sample_ids"])
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if sample["sample_id"] in holdout_ids
    ]
    if len(samples) != 50:
        raise SystemExit("guarded split shape mismatch")
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    candidate = evaluate(
        GhostLabRuntime(
            ROOT / "data/catalog.jsonl", ROOT / "configs/compiled_policy.json"
        ),
        samples,
        catalog_ids,
        categories,
        products,
    )
    baseline = evaluate(
        GhostLabRuntime(
            ROOT / "data/catalog.jsonl",
            ROOT / "configs/techniques/manual_strong_v1.json",
        ),
        samples,
        catalog_ids,
        categories,
        products,
    )
    deltas = paired_delta(candidate["sessions"], baseline["sessions"])
    interval = bootstrap_mean_interval(
        deltas,
        resamples=int(analysis["bootstrap_resamples"]),
        confidence=float(analysis["confidence_level"]),
    )
    p_value = paired_randomization_pvalue(
        deltas, resamples=int(analysis["randomization_resamples"])
    )
    metric = str(analysis["primary_metric"])
    primary_delta = float(candidate[metric]) - float(baseline[metric])
    passed = primary_delta >= float(analysis["minimum_delta"])
    report = {
        "phase": 12,
        "gate": "one_shot_f3_confirmation",
        "access_started_at": started,
        "sample_count": len(samples),
        "candidate_policy_id": analysis["candidate_policy_id"],
        "baseline_policy_id": analysis["baseline_policy_id"],
        "primary_metric": metric,
        "candidate": {
            key: value for key, value in candidate.items() if key != "sessions"
        },
        "baseline": {
            key: value for key, value in baseline.items() if key != "sessions"
        },
        "primary_delta": round(primary_delta, 6),
        "paired_mean_reward_delta": round(sum(deltas) / len(deltas), 6),
        "paired_bootstrap_interval_95": [round(value, 6) for value in interval],
        "paired_randomization_pvalue": round(p_value, 6),
        "passed": passed,
        "post_access_policy_changes_allowed": False,
    }
    output = ROOT / "artifacts/reports/phase12_holdout.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_access(
        {
            "status": "completed",
            "at": datetime.now(UTC).isoformat(),
            "candidate_policy_id": analysis["candidate_policy_id"],
            "passed": passed,
            "primary_delta": round(primary_delta, 6),
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("guarded confirmation did not meet the predeclared threshold")


if __name__ == "__main__":
    main()
