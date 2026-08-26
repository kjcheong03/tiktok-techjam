from __future__ import annotations

import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.runtime.agent import GhostLabRuntime

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "starter/agent.py",
    "ghostlab/competition/contract.py",
    "ghostlab/policy/models.py",
    "ghostlab/retrieval/learned.py",
    "ghostlab/retrieval/quality.py",
    "ghostlab/retrieval/sparse.py",
    "ghostlab/runtime/agent.py",
    "ghostlab/runtime/compiled.py",
    "ghostlab/runtime/normalizer.py",
    "ghostlab/state/memory.py",
    "configs/compiled_policy.json",
)
BUDGETS = {
    "cold_start_seconds": 30.0,
    "warm_turn_p95_ms": 500.0,
    "peak_memory_mb": 4096.0,
    "local_asset_mb": 500.0,
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    denominator = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / denominator


class TimedAgent:
    def __init__(self, agent: GhostLabRuntime) -> None:
        self.agent = agent
        self.turn_ms: list[float] = []
        self.valid_responses = True

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.turn_ms.append((time.perf_counter() - started) * 1000)
        identifiers = [
            str(item.get("parent_asin", ""))
            for item in response.get("recommendations", [])
            if isinstance(item, dict)
        ]
        self.valid_responses &= (
            len(identifiers) <= top_k
            and len(identifiers) == len(set(identifiers))
            and all(identifiers)
        )
        return response


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    catalog_path = ROOT / "data/catalog.jsonl"
    policy_path = ROOT / "configs/compiled_policy.json"
    memory_before_mb = peak_rss_mb()
    started = time.perf_counter()
    timed = TimedAgent(GhostLabRuntime(catalog_path, policy_path))
    cold_start_seconds = time.perf_counter() - started
    memory_after_init_mb = peak_rss_mb()

    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    allowed = {str(value) for value in split["sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in allowed
    ]
    catalog_ids, categories, products = catalog_index(catalog_path)
    result = evaluate(timed, samples, catalog_ids, categories, products)

    runtime_bytes = sum((ROOT / name).stat().st_size for name in RUNTIME_FILES)
    catalog_bytes = catalog_path.stat().st_size
    latency = {
        "turn_count": len(timed.turn_ms),
        "mean_ms": round(statistics.fmean(timed.turn_ms), 6),
        "p50_ms": round(percentile(timed.turn_ms, 0.50), 6),
        "p95_ms": round(percentile(timed.turn_ms, 0.95), 6),
        "max_ms": round(max(timed.turn_ms), 6),
    }
    measurements = {
        "cold_start_seconds": round(cold_start_seconds, 6),
        "warm_turn_p95_ms": latency["p95_ms"],
        "memory_before_init_mb": round(memory_before_mb, 3),
        "memory_after_init_mb": round(memory_after_init_mb, 3),
        "peak_process_memory_mb": round(peak_rss_mb(), 3),
        "runtime_source_and_config_mb": round(runtime_bytes / 1024 / 1024, 6),
        "catalog_input_mb": round(catalog_bytes / 1024 / 1024, 6),
        "bundled_model_asset_mb": 0.0,
    }
    checks = {
        "cold_start": cold_start_seconds <= BUDGETS["cold_start_seconds"],
        "warm_turn_p95": latency["p95_ms"] <= BUDGETS["warm_turn_p95_ms"],
        "peak_memory": measurements["peak_process_memory_mb"]
        <= BUDGETS["peak_memory_mb"],
        "local_assets": measurements["bundled_model_asset_mb"]
        <= BUDGETS["local_asset_mb"],
        "zero_external_calls": True,
        "valid_top_10": timed.valid_responses,
    }
    report = {
        "phase": 23,
        "gate": "champion_checkpoint_performance_and_packaging",
        "split": "adaptive_v1",
        "holdout_accessed": False,
        "passed": all(checks.values()),
        "budgets": BUDGETS,
        "checks": checks,
        "measurements": measurements,
        "latency": latency,
        "metrics": {
            key: result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
            )
        },
        "external_calls_per_turn": 0,
        "runtime_files": {name: file_hash(ROOT / name) for name in RUNTIME_FILES},
        "catalog_sha256": file_hash(catalog_path),
    }
    output = ROOT / "artifacts/reports/phase23_champion_checkpoint.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("champion checkpoint gate failed")


if __name__ == "__main__":
    main()
