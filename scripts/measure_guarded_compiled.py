from __future__ import annotations

import json
import resource
import socket
import statistics
import time
from pathlib import Path
from typing import cast
from unittest.mock import patch

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.competition.contract import AgentProtocol
from ghostlab.policy.models import RuntimeConfig
from ghostlab.runtime.agent import GhostLabRuntime
from ghostlab.runtime.guarded_gbdt import CompiledGuardedGBDTAgent
from scripts.measure_gbdt_runtime import percentile
from scripts.run_gbdt_reranker import sha256_file
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/compiled_policy.json"
REPORT_PATH = ROOT / "artifacts/reports/guarded_compiled_runtime_v1.json"


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024 * 1024)


class _TimedAgent:
    def __init__(self, wrapped: AgentProtocol) -> None:
        self.wrapped = wrapped
        self.latencies_ms: list[float] = []
        self.failures = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.wrapped.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        try:
            return self.wrapped.respond(session_id, user_message, turn, top_k)
        except Exception:
            self.failures += 1
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000)


def main() -> None:
    memory_before = _rss_mb()
    started = time.perf_counter()
    catalog_path = ROOT / "data/catalog.jsonl"
    runtime = GhostLabRuntime(catalog_path, CONFIG_PATH)
    cold_start = time.perf_counter() - started
    primary = runtime._primary
    if not isinstance(primary, CompiledGuardedGBDTAgent):
        raise TypeError("default config did not compile to the guarded runtime")
    models_lazy_before_first_response = primary.models._base is None
    timed = _TimedAgent(runtime)
    nested = json.loads(
        (ROOT / "configs/splits/nested_v1.json").read_text(encoding="utf-8")
    )
    adaptive = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive
    ]
    catalog_ids, categories, products = catalog_index(catalog_path)
    with patch.object(
        socket,
        "create_connection",
        side_effect=AssertionError("network access is forbidden"),
    ):
        result = evaluate(
            cast(Agent, timed),
            samples,
            catalog_ids,
            categories,
            products,
        )
    config = RuntimeConfig.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))
    assets = tuple(
        asset
        for asset in (
            config.techniques.base_model_asset,
            config.techniques.constraint_model_asset,
        )
        if asset is not None
    )
    asset_bytes = sum((ROOT / asset.path).stat().st_size for asset in assets)
    latencies = timed.latencies_ms
    warm = latencies[1:]
    report = {
        "schema_version": 1,
        "gate": "guarded_compiled_runtime",
        "passed": (
            timed.failures == 0
            and result["recommended_technical_score"] == 0.886852
            and cold_start <= 30.0
            and percentile(warm, 0.95) <= 500.0
            and _rss_mb() <= 4096.0
            and asset_bytes <= 500 * 1024 * 1024
        ),
        "holdout_accessed": False,
        "measurement_scope": "all-development compiled runtime",
        "sample_count": len(samples),
        "turn_count": len(latencies),
        "cold_start_seconds": round(cold_start, 6),
        "first_response_ms": round(latencies[0], 6),
        "warm_turn_mean_ms": round(statistics.fmean(warm), 6),
        "warm_turn_p95_ms": round(percentile(warm, 0.95), 6),
        "warm_turn_max_ms": round(max(warm), 6),
        "memory_before_init_mb": round(memory_before, 3),
        "peak_process_memory_mb": round(_rss_mb(), 3),
        "failure_count": timed.failures,
        "external_calls_per_turn": 0,
        "offline_network_guard": "passed",
        "models_lazy_before_first_response": models_lazy_before_first_response,
        "models_loaded_after_evaluation": (
            primary.models._base is not None and primary.models._constraint is not None
        ),
        "experiment_trace_present": any(
            hasattr(primary, name) for name in ("routing_trace", "question_trace")
        ),
        "metrics": {
            key: result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
            )
        },
        "assets": {
            "combined_bytes": asset_bytes,
            "combined_mb": round(asset_bytes / (1024 * 1024), 6),
            "models": [
                {
                    "path": asset.path,
                    "configured_sha256": asset.sha256,
                    "actual_sha256": sha256_file(ROOT / asset.path),
                }
                for asset in assets
            ],
        },
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
            "canonical_sha256": config.canonical_hash(),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("compiled runtime gate failed")


if __name__ == "__main__":
    main()
