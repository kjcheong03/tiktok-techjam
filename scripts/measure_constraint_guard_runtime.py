from __future__ import annotations

import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import cast

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.retrieval.constraint_gbdt import (
    ConstraintAgentAdapter,
    ConstraintGBDTFeatureStore,
    RuntimeConstraintReranker,
)
from ghostlab.retrieval.gbdt import LambdaMARTModel, LambdaMARTReranker
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.runtime.experimental import ExperimentalAgent
from scripts.measure_gbdt_runtime import TimedAgent, percentile
from scripts.run_gbdt_constraint_interaction import FIELD_WEIGHTS, QUESTION_ORDER, ROOT
from starter.agent import Agent


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / (1024 * 1024)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m scripts.measure_constraint_guard_runtime BASE CONSTRAINT"
        )
    base_path, constraint_path = (Path(value).resolve() for value in sys.argv[1:])
    memory_before = rss_mb()
    started = time.perf_counter()
    catalog_path = ROOT / "data/catalog.jsonl"
    quality = CatalogQualityReranker(catalog_path)
    features = ConstraintGBDTFeatureStore(catalog_path, quality=quality.quality)
    base_model = LambdaMARTModel.load(base_path)
    constraint_model = LambdaMARTModel.load(constraint_path)
    contextual = RuntimeConstraintReranker(
        str(catalog_path),
        FIELD_WEIGHTS,
        features,
        constraint_model,
        fallback=LambdaMARTReranker(features, base_model),
    )
    base_agent = ExperimentalAgent(
        catalog_path,
        state_variant="raw_history",
        question_variant="sequence",
        question_order=QUESTION_ORDER,
        negative_evidence=True,
        provenance=True,
        override_invalidation=True,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        quality_prior=quality,
        learned_reranker=contextual,
    )
    wrapped = ConstraintAgentAdapter(base_agent, contextual)
    cold_start = time.perf_counter() - started
    timed = TimedAgent(cast(ExperimentalAgent, wrapped))
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive
    ]
    catalog_ids, categories, products = catalog_index(catalog_path)
    result = evaluate(cast(Agent, timed), samples, catalog_ids, categories, products)
    latencies = timed.latencies_ms
    asset_bytes = base_path.stat().st_size + constraint_path.stat().st_size
    fallback_turns = sum(
        item["route"] == "base_override_fallback" for item in contextual.routing_trace
    )
    fallback_sessions = len(
        {
            str(item["session_id"])
            for item in contextual.routing_trace
            if item["route"] == "base_override_fallback"
        }
    )
    output = {
        "measurement_scope": "isolated all-development guarded runtime",
        "contention_affected": False,
        "sample_count": len(samples),
        "turn_count": len(latencies),
        "cold_start_seconds": round(cold_start, 6),
        "warm_turn_p95_ms": round(percentile(latencies, 0.95), 6),
        "warm_turn_mean_ms": round(statistics.fmean(latencies), 6),
        "warm_turn_max_ms": round(max(latencies), 6),
        "memory_before_init_mb": round(memory_before, 3),
        "peak_process_memory_mb": round(rss_mb(), 3),
        "model_asset_bytes": asset_bytes,
        "model_asset_mb": round(asset_bytes / (1024 * 1024), 6),
        "external_calls_per_turn": 0,
        "response_calls": timed.response_calls,
        "failure_count": timed.failure_count,
        "failure_counts": {
            "reset_exceptions": timed.reset_exception_count,
            "response_exceptions": timed.response_exception_count,
            "invalid_responses": timed.invalid_response_count,
        },
        "fallback_turns": fallback_turns,
        "fallback_sessions": fallback_sessions,
        "metrics": {
            key: result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
            )
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
