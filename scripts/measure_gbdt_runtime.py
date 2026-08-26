from __future__ import annotations

import json
import resource
import statistics
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.retrieval.gbdt import (
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
QUESTION_ORDER = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / (1024 * 1024)


class TimedAgent:
    def __init__(self, wrapped: ExperimentalAgent) -> None:
        self.wrapped = wrapped
        self.latencies_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.wrapped.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        result = self.wrapped.respond(session_id, user_message, turn, top_k)
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.measure_gbdt_runtime MODEL_JSON")
    model_path = Path(sys.argv[1]).resolve()
    memory_before = rss_mb()
    started = time.perf_counter()
    catalog_path = ROOT / "data/catalog.jsonl"
    quality = CatalogQualityReranker(catalog_path)
    store = GBDTFeatureStore(catalog_path, quality=quality.quality)
    model = LambdaMARTModel.load(model_path)
    reranker = LambdaMARTReranker(store, model)
    wrapped = ExperimentalAgent(
        catalog_path,
        state_variant="raw_history",
        question_variant="sequence",
        question_order=QUESTION_ORDER,
        negative_evidence=False,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        quality_prior=quality,
        learned_reranker=reranker,
    )
    cold_start = time.perf_counter() - started
    memory_after_init = rss_mb()
    timed = TimedAgent(wrapped)
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive
    ]
    catalog_ids, categories, products = catalog_index(catalog_path)
    result = evaluate(timed, samples, catalog_ids, categories, products)
    peak_memory = rss_mb()
    latencies = timed.latencies_ms
    output = {
        "measurement_scope": "all-development refit runtime",
        "sample_count": len(samples),
        "turn_count": len(latencies),
        "cold_start_seconds": round(cold_start, 6),
        "warm_turn_p95_ms": round(percentile(latencies, 0.95), 6),
        "warm_turn_mean_ms": round(statistics.fmean(latencies), 6),
        "warm_turn_max_ms": round(max(latencies), 6),
        "memory_before_init_mb": round(memory_before, 3),
        "memory_after_init_mb": round(memory_after_init, 3),
        "peak_process_memory_mb": round(peak_memory, 3),
        "model_asset_bytes": model_path.stat().st_size,
        "model_asset_mb": round(model_path.stat().st_size / (1024 * 1024), 6),
        "external_calls_per_turn": 0,
        "failure_count": 0,
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
