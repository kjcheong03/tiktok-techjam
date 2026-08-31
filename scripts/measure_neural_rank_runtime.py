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
from ghostlab.retrieval.neural_rank import (
    NeuralGBDTFeatureStore,
    PinnedCrossEncoderScorer,
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
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    denominator = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(value) / denominator


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def unique_asset_bytes(directory: Path) -> int:
    files = {
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    return sum(path.stat().st_size for path in files if path.is_file())


class TimedAgent:
    def __init__(self, wrapped: ExperimentalAgent) -> None:
        self.wrapped = wrapped
        self.latencies_ms: list[float] = []
        self.failure_count = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.wrapped.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        try:
            return self.wrapped.respond(session_id, user_message, turn, top_k)
        except Exception:
            self.failure_count += 1
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: measure_neural_rank_runtime MODEL_JSON")
    model_path = Path(sys.argv[1]).resolve()
    model_cache = ROOT / "artifacts/cache/cross_encoder"
    memory_before = rss_mb()
    started = time.perf_counter()
    catalog_path = ROOT / "data/catalog.jsonl"
    quality = CatalogQualityReranker(catalog_path)
    base = GBDTFeatureStore(catalog_path, quality=quality.quality)
    scorer = PinnedCrossEncoderScorer(catalog_path, model_cache)
    feature_store = NeuralGBDTFeatureStore(base, live_scorer=scorer)
    model = LambdaMARTModel.load(model_path)
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
        learned_reranker=LambdaMARTReranker(feature_store, model),
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
    model_assets = unique_asset_bytes(model_cache)
    total_assets = model_assets + model_path.stat().st_size
    output = {
        "measurement_scope": "uncached pinned neural scoring with an outer-fold model",
        "outer_fold_model": True,
        "sample_count": len(samples),
        "turn_count": len(timed.latencies_ms),
        "cold_start_seconds": round(cold_start, 6),
        "warm_turn_p95_ms": round(percentile(timed.latencies_ms, 0.95), 6),
        "warm_turn_mean_ms": round(statistics.fmean(timed.latencies_ms), 6),
        "warm_turn_max_ms": round(max(timed.latencies_ms), 6),
        "memory_before_init_mb": round(memory_before, 3),
        "memory_after_init_mb": round(memory_after_init, 3),
        "peak_process_memory_mb": round(rss_mb(), 3),
        "tree_asset_bytes": model_path.stat().st_size,
        "cross_encoder_asset_bytes": model_assets,
        "total_model_asset_bytes": total_assets,
        "total_model_asset_mb": round(total_assets / (1024 * 1024), 6),
        "cross_encoder_initialization_seconds": round(scorer.initialization_seconds, 6),
        "neural_score_seconds": round(scorer.score_seconds, 6),
        "neural_score_calls": scorer.score_calls,
        "neural_scored_pairs": scorer.scored_pairs,
        "external_calls_per_turn": 0,
        "failure_count": timed.failure_count,
        "missing_score_count": feature_store.missing_count,
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
