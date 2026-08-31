from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
)
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.research.replay import session_reward
from ghostlab.retrieval.dense import E5_SMALL_V2, DenseIndex, sha256_file
from ghostlab.retrieval.fusion import sparse_first_union_ids
from ghostlab.retrieval.learned import (
    CandidateFeatureStore,
    LearnedLinearReranker,
    LinearRerankerModel,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.query import build_dense_query
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.memory import ConversationState
from scripts.run_dense_retrieval import (
    CHAMPION_FIELD_WEIGHTS,
    CHAMPION_QUESTION_SEQUENCE,
    percentile,
    process_peak_mb,
)

ROOT = Path(__file__).resolve().parents[1]


class SparseDenseUnionAgent:
    def __init__(
        self,
        catalog_ids: set[str],
        sparse: SparseIndex,
        dense: DenseIndex,
        quality: CatalogQualityReranker,
        feature_store: CandidateFeatureStore,
        model: LinearRerankerModel,
    ) -> None:
        self.catalog_ids = catalog_ids
        self.sparse = sparse
        self.dense = dense
        self.quality = quality
        self.learned = LearnedLinearReranker(feature_store, model)
        self.sessions: dict[str, ConversationState] = {}
        self.latencies: list[float] = []
        self.failure_count = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = ConversationState(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        try:
            state = self.sessions[session_id]
            state.observe(user_message, turn)
            raw_query = ". ".join(state.messages)
            dense_query = build_dense_query(state, "negation_safe_structured")
            sparse = [
                item.parent_asin
                for item in self.sparse.search(
                    raw_query, 200, CHAMPION_FIELD_WEIGHTS
                ).items
            ]
            dense = [
                item.parent_asin for item in self.dense.search(dense_query, 200).items
            ]
            union = sparse_first_union_ids(sparse, dense, limit=400)
            if union[: len(sparse)] != sparse:
                raise RuntimeError("sparse head was not preserved")
            ranked = self.quality.rerank(union, weight=0.2, rerank_k=400)
            ranked = self.learned.rerank(raw_query, ranked, rerank_k=400)
            question = (
                CHAMPION_QUESTION_SEQUENCE[turn - 1]
                if turn <= len(CHAMPION_QUESTION_SEQUENCE)
                else None
            )
            state.last_asked_attribute = question
            if question is not None:
                state.asked_attributes.append(question)
            return normalize_response(
                {
                    "message": (
                        "Here are the closest matches based on what you have shared."
                        if question is None
                        else f"Do you have a preference for {question.replace('_', ' ')}?"
                    ),
                    "ask_attribute": question,
                    "recommendations": ranked,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                self.catalog_ids,
                top_k,
            )
        except Exception:
            self.failure_count += 1
            raise
        finally:
            self.latencies.append((time.perf_counter() - started) * 1000.0)


def summarized_metrics(sessions: list[dict]) -> dict[str, object]:
    overall = metric_summary(sessions)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "recommended_technical_score": round(
            statistics.fmean(session_reward(session) for session in sessions), 6
        ),
        "scenario_metrics": {
            name: metric_summary(values) for name, values in sorted(grouped.items())
        },
    }


def main() -> None:
    started = time.perf_counter()
    manifest_path = ROOT / "configs/experiments/dense_query_e2e_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retrieval_report_path = ROOT / str(manifest["authorization"]["retrieval_report"])
    retrieval_report = json.loads(retrieval_report_path.read_text(encoding="utf-8"))
    if not retrieval_report["gate"]["passed"]:
        raise RuntimeError("end-to-end run lacks retrieval-gate authorization")
    if retrieval_report["gate"]["selected_candidate"] != "negation_safe_structured":
        raise RuntimeError("end-to-end query differs from retrieval-gate selection")

    cache_dir = ROOT / "artifacts/cache/dense"
    if len(list(cache_dir.glob("e5_small_v2-*.npy"))) != 1:
        raise RuntimeError("prebuilt E5 index is required")
    dense = DenseIndex(
        ROOT / "data/catalog.jsonl",
        E5_SMALL_V2,
        cache_dir=cache_dir,
        model_path=ROOT / "artifacts/cache/models/e5-small-v2",
        local_files_only=True,
    )
    if not dense.cache_metadata["cache_hit"]:
        raise RuntimeError("end-to-end interaction cannot build an index")

    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    sparse = SparseIndex(ROOT / "data/catalog.jsonl")
    quality = CatalogQualityReranker(ROOT / "data/catalog.jsonl")
    feature_store = CandidateFeatureStore(
        ROOT / "data/catalog.jsonl",
        enabled_features=("feature_overlap", "catalog_quality"),
        quality=quality.quality,
    )
    champion_report_path = ROOT / "artifacts/reports/phase20_learned_features.json"
    champion = json.loads(champion_report_path.read_text(encoding="utf-8"))
    baseline_sessions = {
        str(session["sample_id"]): session
        for session in champion["oof_sessions"]["feature_quality"]
    }

    candidate_sessions = []
    folds = []
    latencies: list[float] = []
    failure_count = 0
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        fold_model = champion["folds"][fold_index]["variants"]["feature_quality"]
        model = LinearRerankerModel(
            weights=tuple(float(value) for value in fold_model["weights"]),
            l2=0.1,
            training_pairs=0,
        )
        agent = SparseDenseUnionAgent(
            catalog_ids, sparse, dense, quality, feature_store, model
        )
        fold_samples = [samples[str(sample_id)] for sample_id in outer_values]
        result = evaluate(agent, fold_samples, catalog_ids, categories, products)
        candidate_sessions.extend(result["sessions"])
        latencies.extend(agent.latencies)
        failure_count += agent.failure_count
        baseline_fold = [baseline_sessions[str(value)] for value in outer_values]
        baseline_reward = statistics.fmean(
            session_reward(session) for session in baseline_fold
        )
        candidate_reward = statistics.fmean(
            session_reward(session) for session in result["sessions"]
        )
        folds.append(
            {
                "outer_fold": fold_index,
                "sample_count": len(fold_samples),
                "baseline_reward": round(baseline_reward, 6),
                "candidate_reward": round(candidate_reward, 6),
                "delta": round(candidate_reward - baseline_reward, 6),
                "candidate_metrics": {
                    key: result[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "recommended_technical_score",
                    )
                },
            }
        )

    candidate_by_id = {
        str(session["sample_id"]): session for session in candidate_sessions
    }
    deltas = [
        session_reward(candidate_by_id[sample_id])
        - session_reward(baseline_sessions[sample_id])
        for sample_id in sorted(adaptive_ids)
    ]
    lower, upper = bootstrap_mean_interval(deltas, resamples=10000, seed=20260826)
    baseline_metrics = summarized_metrics(list(baseline_sessions.values()))
    candidate_metrics = summarized_metrics(candidate_sessions)
    score_delta = float(candidate_metrics["recommended_technical_score"]) - float(
        baseline_metrics["recommended_technical_score"]
    )
    checks = {
        "technical_score_gain_at_least_0_005": score_delta >= 0.005,
        "hit_rate_does_not_decrease": float(candidate_metrics["hit_rate_at_10"])
        >= float(baseline_metrics["hit_rate_at_10"]),
        "non_negative_delta_in_at_least_4_folds": sum(
            float(fold["delta"]) >= 0.0 for fold in folds
        )
        >= 4,
        "warm_p95_within_500_ms": percentile(latencies, 0.95) <= 500.0,
        "failure_rate_zero": failure_count == 0,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "evaluation_label": "outer-fold/out-of-fold",
        "split": "nested_v1",
        "sample_count": len(candidate_sessions),
        "holdout_accessed": False,
        "model_search_performed": False,
        "index_build_performed": False,
        "candidate_count": 1,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "score_delta": round(score_delta, 6),
        "paired_evidence": {
            "mean_delta": round(statistics.fmean(deltas), 6),
            "paired_bootstrap_95_ci": [round(lower, 6), round(upper, 6)],
            "paired_randomization_p_value": round(
                paired_randomization_pvalue(deltas, resamples=10000, seed=20260826),
                6,
            ),
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        },
        "folds": folds,
        "performance": {
            "turn_count": len(latencies),
            "warm_latency_ms_mean": round(statistics.fmean(latencies), 6),
            "warm_latency_ms_p50": round(percentile(latencies, 0.5), 6),
            "warm_latency_ms_p95": round(percentile(latencies, 0.95), 6),
            "warm_latency_ms_max": round(max(latencies), 6),
            "peak_process_memory_mb": round(process_peak_mb(), 6),
            "failure_count": failure_count,
        },
        "gate": {"checks": checks, "passed": passed},
        "decision": "PROMOTE_INTERACTION" if passed else "PARK_QUERY_DENSE",
        "sessions": candidate_sessions,
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "retrieval_report_sha256": sha256_file(retrieval_report_path),
            "runner_code_sha256": sha256_file(ROOT / "scripts/run_dense_query_e2e.py"),
            "split_sha256": sha256_file(nested_path),
            "champion_oof_report_sha256": sha256_file(champion_report_path),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    output = ROOT / "artifacts/reports/dense_query_e2e_v1.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "baseline_metrics",
                    "candidate_metrics",
                    "score_delta",
                    "paired_evidence",
                    "folds",
                    "performance",
                    "gate",
                    "decision",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
