from __future__ import annotations

import hashlib
import json
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl, metric_summary
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.policy.models import RuntimeConfig
from ghostlab.research.replay import evaluate_replay, paired_delta, session_reward
from ghostlab.retrieval.cross_encoder import (
    PASSAGE_SCHEMA_VERSION,
    CrossEncoderReranker,
)
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiments/challenger_cross_encoder_v1.json"


class TimedAgent:
    def __init__(self, agent: ExperimentalAgent) -> None:
        self.agent = agent
        self.turn_ms: list[float] = []
        self.failure_count = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        try:
            return self.agent.respond(session_id, user_message, turn, top_k)
        except Exception:
            self.failure_count += 1
            raise
        finally:
            self.turn_ms.append((time.perf_counter() - started) * 1000)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def directory_bytes(directory: Path) -> int:
    files = {
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    return sum(path.stat().st_size for path in files if path.is_file())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    denominator = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / denominator


def aggregate_sessions(sessions: list[dict]) -> dict:
    summary = metric_summary(sessions)
    mttc = float(summary["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = (
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **summary,
        "recommended_technical_score": round(score, 6),
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
    }


def main() -> None:
    started_all = time.perf_counter()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["passage_schema_version"] != PASSAGE_SCHEMA_VERSION:
        raise ValueError("manifest and runtime passage schemas differ")
    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    ]
    catalog_path = ROOT / "data/catalog.jsonl"
    _, categories, products = catalog_index(catalog_path)
    config = RuntimeConfig.model_validate_json(
        (ROOT / "configs/compiled_policy.json").read_text(encoding="utf-8")
    ).techniques
    common: dict[str, object] = {
        "state_variant": "raw_history",
        "question_variant": "sequence",
        "question_order": tuple(config.question_order),
        "retrieval_route": "keyword",
        "negative_evidence": False,
        "sparse_weights": config.sparse_field_weights,
        "quality_prior_weight": config.quality_prior_weight,
    }

    results: dict[str, dict] = {}

    def evaluate_variant(
        name: str,
        *,
        reranker: CrossEncoderReranker | None,
        weight: float = 0.0,
        rerank_k: int = 20,
        complexity: int,
    ) -> None:
        agent = ExperimentalAgent(
            catalog_path,
            **common,  # type: ignore[arg-type]
            cross_encoder_reranker=reranker,
            cross_encoder_weight=weight,
            cross_encoder_rerank_k=rerank_k,
        )
        timed = TimedAgent(agent)
        misses_before = reranker.cache_misses if reranker is not None else 0
        calls_before = reranker.score_calls if reranker is not None else 0
        score_seconds_before = reranker.score_seconds if reranker is not None else 0.0
        started = time.perf_counter()
        evaluation = evaluate_replay(timed, samples, categories, products)
        cache_misses = (
            reranker.cache_misses - misses_before if reranker is not None else 0
        )
        score_calls = reranker.score_calls - calls_before if reranker is not None else 0
        neural_seconds = (
            reranker.score_seconds - score_seconds_before
            if reranker is not None
            else 0.0
        )
        results[name] = {
            "weight": weight,
            "rerank_k": rerank_k,
            "complexity": complexity,
            "metrics": {
                key: evaluation[key]
                for key in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "recommended_technical_score",
                    "scenario_metrics",
                )
            },
            "runtime": {
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "mean_turn_ms": round(statistics.fmean(timed.turn_ms), 6),
                "p95_turn_ms": round(percentile(timed.turn_ms, 0.95), 6),
                "max_turn_ms": round(max(timed.turn_ms), 6),
                "uncached_pairs": cache_misses,
                "neural_score_calls": score_calls,
                "neural_score_seconds": round(neural_seconds, 6),
                "mean_neural_seconds_per_score_call": (
                    round(neural_seconds / score_calls, 6) if score_calls else None
                ),
                "latency_mode": (
                    "uncached_neural_reference"
                    if cache_misses
                    else "shared_score_cache_replay_only"
                ),
                "failure_count": timed.failure_count,
            },
            "sessions": evaluation["sessions"],
        }
        print(name, evaluation["recommended_technical_score"], flush=True)

    evaluate_variant("control_field_quality", reranker=None, complexity=0)
    phase20 = json.loads(
        (ROOT / "artifacts/reports/phase20_learned_features.json").read_text()
    )
    champion_sessions = phase20["oof_sessions"]["feature_quality"]
    results["control_linear_champion_oof"] = {
        "weight": None,
        "rerank_k": 50,
        "complexity": 1,
        "metrics": phase20["oof_metrics"]["feature_quality"],
        "runtime": {"source": "phase20 fold-specific OOF predictions"},
        "sessions": champion_sessions,
    }
    cache_folder = ROOT / "artifacts/cache/cross_encoder"
    reranker = CrossEncoderReranker(
        catalog_path,
        model_name=manifest["model"],
        revision=manifest["model_revision"],
        cache_folder=cache_folder,
        batch_size=32,
        max_length=256,
        local_files_only=True,
    )
    for candidate in manifest["predeclared_candidates"]:
        rerank_k = int(candidate["rerank_k"])
        weight = float(candidate["weight"])
        evaluate_variant(
            f"cross_encoder_top{rerank_k}_weight_{weight:g}",
            reranker=reranker,
            weight=weight,
            rerank_k=rerank_k,
            complexity=2 if rerank_k == 20 else 3,
        )

    comparisons = {}
    for control_name in ("control_field_quality", "control_linear_champion_oof"):
        baseline_sessions = results[control_name]["sessions"]
        comparisons[control_name] = {}
        for name, result in results.items():
            deltas = paired_delta(result["sessions"], baseline_sessions)
            interval = bootstrap_mean_interval(deltas, resamples=5000)
            comparisons[control_name][name] = {
                "mean_paired_delta": round(statistics.fmean(deltas), 6),
                "bootstrap_95": [round(interval[0], 6), round(interval[1], 6)],
                "randomization_pvalue": round(
                    paired_randomization_pvalue(deltas, resamples=5000), 6
                ),
                "wins": sum(value > 1e-12 for value in deltas),
                "ties": sum(abs(value) <= 1e-12 for value in deltas),
                "losses": sum(value < -1e-12 for value in deltas),
            }

    reward_maps = {
        name: {
            str(session["sample_id"]): session_reward(session)
            for session in result["sessions"]
        }
        for name, result in results.items()
    }
    fold_selections = []
    stitched_sessions: list[dict] = []
    selection_names = [
        name for name in results if name != "control_linear_champion_oof"
    ]
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        training = adaptive_ids - outer
        training_scores = {
            name: statistics.fmean(rewards[item] for item in training)
            for name, rewards in reward_maps.items()
            if name in selection_names
        }
        best = max(training_scores.values())
        selected = min(
            (name for name, score in training_scores.items() if score >= best - 0.005),
            key=lambda name: (
                results[name]["complexity"],
                -training_scores[name],
                name,
            ),
        )
        selected_outer_sessions = [
            session
            for session in results[selected]["sessions"]
            if str(session["sample_id"]) in outer
        ]
        outer_rewards = [session_reward(session) for session in selected_outer_sessions]
        stitched_sessions.extend(selected_outer_sessions)
        fold_selections.append(
            {
                "outer_fold": fold_index,
                "selected": selected,
                "training_reward": round(training_scores[selected], 6),
                "outer_reward": round(statistics.fmean(outer_rewards), 6),
            }
        )

    selected_oof_metrics = aggregate_sessions(stitched_sessions)
    champion_oof_sessions = results["control_linear_champion_oof"]["sessions"]
    selected_deltas = paired_delta(stitched_sessions, champion_oof_sessions)
    selected_interval = bootstrap_mean_interval(selected_deltas, resamples=5000)
    selected_vs_champion = {
        "mean_paired_delta": round(statistics.fmean(selected_deltas), 6),
        "bootstrap_95": [
            round(selected_interval[0], 6),
            round(selected_interval[1], 6),
        ],
        "randomization_pvalue": round(
            paired_randomization_pvalue(selected_deltas, resamples=5000), 6
        ),
        "wins": sum(value > 1e-12 for value in selected_deltas),
        "ties": sum(abs(value) <= 1e-12 for value in selected_deltas),
        "losses": sum(value < -1e-12 for value in selected_deltas),
    }
    model_files = {
        str(path.relative_to(cache_folder)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in cache_folder.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    fold_outer_scores = [float(item["outer_reward"]) for item in fold_selections]
    deployment_candidate = min(
        Counter(item["selected"] for item in fold_selections),
        key=lambda name: (
            -Counter(item["selected"] for item in fold_selections)[name],
            results[name]["complexity"],
            name,
        ),
    )
    selected_cross_encoder_folds = sum(
        str(item["selected"]).startswith("cross_encoder") for item in fold_selections
    )
    elapsed_seconds = time.perf_counter() - started_all
    decision_status = (
        "PROMOTE_PENDING_INTEGRATION"
        if selected_cross_encoder_folds > 0
        and selected_vs_champion["mean_paired_delta"] > 0.0
        and selected_vs_champion["bootstrap_95"][0] >= -0.005
        else "PARKED_STANDALONE"
    )
    report = {
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "split": manifest["split"],
        "sample_count": len(samples),
        "holdout_accessed": False,
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "passage_schema_version": PASSAGE_SCHEMA_VERSION,
        "evaluation_controls": [
            "fixed_field_bm25_plus_quality",
            "stored_fold_specific_two_feature_linear_champion_oof",
        ],
        "candidate_set_predeclared_before_neural_completion": True,
        "nested_selected_metrics": selected_oof_metrics,
        "nested_selected_sessions": sorted(
            stitched_sessions, key=lambda item: str(item["sample_id"])
        ),
        "nested_selected_vs_linear_champion": selected_vs_champion,
        "fold_stability": {
            "outer_scores": fold_outer_scores,
            "mean": round(statistics.fmean(fold_outer_scores), 6),
            "population_sd": round(statistics.pstdev(fold_outer_scores), 6),
            "worst_fold": round(min(fold_outer_scores), 6),
        },
        "decision": {
            "status": decision_status,
            "deployment_candidate": deployment_candidate,
            "selected_cross_encoder_folds": selected_cross_encoder_folds,
        },
        "fold_selections": fold_selections,
        "model_runtime": {
            "initialization_seconds": round(reranker.initialization_seconds, 6),
            "neural_score_seconds": round(reranker.score_seconds, 6),
            "score_cache_hits": reranker.cache_hits,
            "score_cache_misses": reranker.cache_misses,
            "neural_score_calls": reranker.score_calls,
            "mean_seconds_per_score_call": round(
                reranker.score_seconds / max(1, reranker.score_calls), 6
            ),
            "cache_and_asset_bytes": directory_bytes(cache_folder),
            "asset_sha256": model_files,
            "peak_process_memory_mb": round(peak_rss_mb(), 3),
            "uncached_latency_references": {
                name: result["runtime"]
                for name, result in results.items()
                if result["runtime"].get("uncached_pairs", 0) > 0
            },
        },
        "paired_comparisons": comparisons,
        "variants": results,
        "hashes": {
            "catalog_sha256": file_hash(catalog_path),
            "public_set_sha256": file_hash(ROOT / "data/public_set.jsonl"),
            "split_sha256": file_hash(ROOT / "configs/splits/nested_v1.json"),
            "manifest_sha256": file_hash(MANIFEST),
            "implementation_sha256": file_hash(
                ROOT / "ghostlab/retrieval/cross_encoder.py"
            ),
            "runner_sha256": file_hash(Path(__file__)),
        },
        "elapsed_seconds": round(elapsed_seconds, 6),
        "within_wall_clock_limit": elapsed_seconds
        <= float(manifest["wall_clock_limit_seconds"]),
        "total_runtime_failures": sum(
            int(result["runtime"].get("failure_count", 0))
            for result in results.values()
        ),
    }
    output = ROOT / "artifacts/reports/challenger_cross_encoder_v1.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fold_selections": fold_selections,
                "model_runtime": report["model_runtime"],
                "nested_selected_metrics": report["nested_selected_metrics"],
                "nested_selected_vs_linear_champion": selected_vs_champion,
                "scores": {
                    name: result["metrics"]["recommended_technical_score"]
                    for name, result in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
