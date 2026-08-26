from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.retrieval.dense import (
    E5_SMALL_V2,
    MINILM_CONTROL,
    DenseIndex,
    DenseModelSpec,
    rank_biased_overlap,
    sha256_file,
)
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.state.memory import ConversationState

ROOT = Path(__file__).resolve().parents[1]
KS = (10, 50, 100, 200)
CHAMPION_FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
CHAMPION_QUESTION_SEQUENCE = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)
MODEL_ASSET_PATHS = {
    "e5_small_v2": ROOT / "artifacts/cache/models/e5-small-v2",
}


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def process_peak_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def tree_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    file_count = 0
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        relative = item.relative_to(path).as_posix()
        item_hash = sha256_file(item)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(item_hash.encode())
        digest.update(b"\n")
        size += item.stat().st_size
        file_count += 1
    return {
        "tree_sha256": digest.hexdigest(),
        "bytes": size,
        "mb": round(size / (1024 * 1024), 6),
        "file_count": file_count,
    }


def target_rank(ranking: list[str], target: str) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def recall_summary(
    records: list[dict[str, object]], rankings: dict[str, list[str]]
) -> dict[str, float]:
    count = len(records)
    return {
        f"recall_at_{k}": round(
            sum(
                str(record["target"]) in rankings[str(record["record_id"])][:k]
                for record in records
            )
            / count,
            6,
        )
        for k in KS
    }


def route_report(
    records: list[dict[str, object]], rankings: dict[str, list[str]]
) -> dict[str, object]:
    grouped_stage: dict[int, list[dict[str, object]]] = defaultdict(list)
    grouped_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_period: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_fold: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped_stage[int(record["turn"])].append(record)
        grouped_scenario[str(record["scenario_type"])].append(record)
        grouped_period[str(record["period"])].append(record)
        grouped_fold[int(record["outer_fold"])].append(record)
    return {
        "all_query_stages": recall_summary(records, rankings),
        "first_turn": recall_summary(
            [record for record in records if record["turn"] == 1], rankings
        ),
        "by_stage": {
            str(key): recall_summary(grouped_stage[key], rankings)
            for key in sorted(grouped_stage)
        },
        "early_late": {
            key: recall_summary(grouped_period[key], rankings)
            for key in sorted(grouped_period)
        },
        "by_scenario": {
            key: recall_summary(grouped_scenario[key], rankings)
            for key in sorted(grouped_scenario)
        },
        "outer_folds": {
            str(key): recall_summary(grouped_fold[key], rankings)
            for key in sorted(grouped_fold)
        },
    }


def complementarity(
    records: list[dict[str, object]],
    sparse: dict[str, list[str]],
    dense: dict[str, list[str]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for k in KS:
        dense_only = []
        sparse_only = []
        union_hits = 0
        for record in records:
            key = str(record["record_id"])
            target = str(record["target"])
            sparse_hit = target in sparse[key][:k]
            dense_hit = target in dense[key][:k]
            if dense_hit and not sparse_hit:
                dense_only.append(key)
            if sparse_hit and not dense_hit:
                sparse_only.append(key)
            union_hits += sparse_hit or dense_hit
        result[f"at_{k}"] = {
            "dense_only_query_rescues": len(dense_only),
            "sparse_only_query_losses": len(sparse_only),
            "dense_only_unique_sessions": len(
                {key.split("::", 1)[0] for key in dense_only}
            ),
            "sparse_only_unique_sessions": len(
                {key.split("::", 1)[0] for key in sparse_only}
            ),
            "union_recall": round(union_hits / len(records), 6),
            "union_candidate_ceiling": 2 * k,
        }
    return result


def overlap_by_stage(
    records: list[dict[str, object]],
    sparse: dict[str, list[str]],
    dense: dict[str, list[str]],
) -> dict[str, object]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["turn"])].append(record)
    report: dict[str, object] = {}
    for stage, stage_records in sorted(grouped.items()):
        by_k: dict[str, object] = {}
        for k in KS:
            shared = []
            jaccards = []
            rbo = []
            for record in stage_records:
                key = str(record["record_id"])
                left = sparse[key][:k]
                right = dense[key][:k]
                intersection = len(set(left) & set(right))
                shared.append(intersection)
                denominator = len(set(left) | set(right))
                jaccards.append(intersection / denominator if denominator else 0.0)
                rbo.append(rank_biased_overlap(left, right, limit=k))
            by_k[str(k)] = {
                "mean_shared_count": round(statistics.fmean(shared), 6),
                "mean_overlap_coefficient": round(
                    statistics.fmean(value / k for value in shared), 6
                ),
                "mean_jaccard": round(statistics.fmean(jaccards), 6),
                "mean_rank_biased_overlap_p_0_9": round(statistics.fmean(rbo), 6),
            }
        report[str(stage)] = by_k
    return report


def build_query_records(
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    fold_by_sample: dict[str, int],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        environment = ReplayEnvironment(sample, categories, products)
        state = ConversationState(sample_id, sample["user_profile"])
        observation = environment.observe()
        while observation is not None and observation.turn <= 9:
            state.observe(observation.user_message, observation.turn)
            records.append(
                {
                    "record_id": f"{sample_id}::{observation.turn}",
                    "sample_id": sample_id,
                    "turn": observation.turn,
                    "period": "early" if observation.turn <= 3 else "late",
                    "scenario_type": str(sample["scenario_type"]),
                    "outer_fold": fold_by_sample[sample_id],
                    "target": str(sample["ground_truth"]["parent_asin"]),
                    "query": ". ".join(state.messages),
                }
            )
            if observation.turn > len(CHAMPION_QUESTION_SEQUENCE):
                break
            observation = environment.step(
                {
                    "message": "",
                    "ask_attribute": CHAMPION_QUESTION_SEQUENCE[observation.turn - 1],
                    "recommendations": [],
                }
            )
    return records


def resolve_control_asset_path() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MINILM_CONTROL.model_name,
            revision=MINILM_CONTROL.revision,
            local_files_only=True,
        )
    )


def evaluate_dense_model(
    spec: DenseModelSpec,
    records: list[dict[str, object]],
    sparse_rankings: dict[str, list[str]],
) -> tuple[dict[str, object], dict[str, list[str]]]:
    supplied_path = MODEL_ASSET_PATHS.get(spec.key)
    asset_path = supplied_path or resolve_control_asset_path()
    started = time.perf_counter()
    dense = DenseIndex(
        ROOT / "data/catalog.jsonl",
        spec,
        cache_dir=ROOT / "artifacts/cache/dense",
        model_path=asset_path,
        local_files_only=True,
    )
    queries = [str(record["query"]) for record in records]
    ranked_values = dense.search_many(queries, 200)
    rankings = {
        str(record["record_id"]): ranking
        for record, ranking in zip(records, ranked_values, strict=True)
    }
    first_turn_queries = [
        str(record["query"]) for record in records if record["turn"] == 1
    ]
    warm_latencies = [
        dense.search(query, 200).elapsed_ms for query in first_turn_queries
    ]
    embedding_path = Path(str(dense.cache_metadata["embedding_path"]))
    metadata_path = Path(str(dense.cache_metadata["metadata_path"]))
    report = {
        "spec": {
            "key": spec.key,
            "model_name": spec.model_name,
            "revision": spec.revision,
            "query_prefix": spec.query_prefix,
            "passage_prefix": spec.passage_prefix,
            "embedding_dimension": spec.embedding_dimension,
        },
        "retrieval": route_report(records, rankings),
        "complementarity_with_sparse": complementarity(
            records, sparse_rankings, rankings
        ),
        "overlap_with_sparse_by_stage": overlap_by_stage(
            records, sparse_rankings, rankings
        ),
        "performance": {
            "model_load_seconds": round(dense.model_load_seconds, 6),
            "index_cache_hit": bool(dense.cache_metadata["cache_hit"]),
            "index_build_seconds": dense.cache_metadata["build_seconds"],
            "index_load_seconds": round(
                float(dense.cache_metadata["elapsed_seconds"]), 6
            ),
            "total_evaluation_seconds": round(time.perf_counter() - started, 6),
            "warm_latency_ms_mean": round(statistics.fmean(warm_latencies), 6),
            "warm_latency_ms_p50": round(percentile(warm_latencies, 0.5), 6),
            "warm_latency_ms_p95": round(percentile(warm_latencies, 0.95), 6),
            "warm_latency_sample_count": len(warm_latencies),
            "process_peak_memory_mb": dense.cache_metadata.get(
                "build_peak_process_memory_mb"
            )
            or round(process_peak_mb(), 6),
            "current_cache_run_peak_memory_mb": round(process_peak_mb(), 6),
            "embedding_matrix_mb": round(dense.embeddings.nbytes / (1024 * 1024), 6),
        },
        "assets": tree_metadata(asset_path),
        "cache": {
            "metadata": {
                key: value
                for key, value in dense.cache_metadata.items()
                if key not in {"embedding_path", "metadata_path", "elapsed_seconds"}
            },
            "embedding_sha256": sha256_file(embedding_path),
            "metadata_sha256": sha256_file(metadata_path),
            "embedding_bytes": embedding_path.stat().st_size,
        },
    }
    del dense
    gc.collect()
    return report, rankings


def first_turn_complement(
    records: list[dict[str, object]],
    sparse: dict[str, list[str]],
    dense: dict[str, list[str]],
) -> dict[str, object]:
    first = [record for record in records if record["turn"] == 1]
    return complementarity(first, sparse, dense)


def paired_first_turn_evidence(
    records: list[dict[str, object]],
    control: dict[str, list[str]],
    candidate: dict[str, list[str]],
) -> dict[str, object]:
    first = [record for record in records if record["turn"] == 1]
    report: dict[str, object] = {}
    for k in KS:
        deltas = []
        fold_values: dict[int, list[float]] = defaultdict(list)
        for record in first:
            key = str(record["record_id"])
            target = str(record["target"])
            delta = float(target in candidate[key][:k]) - float(
                target in control[key][:k]
            )
            deltas.append(delta)
            fold_values[int(record["outer_fold"])].append(delta)
        lower, upper = bootstrap_mean_interval(deltas, resamples=10000, seed=20260826)
        report[f"at_{k}"] = {
            "mean_paired_recall_delta": round(statistics.fmean(deltas), 6),
            "paired_bootstrap_95_ci": [round(lower, 6), round(upper, 6)],
            "paired_randomization_p_value": round(
                paired_randomization_pvalue(deltas, resamples=10000, seed=20260826),
                6,
            ),
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
            "fold_deltas": {
                str(fold): round(statistics.fmean(values), 6)
                for fold, values in sorted(fold_values.items())
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen dense retrieval coverage gate")
    parser.add_argument("--output", default="artifacts/reports/dense_retrieval_v1.json")
    args = parser.parse_args()
    run_started = time.perf_counter()
    manifest_path = ROOT / "configs/experiments/dense_retrieval_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False:
        raise RuntimeError("dense manifest must keep the protected holdout sealed")

    nested_path = ROOT / "configs/splits/nested_v1.json"
    asset_manifest_path = ROOT / "configs/assets/e5_small_v2.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    allowed = {str(value) for value in nested["adaptive_sample_ids"]}
    fold_by_sample = {
        str(sample_id): fold_index
        for fold_index, fold in enumerate(nested["outer_folds"])
        for sample_id in fold
    }
    if set(fold_by_sample) != allowed:
        raise RuntimeError("outer folds do not exactly partition the adaptive split")
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in allowed
    ]
    if len(samples) != 150:
        raise RuntimeError("dense gate must run on exactly 150 adaptive sessions")
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    records = build_query_records(samples, categories, products, fold_by_sample)
    print(f"query records: {len(records)}", flush=True)

    sparse = SparseIndex(ROOT / "data/catalog.jsonl")
    sparse_rankings: dict[str, list[str]] = {}
    sparse_latencies: list[float] = []
    for index, record in enumerate(records, start=1):
        result = sparse.search(str(record["query"]), 200, CHAMPION_FIELD_WEIGHTS)
        sparse_rankings[str(record["record_id"])] = [
            item.parent_asin for item in result.items
        ]
        sparse_latencies.append(result.elapsed_ms)
        if index % 250 == 0:
            print(f"sparse queries: {index}/{len(records)}", flush=True)

    model_reports: dict[str, dict[str, object]] = {}
    model_rankings: dict[str, dict[str, list[str]]] = {}
    for spec in (MINILM_CONTROL, E5_SMALL_V2):
        print(f"evaluating {spec.key}", flush=True)
        report, rankings = evaluate_dense_model(spec, records, sparse_rankings)
        model_reports[spec.key] = report
        model_rankings[spec.key] = rankings

    minilm_first = model_reports["minilm_control"]["retrieval"]["first_turn"]
    e5_first = model_reports["e5_small_v2"]["retrieval"]["first_turn"]
    minilm_complement = first_turn_complement(
        records, sparse_rankings, model_rankings["minilm_control"]
    )
    e5_complement = first_turn_complement(
        records, sparse_rankings, model_rankings["e5_small_v2"]
    )
    checks = {
        "dense_recall_at_200_gain_at_least_0_03": (
            float(e5_first["recall_at_200"])
            >= float(minilm_first["recall_at_200"]) + 0.03
        ),
        "union_recall_at_200_gain_at_least_0_01": (
            float(e5_complement["at_200"]["union_recall"])
            >= float(minilm_complement["at_200"]["union_recall"]) + 0.01
        ),
        "at_least_5_first_turn_dense_only_rescues": (
            int(e5_complement["at_200"]["dense_only_query_rescues"]) >= 5
        ),
        "warm_p95_within_500_ms": (
            float(model_reports["e5_small_v2"]["performance"]["warm_latency_ms_p95"])
            <= 500.0
        ),
        "peak_memory_within_4096_mb": (
            float(model_reports["e5_small_v2"]["performance"]["process_peak_memory_mb"])
            <= 4096.0
        ),
        "model_assets_within_500_mb": (
            float(model_reports["e5_small_v2"]["assets"]["mb"]) <= 500.0
        ),
    }
    gate_passed = all(checks.values())

    observations = []
    for record in records:
        key = str(record["record_id"])
        observations.append(
            {
                key_name: record[key_name]
                for key_name in (
                    "record_id",
                    "sample_id",
                    "turn",
                    "period",
                    "scenario_type",
                    "outer_fold",
                )
            }
            | {
                "sparse_target_rank_at_200": target_rank(
                    sparse_rankings[key], str(record["target"])
                ),
                "minilm_target_rank_at_200": target_rank(
                    model_rankings["minilm_control"][key], str(record["target"])
                ),
                "e5_target_rank_at_200": target_rank(
                    model_rankings["e5_small_v2"][key], str(record["target"])
                ),
            }
        )

    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "evaluation_label": "outer-fold/out-of-fold",
        "split": "nested_v1",
        "sample_count": len(samples),
        "query_record_count": len(records),
        "holdout_accessed": False,
        "failure_status": None,
        "fit": {
            "learned_parameters": False,
            "training_ids": [],
            "validation_ids": sorted(allowed),
            "fold_count": len(nested["outer_folds"]),
        },
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "asset_manifest_sha256": sha256_file(asset_manifest_path),
            "split_sha256": sha256_file(nested_path),
            "catalog_sha256": sha256_file(ROOT / "data/catalog.jsonl"),
            "public_set_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
            "dense_code_sha256": sha256_file(ROOT / "ghostlab/retrieval/dense.py"),
            "runner_code_sha256": sha256_file(ROOT / "scripts/run_dense_retrieval.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "sentence_transformers": importlib.metadata.version(
                "sentence-transformers"
            ),
            "torch": importlib.metadata.version("torch"),
            "seed": 20260826,
            "external_calls_during_runtime": 0,
        },
        "sparse_control": {
            "weights": CHAMPION_FIELD_WEIGHTS,
            "retrieval": route_report(records, sparse_rankings),
            "performance": {
                "warm_latency_ms_mean": round(statistics.fmean(sparse_latencies), 6),
                "warm_latency_ms_p95": round(percentile(sparse_latencies, 0.95), 6),
            },
        },
        "models": model_reports,
        "first_turn_gate_comparison": {
            "minilm_control": minilm_complement,
            "e5_small_v2": e5_complement,
            "checks": checks,
            "passed": gate_passed,
        },
        "paired_first_turn_e5_vs_minilm": paired_first_turn_evidence(
            records,
            model_rankings["minilm_control"],
            model_rankings["e5_small_v2"],
        ),
        "decision": ("CONTINUE_TO_END_TO_END" if gate_passed else "PARK_STANDALONE"),
        "decision_rationale": (
            "The predeclared recall and packaging gate passed; downstream frozen-fold fusion evaluation is authorized."
            if gate_passed
            else "E5 lost 0.013333 first-turn dense Recall@200 and 0.033333 first-turn union Recall@200 versus MiniLM. Across all query stages it improved dense Recall@200 by 0.078518 but produced fewer dense-only rescues beyond BM25 and a 0.001481 lower union recall, so the added semantic ranking did not create useful coverage. End-to-end fusion selection was not run."
        ),
        "observations": observations,
        "elapsed_seconds": round(time.perf_counter() - run_started, 6),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sparse_first_turn": report["sparse_control"]["retrieval"][
                    "first_turn"
                ],
                "minilm_first_turn": minilm_first,
                "e5_first_turn": e5_first,
                "gate": report["first_turn_gate_comparison"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
