from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.retrieval.dense import E5_SMALL_V2, DenseIndex, sha256_file
from ghostlab.retrieval.query import QUERY_VARIANTS, build_dense_query
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.state.memory import ConversationState
from scripts.run_dense_retrieval import (
    CHAMPION_FIELD_WEIGHTS,
    CHAMPION_QUESTION_SEQUENCE,
    KS,
    complementarity,
    overlap_by_stage,
    percentile,
    route_report,
    target_rank,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "raw_history"
CANDIDATES = ("raw_plus_active", "negation_safe_structured")


def build_records(
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
                    "queries": {
                        variant: build_dense_query(state, variant)
                        for variant in QUERY_VARIANTS
                    },
                }
            )
            if observation.turn > len(CHAMPION_QUESTION_SEQUENCE):
                break
            question = CHAMPION_QUESTION_SEQUENCE[observation.turn - 1]
            state.last_asked_attribute = question
            state.asked_attributes.append(question)
            observation = environment.step(
                {
                    "message": "",
                    "ask_attribute": question,
                    "recommendations": [],
                }
            )
    return records


def union_recall(
    records: list[dict[str, object]],
    sparse: dict[str, list[str]],
    dense: dict[str, list[str]],
    k: int,
) -> float:
    return sum(
        str(record["target"]) in sparse[str(record["record_id"])][:k]
        or str(record["target"]) in dense[str(record["record_id"])][:k]
        for record in records
    ) / len(records)


def union_stability(
    records: list[dict[str, object]],
    sparse: dict[str, list[str]],
    dense: dict[str, list[str]],
) -> dict[str, object]:
    groups: dict[str, dict[str, list[dict[str, object]]]] = {
        "outer_folds": defaultdict(list),
        "scenarios": defaultdict(list),
        "stages": defaultdict(list),
    }
    for record in records:
        groups["outer_folds"][str(record["outer_fold"])].append(record)
        groups["scenarios"][str(record["scenario_type"])].append(record)
        groups["stages"][str(record["turn"])].append(record)
    return {
        name: {
            key: {
                f"recall_at_{k}": round(union_recall(values, sparse, dense, k), 6)
                for k in KS
            }
            for key, values in sorted(grouped.items())
        }
        for name, grouped in groups.items()
    }


def paired_union_evidence(
    records: list[dict[str, object]],
    sparse: dict[str, list[str]],
    control: dict[str, list[str]],
    candidate: dict[str, list[str]],
    *,
    k: int = 200,
) -> dict[str, object]:
    by_session: dict[str, list[float]] = defaultdict(list)
    fold_by_session: dict[str, int] = {}
    for record in records:
        key = str(record["record_id"])
        target = str(record["target"])
        sparse_hit = target in sparse[key][:k]
        control_hit = sparse_hit or target in control[key][:k]
        candidate_hit = sparse_hit or target in candidate[key][:k]
        sample_id = str(record["sample_id"])
        by_session[sample_id].append(float(candidate_hit) - float(control_hit))
        fold_by_session[sample_id] = int(record["outer_fold"])
    deltas = [statistics.fmean(values) for _, values in sorted(by_session.items())]
    lower, upper = bootstrap_mean_interval(deltas, resamples=10000, seed=20260826)
    folds: dict[int, list[float]] = defaultdict(list)
    for sample_id, values in by_session.items():
        folds[fold_by_session[sample_id]].append(statistics.fmean(values))
    return {
        "mean_paired_session_delta": round(statistics.fmean(deltas), 6),
        "paired_bootstrap_95_ci": [round(lower, 6), round(upper, 6)],
        "paired_randomization_p_value": round(
            paired_randomization_pvalue(deltas, resamples=10000, seed=20260826), 6
        ),
        "wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
        "fold_deltas": {
            str(fold): round(statistics.fmean(values), 6)
            for fold, values in sorted(folds.items())
        },
    }


def behavioral_dedup(
    records: list[dict[str, object]],
    rankings: dict[str, dict[str, list[str]]],
) -> dict[str, object]:
    pairs = []
    for left_index, left in enumerate(QUERY_VARIANTS):
        for right in QUERY_VARIANTS[left_index + 1 :]:
            identical_queries = 0
            identical_top200 = 0
            top10_jaccards = []
            for record in records:
                key = str(record["record_id"])
                queries = dict(record["queries"])
                identical_queries += queries[left] == queries[right]
                left_rank = rankings[left][key]
                right_rank = rankings[right][key]
                identical_top200 += left_rank == right_rank
                intersection = len(set(left_rank[:10]) & set(right_rank[:10]))
                top10_jaccards.append(intersection / (20 - intersection))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "identical_query_count": identical_queries,
                    "identical_top200_count": identical_top200,
                    "mean_top10_jaccard": round(statistics.fmean(top10_jaccards), 6),
                }
            )
    unique_signatures = {
        variant: len(
            {
                hashlib.sha256("\n".join(ranking).encode()).hexdigest()
                for ranking in rankings[variant].values()
            }
        )
        for variant in QUERY_VARIANTS
    }
    return {"pairs": pairs, "unique_top200_signatures": unique_signatures}


def main() -> None:
    started = time.perf_counter()
    manifest_path = ROOT / "configs/experiments/dense_query_interaction_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False:
        raise RuntimeError("protected holdout must remain sealed")

    cache_dir = ROOT / "artifacts/cache/dense"
    cache_files = list(cache_dir.glob("e5_small_v2-*.npy"))
    metadata_files = list(cache_dir.glob("e5_small_v2-*.json"))
    if len(cache_files) != 1 or len(metadata_files) != 1:
        raise RuntimeError("exactly one prebuilt E5 embedding cache is required")

    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    allowed = {str(value) for value in nested["adaptive_sample_ids"]}
    fold_by_sample = {
        str(sample_id): fold_index
        for fold_index, fold in enumerate(nested["outer_folds"])
        for sample_id in fold
    }
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in allowed
    ]
    _, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    records = build_records(samples, categories, products, fold_by_sample)
    if len(records) != 1350:
        raise RuntimeError("query interaction requires exactly 1,350 staged queries")

    sparse = SparseIndex(ROOT / "data/catalog.jsonl")
    sparse_rankings: dict[str, list[str]] = {}
    for record in records:
        key = str(record["record_id"])
        raw_query = str(dict(record["queries"])[CONTROL])
        sparse_rankings[key] = [
            item.parent_asin
            for item in sparse.search(raw_query, 200, CHAMPION_FIELD_WEIGHTS).items
        ]

    dense = DenseIndex(
        ROOT / "data/catalog.jsonl",
        E5_SMALL_V2,
        cache_dir=cache_dir,
        model_path=ROOT / "artifacts/cache/models/e5-small-v2",
        local_files_only=True,
    )
    if not dense.cache_metadata["cache_hit"]:
        raise RuntimeError("interaction is forbidden from building a new E5 index")

    rankings: dict[str, dict[str, list[str]]] = {}
    model_reports: dict[str, dict[str, object]] = {}
    for variant in QUERY_VARIANTS:
        queries = [str(dict(record["queries"])[variant]) for record in records]
        values = dense.search_many(queries, 200)
        route = {
            str(record["record_id"]): ranking
            for record, ranking in zip(records, values, strict=True)
        }
        rankings[variant] = route
        first_queries = [
            str(dict(record["queries"])[variant])
            for record in records
            if record["turn"] == 1
        ]
        latencies = [dense.search(query, 200).elapsed_ms for query in first_queries]
        first_records = [record for record in records if record["turn"] == 1]
        model_reports[variant] = {
            "dense_retrieval": route_report(records, route),
            "first_turn_complementarity": complementarity(
                first_records, sparse_rankings, route
            ),
            "all_stage_complementarity": complementarity(
                records, sparse_rankings, route
            ),
            "union_stability": union_stability(records, sparse_rankings, route),
            "overlap_with_sparse_by_stage": overlap_by_stage(
                records, sparse_rankings, route
            ),
            "latency": {
                "warm_ms_mean": round(statistics.fmean(latencies), 6),
                "warm_ms_p50": round(percentile(latencies, 0.5), 6),
                "warm_ms_p95": round(percentile(latencies, 0.95), 6),
                "sample_count": len(latencies),
            },
        }

    priority = {"raw_plus_active": 0, "negation_safe_structured": 1}
    selected = min(
        CANDIDATES,
        key=lambda variant: (
            -float(
                model_reports[variant]["all_stage_complementarity"]["at_200"][
                    "union_recall"
                ]
            ),
            priority[variant],
        ),
    )
    evidence = {
        variant: paired_union_evidence(
            records, sparse_rankings, rankings[CONTROL], rankings[variant]
        )
        for variant in CANDIDATES
    }
    control_all = model_reports[CONTROL]["all_stage_complementarity"]["at_200"]
    selected_all = model_reports[selected]["all_stage_complementarity"]["at_200"]
    control_first = model_reports[CONTROL]["first_turn_complementarity"]["at_200"]
    selected_first = model_reports[selected]["first_turn_complementarity"]["at_200"]
    fold_deltas = evidence[selected]["fold_deltas"]
    checks = {
        "all_stage_union_recall_gain_at_least_0_005": (
            float(selected_all["union_recall"])
            >= float(control_all["union_recall"]) + 0.005
        ),
        "at_least_3_more_dense_only_unique_session_rescues": (
            int(selected_all["dense_only_unique_sessions"])
            >= int(control_all["dense_only_unique_sessions"]) + 3
        ),
        "first_turn_union_recall_degradation_at_most_0_01": (
            float(selected_first["union_recall"])
            >= float(control_first["union_recall"]) - 0.01
        ),
        "non_negative_union_delta_in_at_least_4_folds": sum(
            float(value) >= 0.0 for value in dict(fold_deltas).values()
        )
        >= 4,
        "warm_p95_within_500_ms": (
            float(model_reports[selected]["latency"]["warm_ms_p95"]) <= 500.0
        ),
    }
    gate_passed = all(checks.values())

    observations = []
    for record in records:
        key = str(record["record_id"])
        observations.append(
            {
                name: record[name]
                for name in (
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
                "dense_target_ranks_at_200": {
                    variant: target_rank(rankings[variant][key], str(record["target"]))
                    for variant in QUERY_VARIANTS
                },
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
        "model_search_performed": False,
        "index_build_performed": False,
        "sparse_query_channel": "unchanged_raw_history",
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "split_sha256": sha256_file(nested_path),
            "catalog_sha256": sha256_file(ROOT / "data/catalog.jsonl"),
            "query_code_sha256": sha256_file(ROOT / "ghostlab/retrieval/query.py"),
            "runner_code_sha256": sha256_file(
                ROOT / "scripts/run_dense_query_interaction.py"
            ),
            "embedding_cache_sha256": sha256_file(cache_files[0]),
            "embedding_metadata_sha256": sha256_file(metadata_files[0]),
            "parent_report_sha256": sha256_file(
                ROOT / "artifacts/reports/dense_retrieval_v1.json"
            ),
        },
        "variants": model_reports,
        "behavioral_deduplication": behavioral_dedup(records, rankings),
        "paired_all_stage_union_at_200_vs_raw": evidence,
        "gate": {
            "selected_candidate": selected,
            "checks": checks,
            "passed": gate_passed,
        },
        "decision": (
            "CONTINUE_MINIMAL_END_TO_END" if gate_passed else "PARK_QUERY_DENSE"
        ),
        "observations": observations,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    output = ROOT / "artifacts/reports/dense_query_interaction_v1.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "gate": report["gate"],
                "decision": report["decision"],
                "variant_summary": {
                    variant: {
                        "first_dense": model_reports[variant]["dense_retrieval"][
                            "first_turn"
                        ],
                        "all_dense": model_reports[variant]["dense_retrieval"][
                            "all_query_stages"
                        ],
                        "first_union": model_reports[variant][
                            "first_turn_complementarity"
                        ],
                        "all_union": model_reports[variant][
                            "all_stage_complementarity"
                        ],
                        "latency": model_reports[variant]["latency"],
                    }
                    for variant in QUERY_VARIANTS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
