from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import ReplayEnvironment, evaluate_replay, session_reward
from ghostlab.retrieval.dense import E5_SMALL_V2, DenseIndex, sha256_file
from ghostlab.retrieval.fusion import sparse_first_union_ids
from ghostlab.retrieval.gbdt import (
    METADATA_FEATURES,
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
)
from ghostlab.retrieval.gbdt_dense import DeepGBDTAgent
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.query import DenseQueryVariant, build_dense_query
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.state.memory import ConversationState
from scripts.run_gbdt_reranker import (
    FIELD_WEIGHTS,
    QUESTION_ORDER,
    RankingGroup,
    build_agent,
    collect_groups,
    paired_evidence,
    summarized_metrics,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/gbdt_dense_interaction_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/gbdt_dense_interaction_v1.json"
REFERENCE_PATH = ROOT / "artifacts/reports/gbdt_reranker_v1.json"
SEED = 20260826
QUERY_CANDIDATES: tuple[DenseQueryVariant, ...] = (
    "raw_plus_active",
    "negation_safe_structured",
)
DENSE_VARIANTS: tuple[DenseQueryVariant, ...] = (
    "raw_history",
    *QUERY_CANDIDATES,
)


@dataclass(frozen=True)
class StageRecord:
    sample_id: str
    target: str
    scenario_type: str
    turn: int
    raw_query: str
    queries: dict[DenseQueryVariant, str]
    sparse: tuple[str, ...]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def process_peak_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def technical_score(sessions: list[dict]) -> float:
    return statistics.fmean(session_reward(session) for session in sessions)


def scenario_scores(sessions: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        key: round(technical_score(values), 6)
        for key, values in sorted(grouped.items())
    }


def sessions_by_id(sessions: list[dict]) -> dict[str, dict]:
    return {str(session["sample_id"]): session for session in sessions}


def sessions_equal(left: list[dict], right: list[dict]) -> bool:
    return sessions_by_id(left) == sessions_by_id(right)


def collect_stage_records(
    samples: dict[str, dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
) -> list[StageRecord]:
    records: list[StageRecord] = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        environment = ReplayEnvironment(sample, categories, products)
        state = ConversationState(sample_id, sample["user_profile"])
        observation = environment.observe()
        while observation is not None and observation.turn <= 9:
            state.observe(observation.user_message, observation.turn)
            raw_query = ". ".join(state.messages)
            records.append(
                StageRecord(
                    sample_id=sample_id,
                    target=str(sample["ground_truth"]["parent_asin"]),
                    scenario_type=str(sample["scenario_type"]),
                    turn=observation.turn,
                    raw_query=raw_query,
                    queries={
                        variant: build_dense_query(state, variant)
                        for variant in DENSE_VARIANTS
                    },
                    sparse=tuple(
                        item.parent_asin
                        for item in sparse.search(raw_query, 200, FIELD_WEIGHTS).items
                    ),
                )
            )
            if observation.turn > len(QUESTION_ORDER):
                break
            question = QUESTION_ORDER[observation.turn - 1]
            state.last_asked_attribute = question
            state.asked_attributes.append(question)
            next_observation = environment.step(
                {
                    "message": "training trajectory",
                    "ask_attribute": question,
                    "recommendations": [],
                }
            )
            observation = next_observation
    return records


def dense_rankings(
    records: list[StageRecord], dense: DenseIndex
) -> dict[DenseQueryVariant, list[list[str]]]:
    return {
        variant: dense.search_many([record.queries[variant] for record in records], 200)
        for variant in DENSE_VARIANTS
    }


def union_recall(
    records: list[StageRecord],
    rankings: list[list[str]],
    allowed_ids: set[str],
) -> float:
    selected = [
        (record, dense)
        for record, dense in zip(records, rankings, strict=True)
        if record.sample_id in allowed_ids
    ]
    if not selected:
        raise ValueError("query selection population cannot be empty")
    return sum(
        record.target in record.sparse or record.target in dense
        for record, dense in selected
    ) / len(selected)


def select_queries(
    records: list[StageRecord],
    rankings: dict[DenseQueryVariant, list[list[str]]],
    adaptive_ids: set[str],
    outer_folds: list[set[str]],
) -> list[dict[str, object]]:
    choices = []
    priority = {"raw_plus_active": 0, "negation_safe_structured": 1}
    for outer_index, outer_ids in enumerate(outer_folds):
        training_ids = adaptive_ids - outer_ids
        scores = {
            variant: union_recall(records, rankings[variant], training_ids)
            for variant in QUERY_CANDIDATES
        }
        selected = min(
            QUERY_CANDIDATES,
            key=lambda variant: (-scores[variant], priority[variant]),
        )
        choices.append(
            {
                "outer_fold": outer_index,
                "selection_population_ids": sorted(training_ids),
                "selection_population_records": sum(
                    record.sample_id in training_ids for record in records
                ),
                "all_stage_union_recall_at_200": {
                    key: round(value, 6) for key, value in scores.items()
                },
                "selected_query": selected,
                "tie_break_used": len(set(scores.values())) == 1,
            }
        )
    return choices


def build_candidate_groups(
    records: list[StageRecord],
    rankings: list[list[str]],
    quality: CatalogQualityReranker,
    features: GBDTFeatureStore,
) -> tuple[dict[str, list[RankingGroup]], dict[str, int]]:
    groups: dict[str, list[RankingGroup]] = defaultdict(list)
    positives = rows = 0
    for record, dense_ids in zip(records, rankings, strict=True):
        union = sparse_first_union_ids(record.sparse, dense_ids, limit=400)
        if union[: len(record.sparse)] != list(record.sparse):
            raise RuntimeError("candidate construction changed the sparse head")
        ranked = quality.rerank(union, weight=0.2, rerank_k=400)
        if record.target not in ranked:
            continue
        positives += 1
        labels = tuple(int(identifier == record.target) for identifier in ranked)
        groups[record.sample_id].append(
            RankingGroup(
                sample_id=record.sample_id,
                turn=record.turn,
                query=record.raw_query,
                candidates=tuple(ranked),
                labels=labels,
                matrix=features.matrix(record.raw_query, ranked, METADATA_FEATURES),
            )
        )
        rows += len(ranked)
    return dict(groups), {
        "trajectory_queries": len(records),
        "queries_with_target_in_union": positives,
        "samples_with_ranking_groups": len(groups),
        "ranking_groups": sum(len(values) for values in groups.values()),
        "candidate_rows": rows,
    }


def build_sparse_deep_groups(
    records: list[StageRecord],
    quality: CatalogQualityReranker,
    features: GBDTFeatureStore,
) -> tuple[dict[str, list[RankingGroup]], dict[str, int]]:
    groups: dict[str, list[RankingGroup]] = defaultdict(list)
    positives = rows = 0
    for record in records:
        ranked = quality.rerank(list(record.sparse), weight=0.2, rerank_k=200)
        if record.target not in ranked:
            continue
        positives += 1
        groups[record.sample_id].append(
            RankingGroup(
                sample_id=record.sample_id,
                turn=record.turn,
                query=record.raw_query,
                candidates=tuple(ranked),
                labels=tuple(int(identifier == record.target) for identifier in ranked),
                matrix=features.matrix(record.raw_query, ranked, METADATA_FEATURES),
            )
        )
        rows += len(ranked)
    return dict(groups), {
        "trajectory_queries": len(records),
        "queries_with_target_in_sparse_top200": positives,
        "samples_with_ranking_groups": len(groups),
        "ranking_groups": sum(len(values) for values in groups.values()),
        "candidate_rows": rows,
    }


def evaluate_deep_arm(
    sample_ids: set[str],
    samples: dict[str, dict],
    catalog_path: Path,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    dense: DenseIndex | None,
    quality: CatalogQualityReranker,
    features: GBDTFeatureStore,
    model: LambdaMARTModel,
    query_variant: DenseQueryVariant | None,
) -> tuple[dict, DeepGBDTAgent]:
    agent = DeepGBDTAgent(
        catalog_path,
        sparse=sparse,
        dense=dense,
        quality=quality,
        reranker=LambdaMARTReranker(features, model),
        field_weights=FIELD_WEIGHTS,
        question_order=QUESTION_ORDER,
        dense_query_variant=query_variant,
    )
    result = evaluate(
        agent,
        [samples[sample_id] for sample_id in sorted(sample_ids)],
        catalog_ids,
        categories,
        products,
    )
    return result, agent


def model_bytes(model: LambdaMARTModel) -> bytes:
    return json.dumps(asdict(model), sort_keys=True, separators=(",", ":")).encode()


def hash_tree(path: Path) -> tuple[int, dict[str, str]]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return sum(item.stat().st_size for item in files), {
        str(item.relative_to(path)): sha256_file(item) for item in files
    }


def matched_control_diagnostic(
    candidate: list[dict], reference: list[dict], tolerance: float
) -> dict[str, Any]:
    candidate_by_id = sessions_by_id(candidate)
    reference_by_id = sessions_by_id(reference)
    ids_match = set(candidate_by_id) == set(reference_by_id)
    deltas = (
        [
            session_reward(candidate_by_id[key]) - session_reward(reference_by_id[key])
            for key in sorted(candidate_by_id)
        ]
        if ids_match
        else []
    )
    return {
        "sample_ids_match": ids_match,
        "max_absolute_session_reward_delta": (
            max((abs(value) for value in deltas), default=0.0)
        ),
        "sessions_exact": candidate_by_id == reference_by_id,
        "passed": ids_match and all(abs(value) <= tolerance for value in deltas),
    }


def write_failure_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_deep_arm(
    *,
    arm_id: str,
    groups_by_fold: list[dict[str, list[RankingGroup]]],
    query_by_fold: list[DenseQueryVariant | None],
    dense: DenseIndex | None,
    config: dict,
    adaptive_ids: set[str],
    outer_folds: list[set[str]],
    samples: dict[str, dict],
    catalog_path: Path,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    quality: CatalogQualityReranker,
    features: GBDTFeatureStore,
) -> dict[str, Any]:
    sessions: list[dict] = []
    folds = []
    models = []
    latencies: list[float] = []
    failures = 0
    for outer_index, outer_ids in enumerate(outer_folds):
        inner_validation_ids = outer_folds[(outer_index + 1) % len(outer_folds)]
        inner_training_ids = adaptive_ids - outer_ids - inner_validation_ids
        model, rounds = train_model(
            groups_by_fold[outer_index],
            inner_training_ids,
            inner_validation_ids,
            config,
        )
        result, agent = evaluate_deep_arm(
            outer_ids,
            samples,
            catalog_path,
            catalog_ids,
            categories,
            products,
            sparse,
            dense,
            quality,
            features,
            model,
            query_by_fold[outer_index],
        )
        sessions.extend(result["sessions"])
        models.append(model)
        latencies.extend(agent.latencies_ms)
        failures += agent.failure_count
        folds.append(
            {
                "outer_fold": outer_index,
                "query": query_by_fold[outer_index],
                "selected_rounds": rounds,
                "metrics": summarized_metrics(result["sessions"]),
                "feature_importance": model.split_importance(),
            }
        )
        print(
            f"{arm_id} fold={outer_index} query={query_by_fold[outer_index]} "
            f"rounds={rounds}",
            flush=True,
        )
    return {
        "arm_id": arm_id,
        "sessions": sessions,
        "metrics": summarized_metrics(sessions),
        "folds": folds,
        "models": models,
        "latencies_ms": latencies,
        "failure_count": failures,
    }


def comparison(
    candidate: dict[str, Any],
    control: dict[str, Any],
    outer_folds: list[set[str]],
) -> dict[str, Any]:
    candidate_sessions = list(candidate["sessions"])
    control_sessions = list(control["sessions"])
    candidate_metrics = dict(candidate["metrics"])
    control_metrics = dict(control["metrics"])
    candidate_scenarios = scenario_scores(candidate_sessions)
    control_scenarios = scenario_scores(control_sessions)
    fold_deltas = []
    for fold in outer_folds:
        candidate_fold = [
            session
            for session in candidate_sessions
            if str(session["sample_id"]) in fold
        ]
        control_fold = [
            session for session in control_sessions if str(session["sample_id"]) in fold
        ]
        fold_deltas.append(
            round(technical_score(candidate_fold) - technical_score(control_fold), 6)
        )
    return {
        "technical_score_delta": round(
            float(candidate_metrics["recommended_technical_score"])
            - float(control_metrics["recommended_technical_score"]),
            6,
        ),
        "hit_rate_delta": round(
            float(candidate_metrics["hit_rate_at_10"])
            - float(control_metrics["hit_rate_at_10"]),
            6,
        ),
        "fold_deltas": fold_deltas,
        "nonnegative_fold_count": sum(value >= 0.0 for value in fold_deltas),
        "scenario_deltas": {
            key: round(candidate_scenarios[key] - control_scenarios[key], 6)
            for key in control_scenarios
        },
        "paired_evidence": paired_evidence(candidate_sessions, control_sessions),
    }


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False:
        raise RuntimeError("protected holdout must remain sealed")
    if (
        tuple(manifest["nested_query_candidate"]["dense_query_candidates"])
        != QUERY_CANDIDATES
    ):
        raise RuntimeError("runner query candidates differ from predeclared manifest")

    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    sparse = SparseIndex(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    features = GBDTFeatureStore(catalog_path, quality=quality.quality)

    control_groups, control_collection = collect_groups(
        samples, categories, products, sparse, quality, features
    )
    config = {
        "candidate_id": "shallow_metadata_depth3",
        "feature_set": "metadata",
        "max_depth": 3,
        "num_leaves": 7,
        "learning_rate": 0.03,
        "max_rounds": 160,
        "early_stopping_rounds": 20,
    }
    control_sessions: list[dict] = []
    control_folds = []
    for outer_index, outer_ids in enumerate(outer_folds):
        inner_validation_ids = outer_folds[(outer_index + 1) % len(outer_folds)]
        inner_training_ids = adaptive_ids - outer_ids - inner_validation_ids
        model, rounds = train_model(
            control_groups, inner_training_ids, inner_validation_ids, config
        )
        result = evaluate(
            build_agent(quality, LambdaMARTReranker(features, model)),
            [samples[sample_id] for sample_id in sorted(outer_ids)],
            catalog_ids,
            categories,
            products,
        )
        control_sessions.extend(result["sessions"])
        control_folds.append(
            {
                "outer_fold": outer_index,
                "selected_rounds": rounds,
                "metrics": summarized_metrics(result["sessions"]),
            }
        )
        print(f"control fold={outer_index} rounds={rounds}", flush=True)

    reference_report = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    reference_sessions = reference_report["variants"]["shallow_metadata_depth3"][
        "oof_sessions"
    ]
    reproduction = matched_control_diagnostic(
        control_sessions,
        reference_sessions,
        float(manifest["matched_control_reproduction"]["session_reward_tolerance"]),
    )
    control_metrics = summarized_metrics(control_sessions)
    reference_metrics = reference_report["variants"]["shallow_metadata_depth3"][
        "oof_metrics"
    ]
    reproduction["technical_score_delta"] = round(
        float(control_metrics["recommended_technical_score"])
        - float(reference_metrics["recommended_technical_score"]),
        9,
    )
    reproduction["passed"] = bool(reproduction["passed"]) and abs(
        float(reproduction["technical_score_delta"])
    ) <= float(manifest["matched_control_reproduction"]["technical_score_tolerance"])
    if not reproduction["passed"]:
        report = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "holdout_accessed": False,
            "failure_status": "MATCHED_CONTROL_REPRODUCTION_FAILED",
            "control_reproduction": reproduction,
            "control_metrics": control_metrics,
            "candidate_evaluated": False,
        }
        write_failure_report(report)
        raise RuntimeError("matched GBDT control did not reproduce; candidate not run")

    cache_dir = ROOT / "artifacts/cache/dense"
    cache_files = list(cache_dir.glob("e5_small_v2-*.npy"))
    metadata_files = list(cache_dir.glob("e5_small_v2-*.json"))
    if len(cache_files) != 1 or len(metadata_files) != 1:
        raise RuntimeError("exactly one verified prebuilt E5 index is required")
    dense = DenseIndex(
        catalog_path,
        E5_SMALL_V2,
        cache_dir=cache_dir,
        model_path=ROOT / "artifacts/cache/models/e5-small-v2",
        local_files_only=True,
    )
    if not dense.cache_metadata["cache_hit"]:
        raise RuntimeError("interaction run is forbidden from building an index")

    records = collect_stage_records(samples, categories, products, sparse)
    if len(records) != 1350:
        raise RuntimeError("interaction requires exactly 1,350 staged queries")
    rankings = dense_rankings(records, dense)
    choices = select_queries(records, rankings, adaptive_ids, outer_folds)
    sparse_deep_groups, sparse_deep_collection = build_sparse_deep_groups(
        records, quality, features
    )
    dense_groups: dict[DenseQueryVariant, dict[str, list[RankingGroup]]] = {}
    dense_collection: dict[DenseQueryVariant, dict[str, int]] = {}
    for variant in DENSE_VARIANTS:
        groups, collection = build_candidate_groups(
            records, rankings[variant], quality, features
        )
        dense_groups[variant] = groups
        dense_collection[variant] = collection

    arm_a: dict[str, Any] = {
        "arm_id": "A_current_gbdt_top50",
        "sessions": control_sessions,
        "metrics": control_metrics,
        "folds": control_folds,
        "failure_count": 0,
    }
    arm_b = run_deep_arm(
        arm_id="B_sparse_deep_gbdt",
        groups_by_fold=[sparse_deep_groups] * len(outer_folds),
        query_by_fold=[None] * len(outer_folds),
        dense=None,
        config=config,
        adaptive_ids=adaptive_ids,
        outer_folds=outer_folds,
        samples=samples,
        catalog_path=catalog_path,
        catalog_ids=catalog_ids,
        categories=categories,
        products=products,
        sparse=sparse,
        quality=quality,
        features=features,
    )
    arm_c = run_deep_arm(
        arm_id="C_raw_e5_union_deep_gbdt",
        groups_by_fold=[dense_groups["raw_history"]] * len(outer_folds),
        query_by_fold=["raw_history"] * len(outer_folds),
        dense=dense,
        config=config,
        adaptive_ids=adaptive_ids,
        outer_folds=outer_folds,
        samples=samples,
        catalog_path=catalog_path,
        catalog_ids=catalog_ids,
        categories=categories,
        products=products,
        sparse=sparse,
        quality=quality,
        features=features,
    )
    selected_queries: list[DenseQueryVariant] = []
    selected_groups: list[dict[str, list[RankingGroup]]] = []
    for choice in choices:
        selected = str(choice["selected_query"])
        if selected not in QUERY_CANDIDATES:
            raise RuntimeError("nested query selector returned an undeclared query")
        selected_queries.append(selected)  # type: ignore[arg-type]
        selected_groups.append(dense_groups[selected])  # type: ignore[index]
    arm_d = run_deep_arm(
        arm_id="D_nested_query_e5_union_deep_gbdt",
        groups_by_fold=selected_groups,
        query_by_fold=list(selected_queries),
        dense=dense,
        config=config,
        adaptive_ids=adaptive_ids,
        outer_folds=outer_folds,
        samples=samples,
        catalog_path=catalog_path,
        catalog_ids=catalog_ids,
        categories=categories,
        products=products,
        sparse=sparse,
        quality=quality,
        features=features,
    )

    comparisons = {
        "B_minus_A_depth_only": comparison(arm_b, arm_a, outer_folds),
        "C_minus_B_raw_dense": comparison(arm_c, arm_b, outer_folds),
        "D_minus_C_structured_query": comparison(arm_d, arm_c, outer_folds),
        "D_minus_B_dense_plus_structured": comparison(arm_d, arm_b, outer_folds),
        "D_minus_A_total": comparison(arm_d, arm_a, outer_folds),
    }
    d_models = list(arm_d["models"])
    d_folds = list(arm_d["folds"])
    first_choice = selected_queries[0]
    repeat_model, repeat_rounds = train_model(
        selected_groups[0],
        adaptive_ids - outer_folds[0] - outer_folds[1],
        outer_folds[1],
        config,
    )
    independent_refit = {
        "outer_fold": 0,
        "selected_rounds_match": repeat_rounds == int(d_folds[0]["selected_rounds"]),
        "serialized_bytes_match": model_bytes(repeat_model) == model_bytes(d_models[0]),
    }
    independent_refit["passed"] = all(
        bool(value) for key, value in independent_refit.items() if key != "outer_fold"
    )

    repeat_sessions: list[dict] = []
    repeat_failures = 0
    for outer_index, outer_ids in enumerate(outer_folds):
        result, agent = evaluate_deep_arm(
            outer_ids,
            samples,
            catalog_path,
            catalog_ids,
            categories,
            products,
            sparse,
            dense,
            quality,
            features,
            d_models[outer_index],
            selected_queries[outer_index],
        )
        repeat_sessions.extend(result["sessions"])
        repeat_failures += agent.failure_count
    d_sessions = list(arm_d["sessions"])
    deterministic_replay = {
        "sessions_exact": sessions_equal(d_sessions, repeat_sessions),
        "failure_count": repeat_failures,
        "passed": sessions_equal(d_sessions, repeat_sessions) and repeat_failures == 0,
    }

    parity_agent = DeepGBDTAgent(
        catalog_path,
        sparse=sparse,
        dense=dense,
        quality=quality,
        reranker=LambdaMARTReranker(features, d_models[0]),
        field_weights=FIELD_WEIGHTS,
        question_order=QUESTION_ORDER,
        dense_query_variant=first_choice,
    )
    parity_result = evaluate_replay(
        parity_agent,
        [samples[sample_id] for sample_id in sorted(outer_folds[0])],
        categories,
        products,
    )
    evaluator_fold_zero = [
        session for session in d_sessions if str(session["sample_id"]) in outer_folds[0]
    ]
    adapter_parity = {
        "sessions_exact": sessions_equal(
            evaluator_fold_zero, parity_result["sessions"]
        ),
        "failure_count": parity_agent.failure_count,
    }
    adapter_parity["passed"] = (
        bool(adapter_parity["sessions_exact"])
        and int(adapter_parity["failure_count"]) == 0
    )

    model_dir = ROOT / "artifacts/cache/models/e5-small-v2"
    model_bytes_total, model_hashes = hash_tree(model_dir)
    index_bytes = cache_files[0].stat().st_size + metadata_files[0].stat().st_size
    unique_asset_mb = (model_bytes_total + index_bytes) / (1024 * 1024)
    d_latencies = list(arm_d["latencies_ms"])
    total_failures = sum(
        int(arm["failure_count"]) for arm in (arm_a, arm_b, arm_c, arm_d)
    )
    performance = {
        "turn_count": len(d_latencies),
        "warm_latency_ms_mean": round(statistics.fmean(d_latencies), 6),
        "warm_latency_ms_p50": round(percentile(d_latencies, 0.5), 6),
        "warm_latency_ms_p95": round(percentile(d_latencies, 0.95), 6),
        "warm_latency_ms_max": round(max(d_latencies), 6),
        "peak_process_memory_mb": round(process_peak_mb(), 6),
        "unique_model_and_index_asset_mb": round(unique_asset_mb, 6),
        "failure_count_all_arms": total_failures,
        "external_calls_per_turn": 0,
        "offline_runtime": True,
        "model_load_seconds": round(dense.model_load_seconds, 6),
        "index_cache_hit": bool(dense.cache_metadata["cache_hit"]),
    }
    d_vs_a = comparisons["D_minus_A_total"]
    d_vs_b = comparisons["D_minus_B_dense_plus_structured"]
    d_vs_c = comparisons["D_minus_C_structured_query"]
    checks = {
        "technical_score_gain_vs_A_at_least_0_005": float(
            d_vs_a["technical_score_delta"]
        )
        >= 0.005,
        "technical_score_gain_vs_B_at_least_0_005": float(
            d_vs_b["technical_score_delta"]
        )
        >= 0.005,
        "technical_score_gain_vs_C_at_least_0_005": float(
            d_vs_c["technical_score_delta"]
        )
        >= 0.005,
        "nonnegative_in_at_least_4_folds_vs_B": int(d_vs_b["nonnegative_fold_count"])
        >= 4,
        "nonnegative_in_at_least_4_folds_vs_C": int(d_vs_c["nonnegative_fold_count"])
        >= 4,
        "hit_rate_does_not_regress_vs_B_or_C": float(d_vs_b["hit_rate_delta"]) >= 0.0
        and float(d_vs_c["hit_rate_delta"]) >= 0.0,
        "no_scenario_regression_worse_than_0_005_vs_B_or_C": min(
            *[float(value) for value in dict(d_vs_b["scenario_deltas"]).values()],
            *[float(value) for value in dict(d_vs_c["scenario_deltas"]).values()],
        )
        >= -0.005,
        "failure_count_zero": total_failures == 0,
        "warm_p95_within_500_ms": performance["warm_latency_ms_p95"] <= 500.0,
        "peak_memory_within_4096_mb": performance["peak_process_memory_mb"] <= 4096.0,
        "assets_within_500_mb": performance["unique_model_and_index_asset_mb"] <= 500.0,
        "offline_and_no_external_calls": performance["offline_runtime"]
        and performance["external_calls_per_turn"] == 0,
        "deterministic_refit": bool(independent_refit["passed"]),
        "deterministic_replay": bool(deterministic_replay["passed"]),
        "adapter_parity": bool(adapter_parity["passed"]),
    }
    passed = all(checks.values())

    def report_arm(arm: dict[str, Any]) -> dict[str, object]:
        return {
            "arm_id": arm["arm_id"],
            "metrics": arm["metrics"],
            "folds": arm["folds"],
            "failure_count": arm["failure_count"],
            "sessions": arm["sessions"],
        }

    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "evaluation_label": "nested OOF with matched-depth attribution arms",
        "split": "nested_v1",
        "sample_count": len(d_sessions),
        "holdout_accessed": False,
        "arm_count": 4,
        "query_candidates_inside_outer_training": list(QUERY_CANDIDATES),
        "model_search_performed": False,
        "index_build_performed": False,
        "control_reproduction": reproduction,
        "collections": {
            "A_current_top50": control_collection,
            "B_sparse_deep": sparse_deep_collection,
            "dense_deep_by_query": dense_collection,
        },
        "query_selection_by_outer_fold": choices,
        "arms": {
            "A": report_arm(arm_a),
            "B": report_arm(arm_b),
            "C": report_arm(arm_c),
            "D": report_arm(arm_d),
        },
        "attribution": comparisons,
        "backward_ablations": {
            "remove_deep_budget_from_B": "B_minus_A_depth_only",
            "remove_raw_dense_union_from_C": "C_minus_B_raw_dense",
            "replace_nested_structured_query_with_raw_in_D": "D_minus_C_structured_query",
            "remove_complete_dense_query_channel_from_D": "D_minus_B_dense_plus_structured",
        },
        "determinism": {
            "independent_refit": independent_refit,
            "replay": deterministic_replay,
        },
        "adapter_parity": adapter_parity,
        "performance": performance,
        "gate_for_arm_D": {"checks": checks, "passed": passed},
        "decision": (
            "PROMOTE_DENSE_QUERY_INTERACTION"
            if passed
            else "PARK_STRUCTURED_DENSE_QUERY_INTERACTION"
        ),
        "depth_only_diagnostic": {
            "technical_score_delta_B_minus_A": comparisons["B_minus_A_depth_only"][
                "technical_score_delta"
            ],
            "requires_distinct_technique_decision": True,
        },
        "failure_status": None,
        "asset_hashes": {
            "model_files": model_hashes,
            "embedding_index_sha256": sha256_file(cache_files[0]),
            "embedding_metadata_sha256": sha256_file(metadata_files[0]),
        },
        "hashes": {
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "runner_sha256": sha256_file(
                ROOT / "scripts/run_gbdt_dense_interaction.py"
            ),
            "gbdt_dense_sha256": sha256_file(ROOT / "ghostlab/retrieval/gbdt_dense.py"),
            "split_sha256": sha256_file(nested_path),
            "public_data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
            "catalog_sha256": sha256_file(catalog_path),
            "reference_report_sha256": sha256_file(REFERENCE_PATH),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_failure_report(report)
    print(
        json.dumps(
            {
                "arm_metrics": {
                    key: value["metrics"] for key, value in report["arms"].items()
                },
                "attribution": comparisons,
                "query_selection": choices,
                "performance": performance,
                "gate_for_arm_D": report["gate_for_arm_D"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
