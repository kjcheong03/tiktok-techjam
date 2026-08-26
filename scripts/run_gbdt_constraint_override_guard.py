from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import cast

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.retrieval.constraint_gbdt import (
    ConstraintAgentAdapter,
    ConstraintGBDTFeatureStore,
    RuntimeConstraintReranker,
)
from ghostlab.retrieval.gbdt import LambdaMARTModel, LambdaMARTReranker
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.experimental import ExperimentalAgent
from scripts.run_gbdt_constraint_interaction import (
    FIELD_WEIGHTS,
    QUESTION_ORDER,
    ROOT,
    build_agent,
    collect_groups,
    numeric,
    scenario_rewards,
    train_model,
)
from scripts.run_gbdt_reranker import (
    paired_evidence,
    sha256_file,
    stability,
    summarized_metrics,
)
from starter.agent import Agent

AMENDMENT_PATH = (
    ROOT / "configs/experiments/gbdt_constraint_interaction_v1_amendment_2.json"
)
V2_REPORT_PATH = ROOT / "artifacts/reports/gbdt_constraint_interaction_v2.json"
REPORT_PATH = ROOT / "artifacts/reports/gbdt_constraint_override_guard_v1.json"
BASE_MODEL_PATH = ROOT / "artifacts/models/gbdt_reranker_v2_round56.json"
CONSTRAINT_MODEL_PATH = ROOT / "artifacts/models/gbdt_constraint_interaction_v2.json"


def build_guarded_agent(
    quality: CatalogQualityReranker,
    features: ConstraintGBDTFeatureStore,
    base_model: LambdaMARTModel,
    constraint_model: LambdaMARTModel,
) -> tuple[ConstraintAgentAdapter, RuntimeConstraintReranker]:
    contextual = RuntimeConstraintReranker(
        str(ROOT / "data/catalog.jsonl"),
        FIELD_WEIGHTS,
        features,
        constraint_model,
        fallback=LambdaMARTReranker(features, base_model),
    )
    agent = ExperimentalAgent(
        ROOT / "data/catalog.jsonl",
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
    return ConstraintAgentAdapter(agent, contextual), contextual


def routing_summary(
    trace: list[dict[str, object]], ordered_samples: list[dict]
) -> dict[str, object]:
    session_order = list(dict.fromkeys(str(item["session_id"]) for item in trace))
    if len(session_order) != len(ordered_samples):
        raise RuntimeError("routing sessions do not align with evaluation samples")
    sample_by_session = {
        session_id: sample
        for session_id, sample in zip(session_order, ordered_samples, strict=True)
    }
    fallback = [item for item in trace if item["route"] == "base_override_fallback"]
    fallback_sessions = {str(item["session_id"]) for item in fallback}
    reason_turns: Counter[str] = Counter()
    reason_sessions: dict[str, set[str]] = defaultdict(set)
    for item in fallback:
        session_id = str(item["session_id"])
        for reason in cast(list[str], item["reasons"]):
            reason_turns[reason] += 1
            reason_sessions[reason].add(session_id)
    scenario_sessions: Counter[str] = Counter()
    scenario_turns: Counter[str] = Counter()
    sample_ids = []
    for session_id in sorted(fallback_sessions):
        sample = sample_by_session[session_id]
        scenario_sessions[str(sample["scenario_type"])] += 1
        sample_ids.append(str(sample["sample_id"]))
    for item in fallback:
        sample = sample_by_session[str(item["session_id"])]
        scenario_turns[str(sample["scenario_type"])] += 1
    return {
        "total_turns": len(trace),
        "constraint_turns": len(trace) - len(fallback),
        "fallback_turns": len(fallback),
        "total_sessions": len(session_order),
        "fallback_sessions": len(fallback_sessions),
        "fallback_sample_ids": sorted(sample_ids),
        "fallback_turns_by_reason": dict(sorted(reason_turns.items())),
        "fallback_sessions_by_reason": {
            reason: len(sessions)
            for reason, sessions in sorted(reason_sessions.items())
        },
        "fallback_sessions_by_scenario": dict(sorted(scenario_sessions.items())),
        "fallback_turns_by_scenario": dict(sorted(scenario_turns.items())),
    }


def scenario_deltas(candidate: list[dict], control: list[dict]) -> dict[str, float]:
    left = scenario_rewards(candidate)
    right = scenario_rewards(control)
    return {name: round(left[name] - right[name], 6) for name in right}


def main() -> None:
    started = time.perf_counter()
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if (
        amendment["holdout_accessed"] is not False
        or amendment["guarded_outcomes_observed"] is not False
    ):
        raise RuntimeError("guard amendment is not frozen")
    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    rounds = [
        int(value) for value in amendment["frozen_evaluation"]["outer_fold_rounds"]
    ]
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    sparse = SparseIndex(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    features = ConstraintGBDTFeatureStore(catalog_path, quality=quality.quality)
    groups, collection = collect_groups(
        samples, categories, products, sparse, quality, features
    )
    print(json.dumps(collection, sort_keys=True), flush=True)

    base_sessions: list[dict] = []
    unguarded_sessions: list[dict] = []
    guarded_sessions: list[dict] = []
    base_folds: list[dict] = []
    unguarded_folds: list[dict] = []
    guarded_folds: list[dict] = []
    route_folds: list[dict] = []
    deterministic = True
    for fold_index, (outer_ids, fold_rounds) in enumerate(
        zip(outer_folds, rounds, strict=True)
    ):
        training_ids = adaptive_ids - outer_ids
        base_model = train_model(
            groups, training_ids, candidate=False, rounds=fold_rounds
        )
        constraint_model = train_model(
            groups, training_ids, candidate=True, rounds=fold_rounds
        )
        ordered = [samples[sample_id] for sample_id in sorted(outer_ids)]
        base_result = evaluate(
            cast(Agent, build_agent(quality, features, base_model, candidate=False)),
            ordered,
            catalog_ids,
            categories,
            products,
        )
        unguarded_result = evaluate(
            cast(
                Agent,
                build_agent(quality, features, constraint_model, candidate=True),
            ),
            ordered,
            catalog_ids,
            categories,
            products,
        )
        guarded_agent, guarded_runtime = build_guarded_agent(
            quality, features, base_model, constraint_model
        )
        guarded_result = evaluate(
            cast(Agent, guarded_agent),
            ordered,
            catalog_ids,
            categories,
            products,
        )
        repeated_agent, repeated_runtime = build_guarded_agent(
            quality, features, base_model, constraint_model
        )
        repeated_result = evaluate(
            cast(Agent, repeated_agent),
            ordered,
            catalog_ids,
            categories,
            products,
        )
        route = routing_summary(guarded_runtime.routing_trace, ordered)
        repeated_route = routing_summary(repeated_runtime.routing_trace, ordered)
        deterministic = deterministic and (
            guarded_result["sessions"] == repeated_result["sessions"]
            and route == repeated_route
        )
        base_sessions.extend(base_result["sessions"])
        unguarded_sessions.extend(unguarded_result["sessions"])
        guarded_sessions.extend(guarded_result["sessions"])
        shared = {
            "outer_fold": fold_index,
            "outer_training_ids": sorted(training_ids),
            "outer_validation_ids": sorted(outer_ids),
            "frozen_rounds": fold_rounds,
        }
        base_folds.append(
            {**shared, "outer_metrics": summarized_metrics(base_result["sessions"])}
        )
        unguarded_folds.append(
            {
                **shared,
                "outer_metrics": summarized_metrics(unguarded_result["sessions"]),
            }
        )
        guarded_folds.append(
            {
                **shared,
                "outer_metrics": summarized_metrics(guarded_result["sessions"]),
            }
        )
        route_folds.append({"outer_fold": fold_index, **route})
        print(
            f"fold={fold_index} base={base_result['recommended_technical_score']} "
            f"unguarded={unguarded_result['recommended_technical_score']} "
            f"guarded={guarded_result['recommended_technical_score']} "
            f"fallback_sessions={route['fallback_sessions']}",
            flush=True,
        )

    base_metrics = summarized_metrics(base_sessions)
    unguarded_metrics = summarized_metrics(unguarded_sessions)
    guarded_metrics = summarized_metrics(guarded_sessions)
    required_base = 0.861417
    required_unguarded = 0.876283
    tolerance = numeric(
        amendment["promotion_gates"]["exact_control_reproduction_tolerance"]
    )
    base_reproduced = (
        abs(numeric(base_metrics["recommended_technical_score"]) - required_base)
        <= tolerance
    )
    unguarded_reproduced = (
        abs(
            numeric(unguarded_metrics["recommended_technical_score"])
            - required_unguarded
        )
        <= tolerance
    )
    guarded_vs_base = paired_evidence(guarded_sessions, base_sessions)
    guarded_vs_unguarded = paired_evidence(guarded_sessions, unguarded_sessions)
    scenario_vs_base = scenario_deltas(guarded_sessions, base_sessions)
    scenario_vs_unguarded = scenario_deltas(guarded_sessions, unguarded_sessions)
    fold_vs_base = [
        round(
            numeric(
                cast(dict[str, object], guarded["outer_metrics"])[
                    "recommended_technical_score"
                ]
            )
            - numeric(
                cast(dict[str, object], base["outer_metrics"])[
                    "recommended_technical_score"
                ]
            ),
            6,
        )
        for guarded, base in zip(guarded_folds, base_folds, strict=True)
    ]
    fold_vs_unguarded = [
        round(
            numeric(
                cast(dict[str, object], guarded["outer_metrics"])[
                    "recommended_technical_score"
                ]
            )
            - numeric(
                cast(dict[str, object], unguarded["outer_metrics"])[
                    "recommended_technical_score"
                ]
            ),
            6,
        )
        for guarded, unguarded in zip(guarded_folds, unguarded_folds, strict=True)
    ]
    aggregate_route = {
        "total_turns": sum(int(item["total_turns"]) for item in route_folds),
        "constraint_turns": sum(int(item["constraint_turns"]) for item in route_folds),
        "fallback_turns": sum(int(item["fallback_turns"]) for item in route_folds),
        "total_sessions": sum(int(item["total_sessions"]) for item in route_folds),
        "fallback_sessions": sum(
            int(item["fallback_sessions"]) for item in route_folds
        ),
        "fallback_sample_ids": sorted(
            sample_id
            for item in route_folds
            for sample_id in cast(list[str], item["fallback_sample_ids"])
        ),
    }
    for field in (
        "fallback_turns_by_reason",
        "fallback_sessions_by_reason",
        "fallback_sessions_by_scenario",
        "fallback_turns_by_scenario",
    ):
        values: Counter[str] = Counter()
        for item in route_folds:
            values.update(cast(dict[str, int], item[field]))
        aggregate_route[field] = dict(sorted(values.items()))

    runtime_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.measure_constraint_guard_runtime",
            str(BASE_MODEL_PATH),
            str(CONSTRAINT_MODEL_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(runtime_process.stdout.strip().splitlines()[-1])
    gates = amendment["promotion_gates"]
    score_delta = numeric(guarded_vs_base["mean_paired_session_reward_delta"])
    hit_delta = numeric(guarded_metrics["hit_rate_at_10"]) - numeric(
        base_metrics["hit_rate_at_10"]
    )
    gate_results = {
        "exact_base_reproduction": base_reproduced,
        "exact_unguarded_reproduction": unguarded_reproduced,
        "minimum_oof_score_delta_vs_base": score_delta
        >= numeric(gates["minimum_oof_score_delta_vs_base"]),
        "minimum_nonnegative_outer_folds_vs_base": sum(
            delta >= 0.0 for delta in fold_vs_base
        )
        >= int(gates["minimum_nonnegative_outer_folds_vs_base"]),
        "minimum_hit_rate_delta_vs_base": hit_delta
        >= numeric(gates["minimum_hit_rate_delta_vs_base"]),
        "minimum_each_scenario_score_delta_vs_base": min(scenario_vs_base.values())
        >= numeric(gates["minimum_each_scenario_score_delta_vs_base"]),
        "minimum_incremental_intent_override_delta_vs_unguarded": scenario_vs_unguarded[
            "intent_override"
        ]
        >= numeric(gates["minimum_incremental_intent_override_delta_vs_unguarded"]),
        "deterministic_guard": deterministic,
        "zero_failures": int(runtime["failure_count"])
        == int(gates["required_failure_count"]),
        "cold_start_budget": float(runtime["cold_start_seconds"])
        <= float(gates["cold_start_seconds_max"]),
        "warm_turn_budget": float(runtime["warm_turn_p95_ms"])
        <= float(gates["warm_turn_p95_ms_max"]),
        "memory_budget": float(runtime["peak_process_memory_mb"])
        <= float(gates["peak_process_memory_mb_max"]),
        "asset_budget": float(runtime["model_asset_mb"])
        <= float(gates["model_asset_mb_max"]),
    }
    decision = "PROMOTE" if all(gate_results.values()) else "STOP_CONSTRAINT_SEARCH"
    report = {
        "schema_version": 1,
        "experiment_id": amendment["amendment_id"],
        "amendment_path": str(AMENDMENT_PATH.relative_to(ROOT)),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "v2_report_path": str(V2_REPORT_PATH.relative_to(ROOT)),
        "v2_report_sha256": sha256_file(V2_REPORT_PATH),
        "split": "nested_v1",
        "split_sha256": sha256_file(nested_path),
        "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": sha256_file(catalog_path),
        "holdout_accessed": False,
        "collection": collection,
        "base": {
            "oof_metrics": base_metrics,
            "oof_sessions": base_sessions,
            "folds": base_folds,
            "stability": stability(base_folds),
            "exactly_reproduced": base_reproduced,
        },
        "unguarded_v2": {
            "oof_metrics": unguarded_metrics,
            "oof_sessions": unguarded_sessions,
            "folds": unguarded_folds,
            "stability": stability(unguarded_folds),
            "exactly_reproduced": unguarded_reproduced,
        },
        "guarded_v2": {
            "oof_metrics": guarded_metrics,
            "oof_sessions": guarded_sessions,
            "folds": guarded_folds,
            "stability": stability(guarded_folds),
            "paired_vs_base": guarded_vs_base,
            "paired_vs_unguarded": guarded_vs_unguarded,
            "fold_score_deltas_vs_base": fold_vs_base,
            "fold_score_deltas_vs_unguarded": fold_vs_unguarded,
            "scenario_reward_deltas_vs_base": scenario_vs_base,
            "scenario_reward_deltas_vs_unguarded": scenario_vs_unguarded,
        },
        "routing": {"aggregate": aggregate_route, "folds": route_folds},
        "determinism": {
            "same_fitted_models_replayed_twice": True,
            "sessions_and_routing_exact": deterministic,
        },
        "runtime": runtime,
        "model_assets": {
            "base_path": str(BASE_MODEL_PATH.relative_to(ROOT)),
            "base_sha256": sha256_file(BASE_MODEL_PATH),
            "constraint_path": str(CONSTRAINT_MODEL_PATH.relative_to(ROOT)),
            "constraint_sha256": sha256_file(CONSTRAINT_MODEL_PATH),
        },
        "promotion": {
            "decision": decision,
            "gate_results": gate_results,
            "all_gates_passed": all(gate_results.values()),
            "stop_rule_applied": decision == "STOP_CONSTRAINT_SEARCH",
        },
        "code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "ghostlab/state/memory.py",
                ROOT / "ghostlab/retrieval/constraint_gbdt.py",
                ROOT / "scripts/run_gbdt_constraint_override_guard.py",
                ROOT / "scripts/measure_constraint_guard_runtime.py",
            )
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "base": base_metrics,
                "unguarded": unguarded_metrics,
                "guarded": guarded_metrics,
                "paired_vs_base": guarded_vs_base,
                "paired_vs_unguarded": guarded_vs_unguarded,
                "fold_deltas_vs_base": fold_vs_base,
                "fold_deltas_vs_unguarded": fold_vs_unguarded,
                "scenario_deltas_vs_base": scenario_vs_base,
                "scenario_deltas_vs_unguarded": scenario_vs_unguarded,
                "routing": aggregate_route,
                "promotion": report["promotion"],
                "runtime": runtime,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
