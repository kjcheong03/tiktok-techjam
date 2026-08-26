from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.research.replay import ReplayEnvironment, session_reward
from ghostlab.retrieval.constraint_gbdt import (
    CONSTRAINT_METADATA_FEATURES,
    ConstraintAgentAdapter,
    ConstraintContext,
    ConstraintGBDTFeatureStore,
    RuntimeConstraintReranker,
)
from ghostlab.retrieval.gbdt import (
    METADATA_FEATURES,
    LambdaMARTModel,
    LambdaMARTReranker,
    fit_lambdamart,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState
from scripts.run_gbdt_reranker import (
    FIELD_WEIGHTS,
    QUESTION_ORDER,
    SEED,
    paired_evidence,
    sha256_file,
    stability,
    summarized_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/gbdt_constraint_interaction_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/gbdt_constraint_interaction_v1.json"
MODEL_PATH = ROOT / "artifacts/models/gbdt_constraint_interaction_v1.json"


@dataclass(frozen=True)
class ContextRankingGroup:
    sample_id: str
    turn: int
    query: str
    candidates: tuple[str, ...]
    labels: tuple[int, ...]
    base_matrix: NDArray[np.float64]
    candidate_matrix: NDArray[np.float64]


def collect_groups(
    samples: dict[str, dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    quality: CatalogQualityReranker,
    features: ConstraintGBDTFeatureStore,
) -> tuple[dict[str, list[ContextRankingGroup]], dict[str, int]]:
    by_sample: dict[str, list[ContextRankingGroup]] = defaultdict(list)
    trajectory_queries = positive_queries = candidate_rows = 0
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        state = ConversationState(
            environment.session_id,
            sample["user_profile"],
            multi_value=False,
            negative_evidence=True,
            provenance_enabled=True,
            override_invalidation=True,
        )
        target = str(sample["ground_truth"]["parent_asin"])
        while not environment.done:
            state.observe(observation.user_message, observation.turn)
            turn = observation.turn
            question = QUESTION_ORDER[turn - 1] if turn <= len(QUESTION_ORDER) else None
            if question is not None:
                state.asked_attributes.append(question)
            state.last_asked_attribute = question
            query = ". ".join(state.messages)
            scored = sparse.search(query, 200, FIELD_WEIGHTS)
            raw_scores = [
                float(item.raw_score)
                for item in scored.items
                if item.raw_score is not None
            ]
            ranking = quality.rerank(
                [item.parent_asin for item in scored.items],
                weight=0.2,
                rerank_k=50,
            )
            head = tuple(ranking[:50])
            trajectory_queries += 1
            if target in head:
                context = ConstraintContext.from_runtime(
                    state, turn=turn, retrieval_scores=raw_scores
                )
                labels = tuple(int(identifier == target) for identifier in head)
                by_sample[sample_id].append(
                    ContextRankingGroup(
                        sample_id=sample_id,
                        turn=turn,
                        query=query,
                        candidates=head,
                        labels=labels,
                        base_matrix=features.matrix(query, head, METADATA_FEATURES),
                        candidate_matrix=features.contextual_matrix(
                            query, head, context, CONSTRAINT_METADATA_FEATURES
                        ),
                    )
                )
                positive_queries += 1
                candidate_rows += len(head)
            next_observation = environment.step(
                {
                    "message": "training trajectory",
                    "ask_attribute": question,
                    "recommendations": [],
                }
            )
            if next_observation is not None:
                observation = next_observation
    return dict(by_sample), {
        "trajectory_queries": trajectory_queries,
        "queries_with_target_in_top50": positive_queries,
        "samples_with_ranking_groups": len(by_sample),
        "ranking_groups": sum(len(values) for values in by_sample.values()),
        "candidate_rows": candidate_rows,
    }


def ranking_dataset(
    groups: dict[str, list[ContextRankingGroup]],
    sample_ids: set[str],
    *,
    candidate: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[int]]:
    selected = [
        group for sample_id in sorted(sample_ids) for group in groups.get(sample_id, [])
    ]
    if not selected:
        raise ValueError("ranking dataset cannot be empty")
    return (
        np.vstack(
            [
                group.candidate_matrix if candidate else group.base_matrix
                for group in selected
            ]
        ),
        np.concatenate(
            [np.asarray(group.labels, dtype=np.int64) for group in selected]
        ),
        [len(group.labels) for group in selected],
    )


def train_model(
    groups: dict[str, list[ContextRankingGroup]],
    training_ids: set[str],
    *,
    candidate: bool,
    rounds: int,
) -> LambdaMARTModel:
    names = CONSTRAINT_METADATA_FEATURES if candidate else METADATA_FEATURES
    return fit_lambdamart(
        *ranking_dataset(groups, training_ids, candidate=candidate),
        candidate_id=(
            "metadata_depth3_plus_runtime_constraints"
            if candidate
            else "shallow_metadata_depth3_matched_control"
        ),
        feature_names=names,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        max_rounds=rounds,
        early_stopping_rounds=20,
        validation=None,
        seed=SEED,
    )


def build_agent(
    quality: CatalogQualityReranker,
    features: ConstraintGBDTFeatureStore,
    model: LambdaMARTModel,
    *,
    candidate: bool,
) -> ExperimentalAgent | ConstraintAgentAdapter:
    contextual = (
        RuntimeConstraintReranker(
            str(ROOT / "data/catalog.jsonl"), FIELD_WEIGHTS, features, model
        )
        if candidate
        else None
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
        learned_reranker=(
            contextual if candidate else LambdaMARTReranker(features, model)
        ),
    )
    return (
        ConstraintAgentAdapter(agent, contextual) if contextual is not None else agent
    )


def scenario_rewards(sessions: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session_reward(session))
    return {
        name: round(float(np.mean(rewards)), 6)
        for name, rewards in sorted(grouped.items())
    }


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["holdout_accessed"] is not False:
        raise RuntimeError("experiment manifest does not preserve holdout firewall")
    config = manifest["candidate"]
    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    rounds = [int(value) for value in config["outer_fold_rounds"]]
    if len(rounds) != len(outer_folds):
        raise RuntimeError("frozen round vector does not match outer folds")
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

    control_sessions: list[dict] = []
    candidate_sessions: list[dict] = []
    control_folds: list[dict] = []
    candidate_folds: list[dict] = []
    importance = {name: 0 for name in CONSTRAINT_METADATA_FEATURES}
    deterministic = True
    for fold_index, (outer_ids, fold_rounds) in enumerate(
        zip(outer_folds, rounds, strict=True)
    ):
        training_ids = adaptive_ids - outer_ids
        control_model = train_model(
            groups, training_ids, candidate=False, rounds=fold_rounds
        )
        candidate_model = train_model(
            groups, training_ids, candidate=True, rounds=fold_rounds
        )
        repeated_model = train_model(
            groups, training_ids, candidate=True, rounds=fold_rounds
        )
        deterministic = deterministic and candidate_model == repeated_model
        for name, count in candidate_model.split_importance().items():
            importance[name] += count
        control_result = evaluate(
            build_agent(quality, features, control_model, candidate=False),
            [samples[sample_id] for sample_id in sorted(outer_ids)],
            catalog_ids,
            categories,
            products,
        )
        candidate_result = evaluate(
            build_agent(quality, features, candidate_model, candidate=True),
            [samples[sample_id] for sample_id in sorted(outer_ids)],
            catalog_ids,
            categories,
            products,
        )
        control_sessions.extend(control_result["sessions"])
        candidate_sessions.extend(candidate_result["sessions"])
        shared = {
            "outer_fold": fold_index,
            "outer_training_ids": sorted(training_ids),
            "outer_validation_ids": sorted(outer_ids),
            "frozen_rounds": fold_rounds,
        }
        control_folds.append(
            {
                **shared,
                "outer_metrics": summarized_metrics(control_result["sessions"]),
                "scenario_rewards": scenario_rewards(control_result["sessions"]),
            }
        )
        candidate_folds.append(
            {
                **shared,
                "outer_metrics": summarized_metrics(candidate_result["sessions"]),
                "scenario_rewards": scenario_rewards(candidate_result["sessions"]),
            }
        )
        print(
            f"fold={fold_index} rounds={fold_rounds} "
            f"control={control_result['recommended_technical_score']} "
            f"candidate={candidate_result['recommended_technical_score']}",
            flush=True,
        )

    control_metrics = summarized_metrics(control_sessions)
    candidate_metrics = summarized_metrics(candidate_sessions)
    required_control = float(manifest["controls"]["required_control_score"])
    control_reproduced = abs(
        float(control_metrics["recommended_technical_score"]) - required_control
    ) <= float(manifest["promotion_gates"]["exact_control_reproduction_tolerance"])
    paired = paired_evidence(candidate_sessions, control_sessions)
    control_scenarios = scenario_rewards(control_sessions)
    candidate_scenarios = scenario_rewards(candidate_sessions)
    scenario_deltas = {
        name: round(candidate_scenarios[name] - control_scenarios[name], 6)
        for name in control_scenarios
    }
    fold_deltas = [
        round(
            float(candidate["outer_metrics"]["recommended_technical_score"])
            - float(control["outer_metrics"]["recommended_technical_score"]),
            6,
        )
        for candidate, control in zip(candidate_folds, control_folds, strict=True)
    ]

    deployable = train_model(
        groups,
        adaptive_ids,
        candidate=True,
        rounds=int(config["all_development_rounds"]),
    )
    repeated_deployable = train_model(
        groups,
        adaptive_ids,
        candidate=True,
        rounds=int(config["all_development_rounds"]),
    )
    deterministic = deterministic and deployable == repeated_deployable
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    deployable.save(MODEL_PATH)
    runtime_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.measure_constraint_gbdt_runtime",
            str(MODEL_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(runtime_process.stdout.strip().splitlines()[-1])
    gates = manifest["promotion_gates"]
    score_delta = float(paired["mean_paired_session_reward_delta"])
    hit_delta = float(candidate_metrics["hit_rate_at_10"]) - float(
        control_metrics["hit_rate_at_10"]
    )
    gate_results = {
        "exact_control_reproduction": control_reproduced,
        "minimum_oof_score_delta": score_delta
        >= float(gates["minimum_oof_score_delta"]),
        "minimum_nonnegative_outer_folds": sum(delta >= 0.0 for delta in fold_deltas)
        >= int(gates["minimum_nonnegative_outer_folds"]),
        "minimum_hit_rate_delta": hit_delta >= float(gates["minimum_hit_rate_delta"]),
        "minimum_each_scenario_score_delta": min(scenario_deltas.values())
        >= float(gates["minimum_each_scenario_score_delta"]),
        "deterministic_models": deterministic,
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
    decision = "PROMOTE" if all(gate_results.values()) else "PARK_INTERACTION"
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "split": "nested_v1",
        "split_sha256": sha256_file(nested_path),
        "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": sha256_file(catalog_path),
        "holdout_accessed": False,
        "seed": SEED,
        "collection": collection,
        "matched_control": {
            "evidence_label": "outer-fold/out-of-fold",
            "oof_metrics": control_metrics,
            "oof_sessions": control_sessions,
            "folds": control_folds,
            "stability": stability(control_folds),
            "required_score": required_control,
            "exactly_reproduced": control_reproduced,
        },
        "candidate": {
            "evidence_label": "outer-fold/out-of-fold",
            "oof_metrics": candidate_metrics,
            "oof_sessions": candidate_sessions,
            "folds": candidate_folds,
            "stability": stability(candidate_folds),
            "paired_vs_matched_control": paired,
            "fold_score_deltas": fold_deltas,
            "scenario_rewards": candidate_scenarios,
            "scenario_reward_deltas": scenario_deltas,
            "feature_importance_split_count_across_outer_models": dict(
                sorted(importance.items(), key=lambda item: (-item[1], item[0]))
            ),
        },
        "backward_ablation": {
            "description": "all runtime-constraint features removed; identical metadata model, folds, training groups, Top-50 head, and frozen rounds",
            "metrics": control_metrics,
            "paired_candidate_minus_ablation": paired,
        },
        "all_development_refit": {
            "evidence_label": "all-development refit",
            "rounds": int(config["all_development_rounds"]),
            "model_path": str(MODEL_PATH.relative_to(ROOT)),
            "model_sha256": sha256_file(MODEL_PATH),
            "training_groups": deployable.training_groups,
            "training_rows": deployable.training_rows,
            "feature_importance_split_count": dict(
                sorted(
                    deployable.split_importance().items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
        "runtime": runtime,
        "determinism": {
            "independent_refits_byte_equivalent": deterministic,
            "outer_models_repeated": len(outer_folds),
            "all_development_model_repeated": True,
        },
        "promotion": {
            "decision": decision,
            "gate_results": gate_results,
            "all_gates_passed": all(gate_results.values()),
        },
        "code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "ghostlab/retrieval/constraint_gbdt.py",
                ROOT / "ghostlab/runtime/experimental.py",
                ROOT / "scripts/run_gbdt_constraint_interaction.py",
                ROOT / "scripts/measure_constraint_gbdt_runtime.py",
            )
        },
        "failure_status": None,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "control": control_metrics,
                "candidate": candidate_metrics,
                "paired": paired,
                "fold_deltas": fold_deltas,
                "scenario_deltas": scenario_deltas,
                "promotion": report["promotion"],
                "runtime": runtime,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
