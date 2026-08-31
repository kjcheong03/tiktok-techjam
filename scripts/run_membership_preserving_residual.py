from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from ghostlab.retrieval.constraint_gbdt import (
    CONSTRAINT_METADATA_FEATURES,
    ConstraintAgentAdapter,
    ConstraintContext,
    ConstraintGBDTFeatureStore,
    RuntimeConstraintReranker,
)
from ghostlab.retrieval.gbdt import LambdaMARTModel
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.residual import (
    FEATURE_SETS,
    RESIDUAL_FEATURES,
    ResidualPolicy,
    membership_preserving_reorder,
)
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.state.memory import ConversationState
from scripts.run_gbdt_constraint_interaction import (
    collect_groups,
    train_model,
)
from scripts.run_gbdt_constraint_override_guard import build_guarded_agent
from scripts.run_gbdt_reranker import (
    FIELD_WEIGHTS,
    paired_evidence,
    sha256_file,
    summarized_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/membership_preserving_residual_v1.json"
REPORT_PATH = ROOT / "artifacts/reports/membership_preserving_residual_v1.json"
PARENT_REPORT_PATH = ROOT / "artifacts/reports/gbdt_constraint_override_guard_v1.json"
SEED = 20260826
EXPECTED_PARENT_SCORE = 0.878963


class ProbabilityModel(Protocol):
    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class TraceTurn:
    sample_id: str
    turn: int
    eligible: bool
    target: str
    ranking: tuple[str, ...]
    features: NDArray[np.float64]
    labels: NDArray[np.int64]


@dataclass(frozen=True)
class TraceSession:
    sample_id: str
    scenario_type: str
    turns: tuple[TraceTurn, ...]
    parent_outcome: dict[str, object]


@dataclass(frozen=True)
class ModelSpec:
    family: str
    feature_set: str
    strength: float
    depth: int | None = None
    learning_rate: float | None = None

    @property
    def candidate_id(self) -> str:
        pieces = [self.family, self.feature_set, f"strength={self.strength:g}"]
        if self.depth is not None:
            pieces.append(f"depth={self.depth}")
        if self.learning_rate is not None:
            pieces.append(f"lr={self.learning_rate:g}")
        return "|".join(pieces)


@dataclass(frozen=True)
class SelectedConfiguration:
    model: ModelSpec
    policy: ResidualPolicy


def _technical_metrics(sessions: list[dict[str, object]]) -> dict[str, object]:
    return cast(dict[str, object], summarized_metrics(cast(list[dict], sessions)))


def _scenario_scores(sessions: list[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        name: float(_technical_metrics(values)["recommended_technical_score"])
        for name, values in sorted(grouped.items())
    }


def _feature_indices(feature_set: str) -> list[int]:
    return [RESIDUAL_FEATURES.index(name) for name in FEATURE_SETS[feature_set]]


def _model_specs() -> list[ModelSpec]:
    logistic = [
        ModelSpec("regularized_logistic", feature_set, strength)
        for feature_set in FEATURE_SETS
        for strength in (0.05, 0.2, 1.0, 5.0)
    ]
    trees = [
        ModelSpec(
            "shallow_histogram_gbdt",
            feature_set,
            strength,
            depth=depth,
            learning_rate=learning_rate,
        )
        for feature_set in ("metadata", "full_context")
        for strength in (1.0, 5.0)
        for depth, learning_rate in ((2, 0.05), (3, 0.05), (3, 0.1))
    ]
    return [*logistic, *trees]


def _policies() -> list[ResidualPolicy]:
    return [
        ResidualPolicy(
            rerank_depth=depth,
            model_weight=weight,
            minimum_expected_gain=gain,
            minimum_probability_margin=margin,
            maximum_moved_ids=moved,
        )
        for depth in (3, 5, 10)
        for weight in (0.5, 0.75, 1.0)
        for gain in (0.0, 0.01, 0.025, 0.05)
        for margin in (0.0, 0.02)
        for moved in (2, 4, 10)
    ]


def _fit_model(
    spec: ModelSpec,
    sessions: list[TraceSession],
) -> ProbabilityModel:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    indices = _feature_indices(spec.feature_set)
    features = np.vstack(
        [turn.features[:, indices] for session in sessions for turn in session.turns]
    )
    labels = np.concatenate(
        [turn.labels for session in sessions for turn in session.turns]
    )
    if len(np.unique(labels)) != 2:
        raise ValueError("residual training requires both positive and negative rows")
    if spec.family == "regularized_logistic":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                C=spec.strength,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED,
            ),
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(
                learning_rate=float(spec.learning_rate),
                max_depth=spec.depth,
                max_iter=100,
                min_samples_leaf=20,
                l2_regularization=spec.strength,
                random_state=SEED,
            ),
        )
    model.fit(features, labels)
    return cast(ProbabilityModel, model)


def _predict(
    model: ProbabilityModel,
    spec: ModelSpec,
    sessions: list[TraceSession],
) -> dict[tuple[str, int], NDArray[np.float64]]:
    indices = _feature_indices(spec.feature_set)
    result: dict[tuple[str, int], NDArray[np.float64]] = {}
    for session in sessions:
        for turn in session.turns:
            probabilities = model.predict_proba(turn.features[:, indices])[:, 1]
            result[(session.sample_id, turn.turn)] = np.asarray(
                probabilities, dtype=np.float64
            )
    return result


def _parent_scores(
    matrix: NDArray[np.float64],
    route: str,
    base_model: LambdaMARTModel,
    constraint_model: LambdaMARTModel,
) -> NDArray[np.float64]:
    model = base_model if route == "base_override_fallback" else constraint_model
    indices = [CONSTRAINT_METADATA_FEATURES.index(name) for name in model.feature_names]
    return model.predict(matrix[:, indices])


def _trace_features(
    agent: ConstraintAgentAdapter,
    runtime: RuntimeConstraintReranker,
    sparse: SparseIndex,
    feature_store: ConstraintGBDTFeatureStore,
    base_model: LambdaMARTModel,
    constraint_model: LambdaMARTModel,
    session_id: str,
    turn: int,
    ranking: tuple[str, ...],
) -> NDArray[np.float64]:
    state = agent.wrapped.sessions[session_id]
    if not isinstance(state, ConversationState):
        raise TypeError("guarded parent must expose ConversationState")
    query = ". ".join(state.messages)
    retrieval = sparse.search(query, 200, FIELD_WEIGHTS)
    raw_scores = [
        float(item.raw_score) for item in retrieval.items if item.raw_score is not None
    ]
    context = ConstraintContext.from_runtime(
        state, turn=turn, retrieval_scores=raw_scores
    )
    matrix = feature_store.contextual_matrix(
        query, ranking, context, CONSTRAINT_METADATA_FEATURES
    )
    route = str(runtime.routing_trace[-1]["route"])
    scores = _parent_scores(matrix, route, base_model, constraint_model)
    top = float(np.max(scores)) if len(scores) else 0.0
    derived = np.column_stack(
        (
            scores,
            scores - top,
            np.full(len(scores), float(route == "base_override_fallback")),
        )
    )
    return np.hstack((matrix, derived))


def _collect_parent_traces(
    agent: ConstraintAgentAdapter,
    runtime: RuntimeConstraintReranker,
    base_model: LambdaMARTModel,
    constraint_model: LambdaMARTModel,
    ordered_samples: list[dict[str, object]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    feature_store: ConstraintGBDTFeatureStore,
) -> list[TraceSession]:
    sessions: list[TraceSession] = []
    for sample in ordered_samples:
        sample_id = str(sample["sample_id"])
        session_id = f"residual:{sample_id}"
        agent.reset(session_id, cast(dict, sample["user_profile"]))
        target = str(cast(dict, sample["ground_truth"])["parent_asin"])
        intent, behavior = materialize_hidden_fields(cast(dict, sample), products)
        effective = {**sample, "intent_card": intent, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            cast(dict, effective),
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        turns: list[TraceTurn] = []
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, 11):
            response = agent.respond(session_id, message, turn, 10)
            ranking = tuple(
                normalize_recommendations(response.get("recommendations"), catalog_ids)
            )
            features = _trace_features(
                agent,
                runtime,
                sparse,
                feature_store,
                base_model,
                constraint_model,
                session_id,
                turn,
                ranking,
            )
            labels = np.asarray(
                [
                    int(override_applied and identifier == target)
                    for identifier in ranking
                ],
                dtype=np.int64,
            )
            turns.append(
                TraceTurn(
                    sample_id,
                    turn,
                    override_applied,
                    target,
                    ranking,
                    features,
                    labels,
                )
            )
            if override_applied and target in ranking:
                hit_turn = turn
                best_rank = ranking.index(target) + 1
                break
            if turn == 10:
                break
            override = cast(dict, effective).get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
            else:
                message, boundary_used = customer_reply(
                    cast(dict, effective),
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        parent = {
            "sample_id": sample_id,
            "scenario_type": str(sample["scenario_type"]),
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
        sessions.append(
            TraceSession(sample_id, str(sample["scenario_type"]), tuple(turns), parent)
        )
    return sessions


def _candidate_outcomes(
    sessions: list[TraceSession],
    predictions: dict[tuple[str, int], NDArray[np.float64]],
    policy: ResidualPolicy,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    outcomes: list[dict[str, object]] = []
    activations = membership_failures = moved_ids = 0
    for session in sessions:
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in session.turns:
            decision = membership_preserving_reorder(
                turn.ranking,
                predictions[(turn.sample_id, turn.turn)],
                policy,
            )
            if len(decision.ranking) != len(turn.ranking) or set(
                decision.ranking
            ) != set(turn.ranking):
                membership_failures += 1
            activations += int(decision.activated)
            moved_ids += decision.moved_ids if decision.activated else 0
            if turn.eligible and turn.target in decision.ranking:
                hit_turn = turn.turn
                best_rank = decision.ranking.index(turn.target) + 1
                break
        outcomes.append(
            {
                "sample_id": session.sample_id,
                "scenario_type": session.scenario_type,
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )
    return outcomes, {
        "activations": activations,
        "moved_ids": moved_ids,
        "membership_failures": membership_failures,
    }


def _oracle_outcomes(sessions: list[TraceSession]) -> list[dict[str, object]]:
    result = []
    for session in sessions:
        parent = session.parent_outcome
        result.append(
            {
                **parent,
                "best_rank": 1 if parent["hit"] else None,
                "reciprocal_rank": 1.0 if parent["hit"] else 0.0,
            }
        )
    return result


def _configuration_evidence(
    sessions: list[TraceSession],
    predictions: dict[tuple[str, int], NDArray[np.float64]],
    policy: ResidualPolicy,
    validation_partitions: list[set[str]],
) -> dict[str, object]:
    control = [session.parent_outcome for session in sessions]
    candidate, behavior = _candidate_outcomes(sessions, predictions, policy)
    control_metrics = _technical_metrics(control)
    candidate_metrics = _technical_metrics(candidate)
    scenario_control = _scenario_scores(control)
    scenario_candidate = _scenario_scores(candidate)
    scenario_deltas = {
        name: scenario_candidate[name] - scenario_control[name]
        for name in scenario_control
    }
    fold_deltas = []
    for partition in validation_partitions:
        left = [row for row in candidate if str(row["sample_id"]) in partition]
        right = [row for row in control if str(row["sample_id"]) in partition]
        fold_deltas.append(
            float(_technical_metrics(left)["recommended_technical_score"])
            - float(_technical_metrics(right)["recommended_technical_score"])
        )
    score_delta = float(candidate_metrics["recommended_technical_score"]) - float(
        control_metrics["recommended_technical_score"]
    )
    worst_scenario = min(scenario_deltas.values())
    stable_folds = sum(delta >= 0.0 for delta in fold_deltas)
    utility = score_delta - 2.0 * max(0.0, -0.005 - worst_scenario)
    utility -= 0.001 * max(0, len(fold_deltas) - stable_folds)
    return {
        "utility": utility,
        "score_delta": score_delta,
        "worst_scenario_delta": worst_scenario,
        "nonnegative_folds": stable_folds,
        "fold_deltas": fold_deltas,
        "scenario_deltas": scenario_deltas,
        "candidate_metrics": candidate_metrics,
        "control_metrics": control_metrics,
        "behavior": behavior,
    }


def _cross_fitted_predictions(
    spec: ModelSpec,
    sessions: list[TraceSession],
    partitions: list[set[str]],
) -> dict[tuple[str, int], NDArray[np.float64]]:
    predictions: dict[tuple[str, int], NDArray[np.float64]] = {}
    for validation_ids in partitions:
        training = [s for s in sessions if s.sample_id not in validation_ids]
        validation = [s for s in sessions if s.sample_id in validation_ids]
        model = _fit_model(spec, training)
        overlap = predictions.keys() & _predict(model, spec, validation).keys()
        if overlap:
            raise RuntimeError("cross-fitted predictions overlap")
        predictions.update(_predict(model, spec, validation))
    expected = {
        (session.sample_id, turn.turn) for session in sessions for turn in session.turns
    }
    if predictions.keys() != expected:
        raise RuntimeError("cross-fitted predictions are incomplete")
    return predictions


def _select_configuration(
    sessions: list[TraceSession],
    partitions: list[set[str]],
    shortlist_size: int,
) -> tuple[SelectedConfiguration, dict[str, object]]:
    default = ResidualPolicy()
    predictions_by_model: dict[str, dict[tuple[str, int], NDArray[np.float64]]] = {}
    model_screen = []
    for spec in _model_specs():
        predictions = _cross_fitted_predictions(spec, sessions, partitions)
        predictions_by_model[spec.candidate_id] = predictions
        evidence = _configuration_evidence(sessions, predictions, default, partitions)
        model_screen.append((spec, evidence))
    model_screen.sort(
        key=lambda item: (
            float(item[1]["utility"]),
            float(item[1]["score_delta"]),
            item[0].candidate_id,
        ),
        reverse=True,
    )
    shortlisted = model_screen[:shortlist_size]
    evaluated: list[tuple[ModelSpec, ResidualPolicy, dict[str, object]]] = []
    for spec, _ in shortlisted:
        predictions = predictions_by_model[spec.candidate_id]
        for policy in _policies():
            evidence = _configuration_evidence(
                sessions, predictions, policy, partitions
            )
            evaluated.append((spec, policy, evidence))
    evaluated.sort(
        key=lambda item: (
            float(item[2]["utility"]),
            float(item[2]["score_delta"]),
            float(item[2]["worst_scenario_delta"]),
            -item[1].rerank_depth,
            item[0].candidate_id,
        ),
        reverse=True,
    )
    winner = evaluated[0]
    return SelectedConfiguration(winner[0], winner[1]), {
        "model_candidates": len(model_screen),
        "shortlisted_models": [item[0].candidate_id for item in shortlisted],
        "policy_candidates_per_model": len(_policies()),
        "evaluated_configurations": len(evaluated),
        "selected_inner_evidence": winner[2],
        "top_configurations": [
            {
                "model": item[0].candidate_id,
                "policy": asdict(item[1]),
                "score_delta": item[2]["score_delta"],
                "worst_scenario_delta": item[2]["worst_scenario_delta"],
                "nonnegative_folds": item[2]["nonnegative_folds"],
            }
            for item in evaluated[:10]
        ],
    }


def _id_digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _trace_parent(
    groups: dict[str, Any],
    samples: dict[str, dict[str, object]],
    validation_ids: set[str],
    training_ids: set[str],
    rounds: int,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    sparse: SparseIndex,
    quality: CatalogQualityReranker,
    feature_store: ConstraintGBDTFeatureStore,
) -> tuple[list[TraceSession], dict[str, object]]:
    if training_ids & validation_ids:
        raise RuntimeError("parent fit receipt is not disjoint")
    base_model = train_model(groups, training_ids, candidate=False, rounds=rounds)
    constraint_model = train_model(groups, training_ids, candidate=True, rounds=rounds)
    agent, runtime = build_guarded_agent(
        quality, feature_store, base_model, constraint_model
    )
    traces = _collect_parent_traces(
        agent,
        runtime,
        base_model,
        constraint_model,
        [samples[sample_id] for sample_id in sorted(validation_ids)],
        catalog_ids,
        categories,
        products,
        sparse,
        feature_store,
    )
    receipt = {
        "training_count": len(training_ids),
        "validation_count": len(validation_ids),
        "training_ids_sha256": _id_digest(training_ids),
        "validation_ids_sha256": _id_digest(validation_ids),
        "disjoint": not bool(training_ids & validation_ids),
        "rounds": rounds,
    }
    return traces, receipt


def _summary_markdown(report: dict[str, object]) -> str:
    control = cast(dict, report["control"])["oof_metrics"]
    candidate = cast(dict, report["candidate"])["oof_metrics"]
    paired = cast(dict, cast(dict, report["candidate"])["paired_vs_control"])
    decision = cast(dict, report["decision"])
    return f"""# Membership-Preserving Residual Reranker Decision

## Decision

**{decision["status"]}** — {decision["reason"]}

This was evaluated independently and has not been registered in the autonomous
engine. The protected holdout was not accessed.

## Nested out-of-fold result

| Variant | Hit@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Guarded GBDT parent | {control["hit_rate_at_10"]:.6f} | {control["mrr"]:.6f} | {control["mttc"]:.6f} | {control["recommended_technical_score"]:.6f} |
| Residual reranker | {candidate["hit_rate_at_10"]:.6f} | {candidate["mrr"]:.6f} | {candidate["mttc"]:.6f} | {candidate["recommended_technical_score"]:.6f} |

Technical-score delta: `{decision["score_delta"]:+.6f}`.

Paired evidence: `{paired["wins"]}` wins, `{paired["losses"]}` losses and
`{paired["ties"]}` ties; bootstrap 95% interval
`[{paired["paired_bootstrap_95_interval"][0]:+.6f}, {paired["paired_bootstrap_95_interval"][1]:+.6f}]`;
paired randomization `p={paired["paired_randomization_p_value"]:.6f}`.

## Safety result

- Exact Top-10 membership failures: `{decision["membership_failures"]}`.
- Hit@10 absolute difference: `{decision["hit_rate_absolute_difference"]:.6f}`.
- MTTC absolute difference: `{decision["mttc_absolute_difference"]:.6f}`.
- Nonnegative outer folds: `{decision["nonnegative_outer_folds"]}/5`.
- Worst scenario delta: `{decision["worst_scenario_delta"]:+.6f}`.
- Parent score reproduced: `{report["parent_reproduced"]}`.

## Interpretation

The search adaptively selected model family, observable feature set, regularization,
rerank depth, champion/model blend, expected-gain threshold, probability-margin
threshold, and movement limit inside each outer fold. Every reported candidate
session remained unseen by both its parent GBDT and residual learner.

The aggregate signal is promising, but the automatic selector was unstable: the
regularized logistic family produced the strongest held-out folds, while two
selected shallow-tree variants regressed. Preserve this implementation for a
future pre-registered conservative-selector and interaction experiment; do not
enable it in the champion or autonomous promotion pool from this result alone.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold-limit", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest["holdout_accessed"] is not False
        or manifest["outcomes_observed"] is not False
    ):
        raise RuntimeError("experiment manifest was not frozen before outcomes")
    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    all_outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    outer_folds = all_outer_folds[: args.outer_fold_limit]
    parent_report = json.loads(PARENT_REPORT_PATH.read_text(encoding="utf-8"))
    rounds = [
        int(fold["frozen_rounds"]) for fold in parent_report["guarded_v2"]["folds"]
    ]
    samples = {
        str(sample["sample_id"]): cast(dict[str, object], sample)
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    sparse = SparseIndex(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    feature_store = ConstraintGBDTFeatureStore(catalog_path, quality=quality.quality)
    groups, collection = collect_groups(
        samples, categories, products, sparse, quality, feature_store
    )
    print(json.dumps({"stage": "collection", **collection}, sort_keys=True), flush=True)

    control_sessions: list[dict[str, object]] = []
    candidate_sessions: list[dict[str, object]] = []
    oracle_sessions: list[dict[str, object]] = []
    folds = []
    parent_receipts = []
    residual_receipts = []
    membership_failures = activations = moved_ids = 0
    for outer_index, outer_ids in enumerate(outer_folds):
        outer_training_ids = adaptive_ids - outer_ids
        inner_partitions = [
            fold & outer_training_ids for fold in all_outer_folds if fold != outer_ids
        ]
        inner_traces: list[TraceSession] = []
        inner_receipts = []
        for inner_index, inner_ids in enumerate(inner_partitions):
            source_index = all_outer_folds.index(
                next(
                    f
                    for f in all_outer_folds
                    if f == inner_ids or f & outer_training_ids == inner_ids
                )
            )
            traces, receipt = _trace_parent(
                groups,
                samples,
                inner_ids,
                outer_training_ids - inner_ids,
                rounds[source_index],
                catalog_ids,
                categories,
                products,
                sparse,
                quality,
                feature_store,
            )
            inner_traces.extend(traces)
            inner_receipts.append({"inner_fold": inner_index, **receipt})
        selected, search = _select_configuration(
            inner_traces,
            inner_partitions,
            int(manifest["adaptive_search"]["model_shortlist"]),
        )
        residual_model = _fit_model(selected.model, inner_traces)
        outer_traces, parent_receipt = _trace_parent(
            groups,
            samples,
            outer_ids,
            outer_training_ids,
            rounds[outer_index],
            catalog_ids,
            categories,
            products,
            sparse,
            quality,
            feature_store,
        )
        predictions = _predict(residual_model, selected.model, outer_traces)
        candidate, behavior = _candidate_outcomes(
            outer_traces, predictions, selected.policy
        )
        control = [trace.parent_outcome for trace in outer_traces]
        oracle = _oracle_outcomes(outer_traces)
        control_metrics = _technical_metrics(control)
        candidate_metrics = _technical_metrics(candidate)
        control_sessions.extend(control)
        candidate_sessions.extend(candidate)
        oracle_sessions.extend(oracle)
        membership_failures += behavior["membership_failures"]
        activations += behavior["activations"]
        moved_ids += behavior["moved_ids"]
        parent_receipts.append(
            {"outer_fold": outer_index, **parent_receipt, "inner": inner_receipts}
        )
        residual_receipts.append(
            {
                "outer_fold": outer_index,
                "training_count": len(outer_training_ids),
                "validation_count": len(outer_ids),
                "training_ids_sha256": _id_digest(outer_training_ids),
                "validation_ids_sha256": _id_digest(outer_ids),
                "disjoint": not bool(outer_training_ids & outer_ids),
            }
        )
        folds.append(
            {
                "outer_fold": outer_index,
                "selected_model": asdict(selected.model),
                "selected_policy": asdict(selected.policy),
                "search": search,
                "control_metrics": control_metrics,
                "candidate_metrics": candidate_metrics,
                "score_delta": float(candidate_metrics["recommended_technical_score"])
                - float(control_metrics["recommended_technical_score"]),
                "behavior": behavior,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_fold",
                    "fold": outer_index,
                    "control": control_metrics["recommended_technical_score"],
                    "candidate": candidate_metrics["recommended_technical_score"],
                    "model": selected.model.candidate_id,
                    "policy": asdict(selected.policy),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    control_metrics = _technical_metrics(control_sessions)
    candidate_metrics = _technical_metrics(candidate_sessions)
    oracle_metrics = _technical_metrics(oracle_sessions)
    score_delta = float(candidate_metrics["recommended_technical_score"]) - float(
        control_metrics["recommended_technical_score"]
    )
    hit_difference = abs(
        float(candidate_metrics["hit_rate_at_10"])
        - float(control_metrics["hit_rate_at_10"])
    )
    mttc_difference = abs(
        float(candidate_metrics["mttc"]) - float(control_metrics["mttc"])
    )
    scenario_control = _scenario_scores(control_sessions)
    scenario_candidate = _scenario_scores(candidate_sessions)
    scenario_deltas = {
        name: scenario_candidate[name] - scenario_control[name]
        for name in scenario_control
    }
    fold_deltas = [float(fold["score_delta"]) for fold in folds]
    nonnegative = sum(delta >= 0.0 for delta in fold_deltas)
    parent_reproduced = (
        len(outer_folds) < 5
        or abs(
            float(control_metrics["recommended_technical_score"])
            - EXPECTED_PARENT_SCORE
        )
        <= 1e-6
    )
    gates = manifest["decision_gates"]
    hard_safe = (
        membership_failures <= int(gates["maximum_membership_failures"])
        and hit_difference <= float(gates["maximum_hit_rate_absolute_difference"])
        and mttc_difference <= float(gates["maximum_mttc_absolute_difference"])
        and all(receipt["disjoint"] for receipt in residual_receipts)
        and all(receipt["disjoint"] for receipt in parent_receipts)
        and parent_reproduced
    )
    promote = (
        hard_safe
        and score_delta >= float(gates["minimum_oof_score_delta"])
        and nonnegative >= int(gates["minimum_nonnegative_outer_folds"])
        and min(scenario_deltas.values()) >= float(gates["minimum_scenario_delta"])
    )
    if promote:
        status = "ADD_AS_RECEIPT_GATED_EXPERIMENTAL"
        reason = "Nested OOF evidence passed every safety and performance gate."
    elif hard_safe:
        status = "PARK_RETESTABLE"
        reason = "The technique is mechanically safe, but evidence did not pass every promotion gate."
    else:
        status = "REJECT_CURRENT_IMPLEMENTATION"
        reason = (
            "A hard invariant, leakage receipt, or parent-reproduction gate failed."
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "holdout_accessed": False,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
        "split_sha256": sha256_file(nested_path),
        "catalog_sha256": sha256_file(catalog_path.resolve()),
        "collection": collection,
        "outer_folds_completed": len(folds),
        "adaptive_search_space": {
            "model_candidates": len(_model_specs()),
            "policies_per_shortlisted_model": len(_policies()),
            "shortlisted_models_per_outer_fold": int(
                manifest["adaptive_search"]["model_shortlist"]
            ),
        },
        "control": {
            "oof_metrics": control_metrics,
            "scenario_scores": scenario_control,
        },
        "candidate": {
            "oof_metrics": candidate_metrics,
            "scenario_scores": scenario_candidate,
            "paired_vs_control": paired_evidence(
                cast(list[dict], candidate_sessions), cast(list[dict], control_sessions)
            ),
            "activations": activations,
            "moved_ids": moved_ids,
        },
        "oracle_membership_ceiling": oracle_metrics,
        "folds": folds,
        "fit_receipts": {"parent": parent_receipts, "residual": residual_receipts},
        "parent_reproduced": parent_reproduced,
        "decision": {
            "status": status,
            "reason": reason,
            "score_delta": score_delta,
            "fold_deltas": fold_deltas,
            "nonnegative_outer_folds": nonnegative,
            "scenario_deltas": scenario_deltas,
            "worst_scenario_delta": min(scenario_deltas.values()),
            "membership_failures": membership_failures,
            "hit_rate_absolute_difference": hit_difference,
            "mttc_absolute_difference": mttc_difference,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"report": str(REPORT_PATH.relative_to(ROOT)), **report["decision"]},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
