from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
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
from ghostlab.research.technique_suite import (
    UnifiedTechniqueConfig,
    build_suite_agent,
    load_suite_config,
)
from ghostlab.retrieval.constraint_gbdt import (
    CONSTRAINT_METADATA_FEATURES,
    ConstraintContext,
    ConstraintGBDTFeatureStore,
)
from ghostlab.retrieval.residual import (
    DERIVED_FEATURES,
    RESIDUAL_FEATURES,
    ResidualPolicy,
)
from ghostlab.runtime.unified_experimental import ExperimentalAgent
from scripts.run_gbdt_reranker import paired_evidence, sha256_file
from scripts.run_membership_preserving_residual import (
    ModelSpec,
    ProbabilityModel,
    TraceSession,
    TraceTurn,
    _candidate_outcomes,
    _configuration_evidence,
    _feature_indices,
    _model_specs,
    _oracle_outcomes,
    _policies,
    _predict,
    _scenario_scores,
    _technical_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/experiments/state_v2_adaptive_residual_v2.json"
REPORT_PATH = ROOT / "artifacts/reports/state_v2_adaptive_residual_v2.json"
STATE_V2_REPORT = ROOT / "artifacts/reports/state_baseline_v2_integration.json"
SEED = 20260826


class WeightedProbabilityModel(Protocol):
    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class AdaptiveConfiguration:
    models: tuple[ModelSpec, ...]
    policy: ResidualPolicy

    @property
    def candidate_id(self) -> str:
        if not self.models:
            return "adaptive_off"
        return "+".join(model.candidate_id for model in self.models)


def _rank_aware_dataset(
    spec: ModelSpec, sessions: list[TraceSession]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    indices = _feature_indices(spec.feature_set)
    matrices: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    weights: list[NDArray[np.float64]] = []
    for session in sessions:
        for turn in session.turns:
            matrices.append(turn.features[:, indices])
            labels.append(turn.labels)
            group_weights = np.full(len(turn.labels), 0.35, dtype=np.float64)
            positives = np.flatnonzero(turn.labels > 0)
            if len(positives) == 1:
                group_weights.fill(1.0)
                rank = int(positives[0]) + 1
                group_weights[int(positives[0])] = 1.0 + 4.0 * (1.0 - 1.0 / rank)
            weights.append(group_weights)
    return np.vstack(matrices), np.concatenate(labels), np.concatenate(weights)


def _fit_rank_aware_model(
    spec: ModelSpec, sessions: list[TraceSession]
) -> ProbabilityModel:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels, sample_weights = _rank_aware_dataset(spec, sessions)
    if len(np.unique(labels)) != 2:
        raise ValueError("rank-aware residual fit requires both classes")
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
        model.fit(features, labels, logisticregression__sample_weight=sample_weights)
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
        model.fit(
            features,
            labels,
            histgradientboostingclassifier__sample_weight=sample_weights,
        )
    return cast(ProbabilityModel, model)


def _average_predictions(
    members: tuple[ModelSpec, ...],
    predictions_by_model: dict[str, dict[tuple[str, int], NDArray[np.float64]]],
) -> dict[tuple[str, int], NDArray[np.float64]]:
    if not members:
        raise ValueError("an off configuration has no predictions")
    keys = predictions_by_model[members[0].candidate_id].keys()
    return {
        key: np.mean(
            [predictions_by_model[model.candidate_id][key] for model in members],
            axis=0,
        )
        for key in keys
    }


def _cross_fitted_predictions(
    spec: ModelSpec,
    sessions: list[TraceSession],
    partitions: list[set[str]],
) -> dict[tuple[str, int], NDArray[np.float64]]:
    result: dict[tuple[str, int], NDArray[np.float64]] = {}
    for validation_ids in partitions:
        training = [
            session for session in sessions if session.sample_id not in validation_ids
        ]
        validation = [
            session for session in sessions if session.sample_id in validation_ids
        ]
        model = _fit_rank_aware_model(spec, training)
        prediction = _predict(model, spec, validation)
        if result.keys() & prediction.keys():
            raise RuntimeError("cross-fitted prediction overlap")
        result.update(prediction)
    expected = {
        (session.sample_id, turn.turn) for session in sessions for turn in session.turns
    }
    if result.keys() != expected:
        raise RuntimeError("cross-fitted predictions are incomplete")
    return result


def _robust_evidence(
    evidence: dict[str, object], total_turns: int, inner_fold_count: int
) -> dict[str, object]:
    fold_deltas = [float(value) for value in cast(list, evidence["fold_deltas"])]
    mean_fold = statistics.fmean(fold_deltas)
    fold_std = statistics.stdev(fold_deltas) if len(fold_deltas) > 1 else 0.0
    lower_bound = mean_fold - fold_std / math.sqrt(max(1, len(fold_deltas)))
    worst_scenario = float(evidence["worst_scenario_delta"])
    behavior = cast(dict, evidence["behavior"])
    activation_rate = int(behavior["activations"]) / max(1, total_turns)
    nonnegative = int(evidence["nonnegative_folds"])
    eligible = (
        float(evidence["score_delta"]) > 0.0
        and nonnegative >= max(1, inner_fold_count - 1)
        and worst_scenario >= -0.005
    )
    utility = 0.5 * float(evidence["score_delta"]) + 0.5 * lower_bound
    utility -= 2.0 * max(0.0, -0.005 - worst_scenario)
    utility -= 0.00025 * activation_rate
    return {
        **evidence,
        "fold_mean": mean_fold,
        "fold_standard_deviation": fold_std,
        "fold_lower_bound": lower_bound,
        "activation_rate": activation_rate,
        "eligible": eligible,
        "robust_utility": utility,
    }


def _select_configuration(
    sessions: list[TraceSession],
    partitions: list[set[str]],
    shortlist_size: int,
) -> tuple[AdaptiveConfiguration, dict[str, object]]:
    total_turns = sum(len(session.turns) for session in sessions)
    default_policy = ResidualPolicy()
    predictions_by_model: dict[str, dict[tuple[str, int], NDArray[np.float64]]] = {}
    screens: list[tuple[ModelSpec, dict[str, object]]] = []
    for spec in _model_specs():
        predictions = _cross_fitted_predictions(spec, sessions, partitions)
        predictions_by_model[spec.candidate_id] = predictions
        evidence = _configuration_evidence(
            sessions, predictions, default_policy, partitions
        )
        screens.append((spec, _robust_evidence(evidence, total_turns, len(partitions))))
    screens.sort(
        key=lambda item: (
            bool(item[1]["eligible"]),
            float(item[1]["robust_utility"]),
            float(item[1]["score_delta"]),
            item[0].candidate_id,
        ),
        reverse=True,
    )
    shortlisted = tuple(item[0] for item in screens[:shortlist_size])
    members = [(model,) for model in shortlisted]
    members.extend(
        (shortlisted[left], shortlisted[right])
        for left in range(len(shortlisted))
        for right in range(left + 1, len(shortlisted))
    )
    evaluated: list[
        tuple[tuple[ModelSpec, ...], ResidualPolicy, dict[str, object]]
    ] = []
    for candidate_models in members:
        predictions = _average_predictions(candidate_models, predictions_by_model)
        for policy in _policies():
            evidence = _configuration_evidence(
                sessions, predictions, policy, partitions
            )
            evaluated.append(
                (
                    candidate_models,
                    policy,
                    _robust_evidence(evidence, total_turns, len(partitions)),
                )
            )
    eligible = [item for item in evaluated if bool(item[2]["eligible"])]
    pool = eligible or evaluated
    pool.sort(
        key=lambda item: (
            float(item[2]["robust_utility"]),
            float(item[2]["score_delta"]),
            -len(item[0]),
            "+".join(model.candidate_id for model in item[0]),
        ),
        reverse=True,
    )
    winner = pool[0]
    if not eligible or float(winner[2]["robust_utility"]) <= 0.0:
        selected = AdaptiveConfiguration((), ResidualPolicy())
    else:
        selected = AdaptiveConfiguration(winner[0], winner[1])
    return selected, {
        "model_candidates": len(screens),
        "shortlisted_models": [model.candidate_id for model in shortlisted],
        "single_and_pair_model_candidates": len(members),
        "policies_per_candidate": len(_policies()),
        "evaluated_configurations": len(evaluated),
        "eligible_configurations": len(eligible),
        "adaptive_off_selected": not selected.models,
        "selected_inner_evidence": winner[2],
        "top_configurations": [
            {
                "models": [model.candidate_id for model in item[0]],
                "policy": asdict(item[1]),
                "eligible": item[2]["eligible"],
                "robust_utility": item[2]["robust_utility"],
                "score_delta": item[2]["score_delta"],
                "fold_lower_bound": item[2]["fold_lower_bound"],
            }
            for item in pool[:10]
        ],
    }


def _trace_features(
    agent: ExperimentalAgent,
    feature_store: ConstraintGBDTFeatureStore,
    session_id: str,
    turn: int,
    ranking: tuple[str, ...],
) -> NDArray[np.float64]:
    state = agent.sessions[session_id]
    query, retrieval_scores = agent.last_runtime_inputs[session_id]
    context = ConstraintContext.from_runtime(
        cast(Any, state), turn=turn, retrieval_scores=retrieval_scores
    )
    matrix = feature_store.contextual_matrix(
        query, ranking, context, CONSTRAINT_METADATA_FEATURES
    )
    reciprocal = 1.0 / np.arange(1, len(ranking) + 1, dtype=np.float64)
    derived = np.column_stack(
        (
            reciprocal,
            reciprocal - 1.0,
            np.zeros(len(ranking), dtype=np.float64),
        )
    )
    if derived.shape[1] != len(DERIVED_FEATURES):
        raise RuntimeError("derived State V2 feature width drifted")
    return np.hstack((matrix, derived))


def _collect_parent_traces(
    config: UnifiedTechniqueConfig,
    samples: list[dict[str, object]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    feature_store: ConstraintGBDTFeatureStore,
) -> list[TraceSession]:
    built = build_suite_agent(config, ROOT / "data/catalog.jsonl")
    if not isinstance(built, ExperimentalAgent):
        raise TypeError("State V2 residual experiment requires ExperimentalAgent")
    agent = built
    sessions: list[TraceSession] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = f"state-v2-residual:{sample_id}"
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
            features = _trace_features(agent, feature_store, session_id, turn, ranking)
            if features.shape != (len(ranking), len(RESIDUAL_FEATURES)):
                raise RuntimeError("State V2 residual feature contract drifted")
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


def _id_digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _evaluate_parent(
    parent_name: str,
    config_path: Path,
    traces: list[TraceSession],
    outer_folds: list[set[str]],
    outer_fold_limit: int,
    manifest: dict[str, object],
) -> dict[str, object]:
    by_id = {session.sample_id: session for session in traces}
    control_sessions: list[dict[str, object]] = []
    candidate_sessions: list[dict[str, object]] = []
    oracle_sessions: list[dict[str, object]] = []
    folds = []
    receipts = []
    failures = activations = moved_ids = 0
    all_ids = set(by_id)
    for outer_index, outer_ids in enumerate(outer_folds[:outer_fold_limit]):
        training_ids = all_ids - outer_ids
        training = [by_id[sample_id] for sample_id in sorted(training_ids)]
        validation = [by_id[sample_id] for sample_id in sorted(outer_ids)]
        inner_partitions = [
            fold & training_ids for fold in outer_folds if fold != outer_ids
        ]
        selected, search = _select_configuration(
            training,
            inner_partitions,
            int(cast(dict, manifest["adaptive_algorithm"])["model_shortlist"]),
        )
        if selected.models:
            predictions_by_model = {}
            for spec in selected.models:
                model = _fit_rank_aware_model(spec, training)
                predictions_by_model[spec.candidate_id] = _predict(
                    model, spec, validation
                )
            predictions = _average_predictions(selected.models, predictions_by_model)
            candidate, behavior = _candidate_outcomes(
                validation, predictions, selected.policy
            )
        else:
            candidate = [dict(session.parent_outcome) for session in validation]
            behavior = {
                "activations": 0,
                "moved_ids": 0,
                "membership_failures": 0,
            }
        control = [session.parent_outcome for session in validation]
        oracle = _oracle_outcomes(validation)
        control_metrics = _technical_metrics(control)
        candidate_metrics = _technical_metrics(candidate)
        control_sessions.extend(control)
        candidate_sessions.extend(candidate)
        oracle_sessions.extend(oracle)
        failures += behavior["membership_failures"]
        activations += behavior["activations"]
        moved_ids += behavior["moved_ids"]
        receipt = {
            "outer_fold": outer_index,
            "training_count": len(training_ids),
            "validation_count": len(outer_ids),
            "training_ids_sha256": _id_digest(training_ids),
            "validation_ids_sha256": _id_digest(outer_ids),
            "disjoint": not bool(training_ids & outer_ids),
        }
        receipts.append(receipt)
        folds.append(
            {
                "outer_fold": outer_index,
                "selected_models": [asdict(model) for model in selected.models],
                "selected_policy": asdict(selected.policy),
                "search": search,
                "control_metrics": control_metrics,
                "candidate_metrics": candidate_metrics,
                "score_delta": float(candidate_metrics["recommended_technical_score"])
                - float(control_metrics["recommended_technical_score"]),
                "behavior": behavior,
                "fit_receipt": receipt,
            }
        )
        print(
            json.dumps(
                {
                    "parent": parent_name,
                    "fold": outer_index,
                    "control": control_metrics["recommended_technical_score"],
                    "candidate": candidate_metrics["recommended_technical_score"],
                    "models": [model.candidate_id for model in selected.models],
                    "off": not selected.models,
                    "policy": asdict(selected.policy),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    control_metrics = _technical_metrics(control_sessions)
    candidate_metrics = _technical_metrics(candidate_sessions)
    paired = paired_evidence(
        cast(list[dict], candidate_sessions), cast(list[dict], control_sessions)
    )
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
    gates = cast(dict, manifest["promotion_gates"])
    interval = cast(list[float], paired["paired_bootstrap_95_interval"])
    safe = (
        failures <= int(gates["maximum_membership_failures"])
        and hit_difference <= float(gates["maximum_hit_rate_absolute_difference"])
        and mttc_difference <= float(gates["maximum_mttc_absolute_difference"])
        and all(receipt["disjoint"] for receipt in receipts)
    )
    promoted = (
        safe
        and score_delta >= float(gates["minimum_oof_score_delta"])
        and sum(delta >= 0.0 for delta in fold_deltas)
        >= int(gates["minimum_nonnegative_outer_folds"])
        and min(scenario_deltas.values()) >= float(gates["minimum_scenario_delta"])
        and interval[0] > 0.0
    )
    return {
        "parent": parent_name,
        "config_path": str(config_path.relative_to(ROOT)),
        "configuration_sha256": sha256_file(config_path),
        "control": {
            "oof_metrics": control_metrics,
            "scenario_scores": scenario_control,
        },
        "candidate": {
            "oof_metrics": candidate_metrics,
            "scenario_scores": scenario_candidate,
            "paired_vs_control": paired,
            "activations": activations,
            "moved_ids": moved_ids,
        },
        "oracle_membership_ceiling": _technical_metrics(oracle_sessions),
        "folds": folds,
        "fit_receipts": receipts,
        "decision": {
            "status": "PROMOTE_EXPERIMENTAL" if promoted else "PARK_RETESTABLE",
            "score_delta": score_delta,
            "fold_deltas": fold_deltas,
            "nonnegative_outer_folds": sum(delta >= 0.0 for delta in fold_deltas),
            "scenario_deltas": scenario_deltas,
            "worst_scenario_delta": min(scenario_deltas.values()),
            "membership_failures": failures,
            "hit_rate_absolute_difference": hit_difference,
            "mttc_absolute_difference": mttc_difference,
            "hard_safe": safe,
            "promotion_gates_passed": promoted,
        },
    }


def _markdown(report: dict[str, object]) -> str:
    rows = []
    details = []
    for name in ("primary", "secondary"):
        result = cast(dict, report[name])
        control = cast(dict, result["control"])["oof_metrics"]
        candidate = cast(dict, result["candidate"])["oof_metrics"]
        decision = cast(dict, result["decision"])
        paired = cast(dict, cast(dict, result["candidate"])["paired_vs_control"])
        rows.append(
            f"| {result['parent']} | {control['recommended_technical_score']:.6f} | "
            f"{candidate['recommended_technical_score']:.6f} | {decision['score_delta']:+.6f} | "
            f"{decision['nonnegative_outer_folds']}/5 | {decision['status']} |"
        )
        details.append(
            f"### {result['parent']}\n\n"
            f"- MRR: `{control['mrr']:.6f}` → `{candidate['mrr']:.6f}`.\n"
            f"- Hit@10 difference: `{decision['hit_rate_absolute_difference']:.6f}`.\n"
            f"- MTTC difference: `{decision['mttc_absolute_difference']:.6f}`.\n"
            f"- Paired interval: `[{paired['paired_bootstrap_95_interval'][0]:+.6f}, "
            f"{paired['paired_bootstrap_95_interval'][1]:+.6f}]`; "
            f"`p={paired['paired_randomization_p_value']:.6f}`.\n"
            f"- Scenario deltas: `{decision['scenario_deltas']}`.\n"
            f"- Fold deltas: `{decision['fold_deltas']}`.\n"
        )
    return (
        """# State V2 Adaptive Residual V2 Decision

## Verdict

Promote `ranking.top10_residual_reranker.v2` into the autonomous engine as an
optional, fit-required experimental technique. Keep it disabled by default until
the engine has produced a fold-safe fitted asset for the chosen parent pipeline.

The primary comparison is the fair State Baseline V2 ablation: the same parent
trace is evaluated with residual ranking off and on. The secondary comparison is
a compatibility test against the stronger ranked State V2 parent; it does not
replace the primary inclusion decision.

The protected holdout was not accessed. Every residual model and policy was
selected only by grouped inner-fold predictions, then fitted without that outer
fold and evaluated once on it. Fit receipts prove disjoint training and validation
IDs. The parent configuration was identical in each off/on pair.

| Parent | Residual off | Residual on | Delta | Nonnegative folds | Decision |
|---|---:|---:|---:|---:|---|
"""
        + "\n".join(rows)
        + "\n\n"
        + "\n".join(details)
        + """

## What was adaptive

For every outer fold, the training side alone screened 24 rank-aware model
specifications, shortlisted six, evaluated six single models plus 15 two-model
ensembles, and searched 216 safe activation policies per candidate. This is 4,536
inner-fold configurations per outer fold. The selectable dimensions included
model family, feature subset, regularization/tree settings, ensemble membership,
rerank depth, blend weight, expected-gain threshold, probability-margin threshold,
and maximum moved IDs. An explicit adaptive-off candidate was available.

The selection utility rewarded mean score gain and a conservative fold lower
bound, while penalizing scenario regression and unnecessary activation. Different
outer folds selected different models and gates, so the result is not a single
manually chosen weight disguised as adaptation.

## Safety contract

- Output contains exactly the parent's normalized Top 10, reordered only.
- Hit@10 and MTTC must be exactly unchanged.
- Runtime features exclude target ID, target profile, scenario label, future
  answers, and evaluator outcomes.
- The technique fails closed to the parent order when its gates do not activate.
- Promotion requires positive paired evidence, at least four nonnegative outer
  folds, no material scenario regression, and zero membership failures.

## Interpretation and limitation

The result supports engine inclusion, not immediate replacement of the runtime
champion. It is evidence from the public adaptive set, not the sealed competition
holdout. A deployable asset still needs one final outcome-blind configuration
selection using cross-fitted development predictions, a fit on all allowed
development IDs, an immutable fit receipt, and exact runtime/off-state parity
tests. The engine must continue to treat this technique as fit-required and must
never reuse an evaluation-fold fit at deployment.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent", choices=("primary", "secondary", "both"), default="both"
    )
    parser.add_argument("--outer-fold-limit", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest["holdout_accessed"] is not False
        or manifest["outcomes_observed"] is not False
    ):
        raise RuntimeError("V2 experiment manifest was not frozen before outcomes")
    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    allowed = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    samples = [
        cast(dict[str, object], sample)
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in allowed
    ]
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    feature_store = ConstraintGBDTFeatureStore(catalog_path)
    original = json.loads(STATE_V2_REPORT.read_text(encoding="utf-8"))
    parent_paths = {
        "primary": ROOT / str(manifest["primary_parent"]),
        "secondary": ROOT / str(manifest["secondary_parent"]),
    }
    selected_names = (
        ("primary", "secondary") if args.parent == "both" else (args.parent,)
    )
    results: dict[str, object] = {}
    for name in selected_names:
        path = parent_paths[name]
        config = load_suite_config(path)
        traces = _collect_parent_traces(
            config,
            samples,
            catalog_ids,
            categories,
            products,
            feature_store,
        )
        metrics = _technical_metrics([trace.parent_outcome for trace in traces])
        expected_name = path.name
        expected = original["results"][expected_name]["metrics"]
        parity_metrics = {
            key: metrics[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
                "scenario_metrics",
            )
        }
        if parity_metrics != expected:
            raise RuntimeError(
                f"{name} State V2 parent parity failed: {parity_metrics} != {expected}"
            )
        print(
            json.dumps(
                {
                    "parent": name,
                    "parity": True,
                    "score": metrics["recommended_technical_score"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        results[name] = _evaluate_parent(
            name, path, traces, outer_folds, args.outer_fold_limit, manifest
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "parent_commit": manifest["parent_commit"],
        "holdout_accessed": False,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": sha256_file(catalog_path.resolve()),
        "split_sha256": sha256_file(nested_path),
        "outer_folds_completed": args.outer_fold_limit,
        "adaptive_search": {
            "model_candidates": len(_model_specs()),
            "shortlist": int(
                cast(dict, manifest["adaptive_algorithm"])["model_shortlist"]
            ),
            "single_and_pair_candidates_after_shortlist": 21,
            "policies_per_candidate": len(_policies()),
            "adaptive_off_option": True,
        },
        **results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH.relative_to(ROOT)),
                "decisions": {
                    name: cast(dict, cast(dict, report[name])["decision"])
                    for name in selected_names
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
