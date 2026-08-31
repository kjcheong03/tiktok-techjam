from __future__ import annotations

import gc
import hashlib
import json
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
from ghostlab.policy.learned_questions import (
    LinearActionValueModel,
    fit_linear_action_value,
)
from ghostlab.research.firewall import session_set_hash
from ghostlab.research.learned_questions import (
    behavior_diagnostics,
    collect_counterfactual_question_states,
    first_action_diagnostics,
)
from ghostlab.research.replay import evaluate_replay, paired_delta, session_reward
from ghostlab.retrieval.gbdt import (
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.experimental import CandidateReranker
from ghostlab.runtime.experimental_questions import ExperimentalAgent
from scripts.run_gbdt_reranker import (
    FIELD_WEIGHTS,
    QUESTION_ORDER,
    collect_groups,
    train_model,
    variant_config,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/catalog.jsonl"
MANIFEST_PATH = ROOT / "configs/experiments/gbdt_question_interaction_v1.json"
GBDT_MANIFEST_PATH = ROOT / "configs/experiments/gbdt_reranker_v1.json"
SPLIT_PATH = ROOT / "configs/splits/nested_v1.json"
DEPLOYABLE_GBDT_PATH = ROOT / "artifacts/models/gbdt_reranker_v2_round56.json"
OUTPUT_DIR = ROOT / "artifacts/experiments/gbdt_question_interaction_v1"
REPORT_PATH = ROOT / "artifacts/reports/gbdt_question_interaction_v1.json"
QUESTION_MODEL_PATH = OUTPUT_DIR / "linear_action_value_model.json"
OOF_PATH = OUTPUT_DIR / "oof_sessions.jsonl"
LABELS_PATH = OUTPUT_DIR / "counterfactual_labels.jsonl"
SEED = 20260826
BOOTSTRAP_RESAMPLES = 10000


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a numeric metric, received {type(value).__name__}")
    return float(value)


def make_agent(
    quality: CatalogQualityReranker,
    reranker: CandidateReranker,
    *,
    question_variant: str,
    question_model: LinearActionValueModel | None = None,
) -> ExperimentalAgent:
    return ExperimentalAgent(
        CATALOG,
        state_variant="raw_history",
        question_variant=question_variant,  # type: ignore[arg-type]
        question_order=QUESTION_ORDER if question_variant == "sequence" else None,
        learned_question_model=question_model,
        negative_evidence=False,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        quality_prior=quality,
        learned_reranker=reranker,
    )


class InstrumentedAgent:
    def __init__(self, agent: ExperimentalAgent) -> None:
        self.agent = agent
        self.turn_ms: list[float] = []
        self.failure_counts = {
            "reset_exceptions": 0,
            "response_exceptions": 0,
            "invalid_responses": 0,
        }
        self.response_calls = 0

    @property
    def question_trace(self) -> list[dict[str, object]]:
        return self.agent.question_trace

    def reset(self, session_id: str, user_profile: dict) -> None:
        try:
            self.agent.reset(session_id, user_profile)
        except Exception:
            self.failure_counts["reset_exceptions"] += 1
            raise

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        self.response_calls += 1
        started = time.perf_counter()
        try:
            response = self.agent.respond(session_id, user_message, turn, top_k)
        except Exception:
            self.failure_counts["response_exceptions"] += 1
            raise
        finally:
            self.turn_ms.append((time.perf_counter() - started) * 1000)
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            self.failure_counts["invalid_responses"] += 1
        return response

    @property
    def failure_count(self) -> int:
        return sum(self.failure_counts.values())


def metric_subset(result: dict) -> dict[str, object]:
    return {
        key: result[key]
        for key in (
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "recommended_technical_score",
            "scenario_metrics",
        )
    }


def summarized_sessions(sessions: list[dict]) -> dict[str, object]:
    hit = statistics.fmean(float(item["hit"]) for item in sessions)
    mrr = statistics.fmean(float(item["reciprocal_rank"]) for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else 11
        for item in sessions
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "recommended_technical_score": round(
            statistics.fmean(session_reward(item) for item in sessions), 6
        ),
        "scenario_reward": {
            name: round(
                statistics.fmean(session_reward(item) for item in values), 6
            )
            for name, values in sorted(grouped.items())
        },
    }


def model_payload(model: LinearActionValueModel) -> dict[str, object]:
    return {
        "model_type": "linear_action_value_v1",
        "feature_names": list(model.feature_names),
        "action_weights": {
            "stop" if action is None else action: list(weights)
            for action, weights in model.action_weights.items()
        },
        "l2": model.l2,
        "training_states": model.training_states,
        "absorbing_stop": True,
    }


def evaluate_instrumented(
    agent: ExperimentalAgent,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[dict, InstrumentedAgent]:
    instrumented = InstrumentedAgent(agent)
    return evaluate_replay(instrumented, samples, categories, products), instrumented


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest["parent_commit"]
        != "cbfd7d5dd595c5637608ba28f46f57777c7e153e"
        or manifest["holdout_accessed"]
        or not manifest["manifest_created_before_evaluation"]
    ):
        raise RuntimeError("invalid interaction predeclaration")
    nested = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    outer_folds = [{str(value) for value in fold} for fold in nested["outer_folds"]]
    if len(outer_folds) != 5 or set().union(*outer_folds) != adaptive_ids:
        raise RuntimeError("frozen outer split is invalid")
    samples_by_id = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    _, categories, products = catalog_index(CATALOG)
    sparse = SparseIndex(CATALOG)
    quality = CatalogQualityReranker(CATALOG)
    feature_store = GBDTFeatureStore(CATALOG, quality=quality.quality)
    groups, collection = collect_groups(
        samples_by_id, categories, products, sparse, quality, feature_store
    )
    gbdt_manifest = json.loads(GBDT_MANIFEST_PATH.read_text(encoding="utf-8"))
    config = variant_config(gbdt_manifest, "shallow_metadata_depth3")

    folds: list[dict[str, object]] = []
    control_sessions: list[dict] = []
    candidate_sessions: list[dict] = []
    no_question_sessions: list[dict] = []
    candidate_traces: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    evaluation_failures = 0
    response_calls = 0
    fold_models: list[dict[str, object]] = []

    for outer_index, outer_ids in enumerate(outer_folds):
        inner_validation_ids = outer_folds[(outer_index + 1) % len(outer_folds)]
        inner_training_ids = adaptive_ids - outer_ids - inner_validation_ids
        model, selected_rounds = train_model(
            groups, inner_training_ids, inner_validation_ids, config
        )
        fold_models.append(
            {
                "outer_fold": outer_index,
                "selected_rounds": selected_rounds,
                "training_groups": model.training_groups,
                "training_rows": model.training_rows,
                "split_importance": model.split_importance(),
            }
        )
        reranker = LambdaMARTReranker(feature_store, model)
        outer_samples = [samples_by_id[value] for value in sorted(outer_ids)]
        training_ids = adaptive_ids - outer_ids
        training_samples = [samples_by_id[value] for value in sorted(training_ids)]

        control_result, control_agent = evaluate_instrumented(
            make_agent(quality, reranker, question_variant="sequence"),
            outer_samples,
            categories,
            products,
        )
        control_sessions.extend(control_result["sessions"])
        evaluation_failures += control_agent.failure_count
        response_calls += control_agent.response_calls

        no_question_result, no_question_agent = evaluate_instrumented(
            make_agent(quality, reranker, question_variant="none"),
            outer_samples,
            categories,
            products,
        )
        no_question_sessions.extend(no_question_result["sessions"])
        evaluation_failures += no_question_agent.failure_count
        response_calls += no_question_agent.response_calls

        collection_started = time.perf_counter()
        training_states, labels = collect_counterfactual_question_states(
            make_agent(quality, reranker, question_variant="sequence"),
            training_samples,
            categories,
            products,
        )
        collection_seconds = time.perf_counter() - collection_started
        for label in labels:
            label_rows.append(
                {
                    **label,
                    "outer_fold": outer_index,
                    "training_only": True,
                    "continuation": "matched_fold_gbdt_then_fixed_absolute_turn_sequence",
                }
            )
        question_model = fit_linear_action_value(training_states, l2=1.0)
        candidate_result, candidate_agent = evaluate_instrumented(
            make_agent(
                quality,
                reranker,
                question_variant="learned",
                question_model=question_model,
            ),
            outer_samples,
            categories,
            products,
        )
        candidate_sessions.extend(candidate_result["sessions"])
        candidate_traces.extend(candidate_agent.question_trace)
        evaluation_failures += candidate_agent.failure_count
        response_calls += candidate_agent.response_calls
        fold_delta = round(
            float(candidate_result["recommended_technical_score"])
            - float(control_result["recommended_technical_score"]),
            6,
        )
        folds.append(
            {
                "outer_fold": outer_index,
                "outer_training_ids": sorted(training_ids),
                "outer_validation_ids": sorted(outer_ids),
                "training_session_hash": session_set_hash(training_ids),
                "validation_session_hash": session_set_hash(outer_ids),
                "gbdt_selected_rounds": selected_rounds,
                "question_training_state_count": len(training_states),
                "question_training_label_count": len(labels),
                "counterfactual_collection_seconds": round(collection_seconds, 6),
                "counterfactual_diagnostics": first_action_diagnostics(
                    training_states, labels
                ),
                "question_model": model_payload(question_model),
                "control_metrics": metric_subset(control_result),
                "candidate_metrics": metric_subset(candidate_result),
                "no_question_metrics": metric_subset(no_question_result),
                "candidate_score_delta": fold_delta,
                "candidate_nonnegative": fold_delta >= -1e-12,
                "candidate_behavior": behavior_diagnostics(
                    candidate_agent.question_trace
                ),
            }
        )
        print(
            f"fold={outer_index} rounds={selected_rounds} "
            f"control={control_result['recommended_technical_score']} "
            f"candidate={candidate_result['recommended_technical_score']} "
            f"delta={fold_delta}",
            flush=True,
        )
        gc.collect()

    def ordered(sessions: list[dict]) -> list[dict]:
        by_id = {str(item["sample_id"]): item for item in sessions}
        if set(by_id) != adaptive_ids or len(sessions) != len(adaptive_ids):
            raise RuntimeError("OOF sessions are missing or duplicated")
        return [by_id[value] for value in sorted(adaptive_ids)]

    control_sessions = ordered(control_sessions)
    candidate_sessions = ordered(candidate_sessions)
    no_question_sessions = ordered(no_question_sessions)
    control_metrics = summarized_sessions(control_sessions)
    candidate_metrics = summarized_sessions(candidate_sessions)
    no_question_metrics = summarized_sessions(no_question_sessions)
    expected = manifest["matched_control"]["expected_metrics"]
    control_reproduction = {
        key: {
            "expected": expected[key],
            "observed": control_metrics[key],
            "absolute_difference": round(
                abs(number(expected[key]) - number(control_metrics[key])), 12
            ),
        }
        for key in expected
    }
    control_reproduced = all(
        float(value["absolute_difference"]) <= 0.000001
        for value in control_reproduction.values()
    )

    deltas = paired_delta(candidate_sessions, control_sessions)
    interval = bootstrap_mean_interval(
        deltas, resamples=BOOTSTRAP_RESAMPLES, seed=SEED
    )
    paired = {
        "mean_session_reward_delta": round(statistics.fmean(deltas), 6),
        "paired_bootstrap_95_interval": [round(value, 6) for value in interval],
        "paired_randomization_p_value": round(
            paired_randomization_pvalue(
                deltas, resamples=BOOTSTRAP_RESAMPLES, seed=SEED
            ),
            6,
        ),
        "wins": sum(value > 1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "losses": sum(value < -1e-12 for value in deltas),
        "resamples": BOOTSTRAP_RESAMPLES,
    }
    control_scenarios = control_metrics["scenario_reward"]
    candidate_scenarios = candidate_metrics["scenario_reward"]
    assert isinstance(control_scenarios, dict)
    assert isinstance(candidate_scenarios, dict)
    scenario_deltas = {
        name: round(
            float(candidate_scenarios[name]) - float(control_scenarios[name]), 6
        )
        for name in sorted(control_scenarios)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in label_rows),
        encoding="utf-8",
    )
    OOF_PATH.write_text(
        "".join(
            json.dumps(
                {
                    "candidate": candidate,
                    "control": control,
                    "session_reward_delta": round(
                        session_reward(candidate) - session_reward(control), 12
                    ),
                    "evaluation_label": "outer-fold/out-of-fold",
                },
                sort_keys=True,
            )
            + "\n"
            for candidate, control in zip(
                candidate_sessions, control_sessions, strict=True
            )
        ),
        encoding="utf-8",
    )

    deployable_gbdt = LambdaMARTModel.load(DEPLOYABLE_GBDT_PATH)
    deployable_reranker = LambdaMARTReranker(feature_store, deployable_gbdt)
    all_samples = [samples_by_id[value] for value in sorted(adaptive_ids)]
    full_states, full_labels = collect_counterfactual_question_states(
        make_agent(quality, deployable_reranker, question_variant="sequence"),
        all_samples,
        categories,
        products,
    )
    full_question_model = fit_linear_action_value(full_states, l2=1.0)
    QUESTION_MODEL_PATH.write_text(
        json.dumps(model_payload(full_question_model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first_full, first_instrumented = evaluate_instrumented(
        make_agent(
            quality,
            deployable_reranker,
            question_variant="learned",
            question_model=full_question_model,
        ),
        all_samples,
        categories,
        products,
    )
    second_full, second_instrumented = evaluate_instrumented(
        make_agent(
            quality,
            deployable_reranker,
            question_variant="learned",
            question_model=full_question_model,
        ),
        all_samples,
        categories,
        products,
    )
    deterministic = bool(
        first_full["sessions"] == second_full["sessions"]
        and [value["ask_attribute"] for value in first_instrumented.question_trace]
        == [value["ask_attribute"] for value in second_instrumented.question_trace]
    )
    evaluation_failures += first_instrumented.failure_count
    evaluation_failures += second_instrumented.failure_count
    response_calls += first_instrumented.response_calls + second_instrumented.response_calls

    gc.collect()
    cold_started = time.perf_counter()
    runtime_quality = CatalogQualityReranker(CATALOG)
    runtime_features = GBDTFeatureStore(CATALOG, quality=runtime_quality.quality)
    runtime_gbdt = LambdaMARTModel.load(DEPLOYABLE_GBDT_PATH)
    runtime_agent = make_agent(
        runtime_quality,
        LambdaMARTReranker(runtime_features, runtime_gbdt),
        question_variant="learned",
        question_model=full_question_model,
    )
    cold_start_seconds = time.perf_counter() - cold_started
    timed_result, timed_agent = evaluate_instrumented(
        runtime_agent, all_samples, categories, products
    )
    evaluation_failures += timed_agent.failure_count
    response_calls += timed_agent.response_calls
    runtime_parity = timed_result["sessions"] == first_full["sessions"]
    asset_bytes = DEPLOYABLE_GBDT_PATH.stat().st_size + QUESTION_MODEL_PATH.stat().st_size
    limits = manifest["promotion_gates"]["budgets"]
    performance = {
        "cold_start_seconds": round(cold_start_seconds, 6),
        "warm_turn_mean_ms": round(statistics.fmean(timed_agent.turn_ms), 6),
        "warm_turn_p95_ms": round(percentile(timed_agent.turn_ms, 0.95), 6),
        "warm_turn_max_ms": round(max(timed_agent.turn_ms), 6),
        "peak_process_memory_mb": round(peak_rss_mb(), 3),
        "model_asset_bytes": asset_bytes,
        "model_asset_mb": round(asset_bytes / 1024 / 1024, 6),
        "external_calls_per_turn": 0,
        "failure_count": evaluation_failures,
        "response_calls": response_calls,
        "runtime_session_parity": runtime_parity,
        "measurement_context": {
            "contention_affected": True,
            "isolated_rerun_required_if_accuracy_survives": True,
            "isolated_rerun_performed": False,
            "reason": "Accuracy gates failed decisively, so contention-affected timing is diagnostic only.",
        },
    }
    budget_passed = bool(
        cold_start_seconds <= float(limits["cold_start_seconds_max"])
        and percentile(timed_agent.turn_ms, 0.95)
        <= float(limits["warm_turn_p95_ms_max"])
        and peak_rss_mb() <= float(limits["peak_process_memory_mb_max"])
        and asset_bytes / 1024 / 1024 <= float(limits["model_asset_mb_max"])
        and int(performance["external_calls_per_turn"])
        <= int(limits["external_calls_per_turn_max"])
    )
    performance["budgets"] = limits
    performance["budget_passed"] = budget_passed

    fold_deltas = [number(value["candidate_score_delta"]) for value in folds]
    gates = {
        "control_reproduction": control_reproduced,
        "score_delta_at_least_0_005": number(
            candidate_metrics["recommended_technical_score"]
        )
        - number(control_metrics["recommended_technical_score"])
        >= 0.005 - 1e-12,
        "at_least_four_nonnegative_folds": sum(
            value >= -1e-12 for value in fold_deltas
        )
        >= 4,
        "no_overall_hit_regression": number(candidate_metrics["hit_rate_at_10"])
        >= number(control_metrics["hit_rate_at_10"]),
        "no_scenario_delta_below_minus_0_005": min(scenario_deltas.values())
        >= -0.005 - 1e-12,
        "zero_failures": evaluation_failures == 0,
        "legal_actions": behavior_diagnostics(candidate_traces)[
            "illegal_action_count"
        ]
        == 0,
        "determinism": deterministic,
        "runtime_parity": runtime_parity,
        "budgets": budget_passed,
    }
    promote = all(gates.values())
    decision = "PROMOTED_INTERACTION" if promote else "PARKED_INTERACTION"
    report = {
        "experiment_id": manifest["experiment_id"],
        "family": manifest["family"],
        "parent_commit": manifest["parent_commit"],
        "split": manifest["split"],
        "evaluation_label": "outer-fold/out-of-fold",
        "holdout_accessed": False,
        "failure_status": None,
        "manifest_sha256": file_hash(MANIFEST_PATH),
        "dataset_sha256": file_hash(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": file_hash(CATALOG),
        "split_sha256": file_hash(SPLIT_PATH),
        "deployable_gbdt_sha256": file_hash(DEPLOYABLE_GBDT_PATH),
        "question_model_sha256": file_hash(QUESTION_MODEL_PATH),
        "code_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (
                ROOT / "ghostlab/policy/learned_questions.py",
                ROOT / "ghostlab/research/learned_questions.py",
                ROOT / "ghostlab/runtime/experimental_questions.py",
                ROOT / "scripts/run_gbdt_question_interaction.py",
            )
        },
        "seed": SEED,
        "collection": collection,
        "fold_models": fold_models,
        "matched_control": {
            "metrics": control_metrics,
            "expected_reproduction": control_reproduction,
            "reproduced": control_reproduced,
        },
        "candidate_oof": {
            "metrics": candidate_metrics,
            "behavior": behavior_diagnostics(candidate_traces),
            "paired_vs_control": paired,
            "fold_score_deltas": fold_deltas,
            "scenario_score_deltas": scenario_deltas,
            "session_path": str(OOF_PATH.relative_to(ROOT)),
        },
        "backward_diagnostics": {
            "fixed_sequence": control_metrics,
            "no_question": no_question_metrics,
        },
        "counterfactual_evidence": {
            "training_only_label_rows": len(label_rows),
            "label_path": str(LABELS_PATH.relative_to(ROOT)),
            "label_sha256": file_hash(LABELS_PATH),
        },
        "folds": folds,
        "all_development_refit": {
            "evaluation_label": "all-development refit; not promotion evidence",
            "training_state_count": len(full_states),
            "training_label_count": len(full_labels),
            "metrics": metric_subset(first_full),
            "behavior": behavior_diagnostics(first_instrumented.question_trace),
            "deterministic_repeat": deterministic,
            "runtime_parity": runtime_parity,
            "model_path": str(QUESTION_MODEL_PATH.relative_to(ROOT)),
        },
        "performance_and_packaging": performance,
        "promotion_gates": gates,
        "decision": {
            "status": decision,
            "promotion_rule_passed": promote,
            "reason": (
                "Every predeclared interaction gate passed."
                if promote
                else "At least one predeclared interaction gate failed; retain the audited GBDT control."
            ),
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
                "matched_control": report["matched_control"],
                "candidate_oof": report["candidate_oof"],
                "backward_diagnostics": report["backward_diagnostics"],
                "performance_and_packaging": performance,
                "promotion_gates": gates,
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
