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

from baseline.state import ASK_ORDER
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
from ghostlab.retrieval.learned import (
    CandidateFeatureStore,
    LearnedLinearReranker,
    LinearRerankerModel,
)
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/catalog.jsonl"
MANIFEST = ROOT / "configs/experiments/learned_question_linear_v1.json"
OUTPUT_DIR = ROOT / "artifacts/experiments/learned_question_linear_v1"
REPORT_PATH = ROOT / "artifacts/reports/learned_question_linear_v1.json"
FIELD_WEIGHTS = (2.0, 8.0, 4.0, 2.5, 1.5, 1.0)
CHAMPION_SEQUENCE = (
    "other",
    "other",
    "use_case",
    "other",
    "size",
    "other",
    "other",
    "size",
)
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
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def make_agent(
    feature_store: CandidateFeatureStore,
    *,
    question_variant: str,
    question_model: LinearActionValueModel | None = None,
    question_order: tuple[str, ...] | None = None,
) -> ExperimentalAgent:
    ranking_model = LinearRerankerModel(
        weights=(0.0, 0.0, 0.634672535385014, 0.0, 0.0, 0.0, 0.4938576967870529),
        l2=0.1,
        training_pairs=32746,
    )
    return ExperimentalAgent(
        CATALOG,
        state_variant="raw_history",
        question_variant=question_variant,  # type: ignore[arg-type]
        question_order=question_order,
        learned_question_model=question_model,
        negative_evidence=False,
        retrieval_route="keyword",
        sparse_weights=FIELD_WEIGHTS,
        quality_prior_weight=0.2,
        learned_reranker=LearnedLinearReranker(feature_store, ranking_model),
    )


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


def scenario_reward(sessions: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session_reward(session))
    return {
        name: round(statistics.fmean(rewards), 6)
        for name, rewards in sorted(grouped.items())
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


class TimedAgent:
    def __init__(self, agent: ExperimentalAgent) -> None:
        self.agent = agent
        self.turn_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.turn_ms.append((time.perf_counter() - started) * 1000)
        return response


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["parent_commit"] != "189f0c6" or manifest["holdout_accessed"]:
        raise RuntimeError("invalid learned-question predeclaration")
    nested_path = ROOT / "configs/splits/nested_v1.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples_by_id = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    samples = [samples_by_id[sample_id] for sample_id in sorted(samples_by_id)]
    _, categories, products = catalog_index(CATALOG)
    feature_store = CandidateFeatureStore(CATALOG)

    collection_started = time.perf_counter()
    collection_agent = make_agent(
        feature_store,
        question_variant="sequence",
        question_order=CHAMPION_SEQUENCE,
    )
    training_states, labels = collect_counterfactual_question_states(
        collection_agent, samples, categories, products
    )
    for label in labels:
        label["continuation_policy"] = (
            "champion_absolute_turn_after_question;absorbing_stop"
        )
    collection_seconds = time.perf_counter() - collection_started
    del collection_agent
    gc.collect()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT_DIR / "counterfactual_labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels),
        encoding="utf-8",
    )

    folds: list[dict[str, object]] = []
    oof_sessions: list[dict] = []
    oof_traces: list[dict[str, object]] = []
    for fold_index, outer_values in enumerate(nested["outer_folds"]):
        outer = {str(value) for value in outer_values}
        fold_states = [
            state for state in training_states if state.sample_id not in outer
        ]
        model = fit_linear_action_value(fold_states, l2=1.0)
        agent = make_agent(
            feature_store, question_variant="learned", question_model=model
        )
        result = evaluate_replay(
            agent,
            [samples_by_id[sample_id] for sample_id in sorted(outer)],
            categories,
            products,
        )
        oof_sessions.extend(result["sessions"])
        oof_traces.extend(agent.question_trace)
        folds.append(
            {
                "outer_fold": fold_index,
                "training_session_count": len(adaptive_ids - outer),
                "training_state_count": len(fold_states),
                "validation_session_count": len(outer),
                "training_session_hash": session_set_hash(adaptive_ids - outer),
                "validation_session_hash": session_set_hash(outer),
                "model": model_payload(model),
                "outer_metrics": metric_subset(result),
            }
        )
        print(f"fold {fold_index}: {result['recommended_technical_score']}", flush=True)

    full_model = fit_linear_action_value(training_states, l2=1.0)
    model_path = OUTPUT_DIR / "linear_action_value_model.json"
    model_path.write_text(
        json.dumps(model_payload(full_model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    full_agent = make_agent(
        feature_store, question_variant="learned", question_model=full_model
    )
    full_result = evaluate_replay(full_agent, samples, categories, products)
    full_behavior = behavior_diagnostics(full_agent.question_trace)

    controls: dict[str, dict[str, object]] = {}
    control_specs = {
        "fixed_sequence_champion": ("sequence", CHAMPION_SEQUENCE),
        "heuristic_adaptive": ("adaptive", None),
        "fixed_no_other": ("sequence", tuple(ASK_ORDER)),
        "absorbing_stop": ("none", None),
    }
    champion_sessions: list[dict] = []
    for name, (variant, order) in control_specs.items():
        control_agent = make_agent(
            feature_store, question_variant=variant, question_order=order
        )
        control_result = evaluate_replay(control_agent, samples, categories, products)
        controls[name] = {
            "evaluation_label": "outer-fold/out-of-fold"
            if name == "fixed_sequence_champion"
            else "fixed non-learned development control",
            "metrics": metric_subset(control_result),
            "scenario_reward": scenario_reward(control_result["sessions"]),
            "behavior": behavior_diagnostics(control_agent.question_trace),
        }
        if name == "fixed_sequence_champion":
            champion_sessions = control_result["sessions"]
        print(
            f"control {name}: {control_result['recommended_technical_score']}",
            flush=True,
        )
        del control_agent
        gc.collect()

    # Stitching fold predictions is the only learned-policy generalization estimate.
    oof_by_id = {str(item["sample_id"]): item for item in oof_sessions}
    oof_sessions = [oof_by_id[sample_id] for sample_id in sorted(adaptive_ids)]
    oof_score = round(statistics.fmean(session_reward(s) for s in oof_sessions), 6)
    oof_overall = {
        "sample_count": len(oof_sessions),
        "hit_rate_at_10": round(
            statistics.fmean(float(item["hit"]) for item in oof_sessions), 6
        ),
        "mrr": round(
            statistics.fmean(float(item["reciprocal_rank"]) for item in oof_sessions),
            6,
        ),
        "mttc": round(
            statistics.fmean(
                item["first_hit_turn"] if item["first_hit_turn"] is not None else 11
                for item in oof_sessions
            ),
            6,
        ),
        "recommended_technical_score": oof_score,
        "scenario_reward": scenario_reward(oof_sessions),
    }
    deltas = paired_delta(oof_sessions, champion_sessions)
    interval = bootstrap_mean_interval(deltas, resamples=BOOTSTRAP_RESAMPLES, seed=SEED)
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
    }
    champion_scenario = scenario_reward(champion_sessions)
    oof_scenario = scenario_reward(oof_sessions)
    scenario_deltas = {
        name: round(oof_scenario[name] - champion_scenario[name], 6)
        for name in champion_scenario
    }

    oof_path = OUTPUT_DIR / "oof_sessions.jsonl"
    oof_path.write_text(
        "".join(
            json.dumps(
                {**session, "evaluation_label": "outer-fold/out-of-fold"},
                sort_keys=True,
            )
            + "\n"
            for session in oof_sessions
        ),
        encoding="utf-8",
    )

    # Determinism and runtime are measured on the deployable all-development refit.
    repeat_agent = make_agent(
        feature_store, question_variant="learned", question_model=full_model
    )
    repeat_result = evaluate_replay(repeat_agent, samples, categories, products)
    deterministic = repeat_result["sessions"] == full_result["sessions"] and [
        item["ask_attribute"] for item in repeat_agent.question_trace
    ] == [item["ask_attribute"] for item in full_agent.question_trace]
    cold_started = time.perf_counter()
    timed_underlying = make_agent(
        feature_store, question_variant="learned", question_model=full_model
    )
    cold_start_seconds = time.perf_counter() - cold_started
    timed_agent = TimedAgent(timed_underlying)
    timed_result = evaluate_replay(timed_agent, samples, categories, products)
    runtime_files = (
        ROOT / "ghostlab/policy/learned_questions.py",
        ROOT / "ghostlab/runtime/experimental.py",
        model_path,
    )
    performance = {
        "challenger_agent_init_seconds": round(cold_start_seconds, 6),
        "warm_turn_mean_ms": round(statistics.fmean(timed_agent.turn_ms), 6),
        "warm_turn_p95_ms": round(percentile(timed_agent.turn_ms, 0.95), 6),
        "warm_turn_max_ms": round(max(timed_agent.turn_ms), 6),
        "peak_process_memory_mb": round(peak_rss_mb(), 3),
        "model_asset_mb": round(model_path.stat().st_size / 1024 / 1024, 6),
        "challenger_code_and_asset_mb": round(
            sum(path.stat().st_size for path in runtime_files) / 1024 / 1024, 6
        ),
        "external_calls_per_turn": 0,
        "reported_tokens": timed_result["reported_token_usage"],
        "budgets": {
            "challenger_agent_init_seconds": 30.0,
            "warm_turn_p95_ms": 500.0,
            "peak_process_memory_mb": 4096.0,
            "model_asset_mb": 500.0,
        },
    }
    performance_passed = bool(
        cold_start_seconds <= 30.0
        and percentile(timed_agent.turn_ms, 0.95) <= 500.0
        and peak_rss_mb() <= 4096.0
        and model_path.stat().st_size / 1024 / 1024 <= 500.0
    )
    performance["passed"] = performance_passed

    fold_scores = [
        float(fold["outer_metrics"]["recommended_technical_score"])  # type: ignore[index]
        for fold in folds
    ]
    unexplained_regression = min(scenario_deltas.values()) < -0.02
    mean_delta = statistics.fmean(deltas)
    promote = bool(
        mean_delta > 0
        and interval[0] > 0
        and not unexplained_regression
        and performance_passed
        and deterministic
        and behavior_diagnostics(oof_traces)["illegal_action_count"] == 0
    )
    decision = "PROMOTED" if promote else "PARKED_STANDALONE"
    report = {
        "experiment_id": manifest["experiment_id"],
        "family": "question",
        "parent_commit": "189f0c6",
        "split": "nested_v1",
        "evaluation_label": "outer-fold/out-of-fold",
        "holdout_accessed": False,
        "failure_status": None,
        "manifest_sha256": file_hash(MANIFEST),
        "code_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (
                ROOT / "ghostlab/policy/learned_questions.py",
                ROOT / "ghostlab/research/learned_questions.py",
                ROOT / "ghostlab/runtime/experimental.py",
                ROOT / "scripts/run_learned_question_challenger.py",
            )
        },
        "dataset_sha256": file_hash(ROOT / "data/public_set.jsonl"),
        "catalog_sha256": file_hash(CATALOG),
        "split_sha256": file_hash(nested_path),
        "training_session_hash": session_set_hash(adaptive_ids),
        "session_id_source": {
            "path": "configs/splits/nested_v1.json",
            "validation_ids": "outer_folds[outer_fold]",
            "training_ids": "adaptive_sample_ids minus outer_folds[outer_fold]",
        },
        "seed": SEED,
        "counterfactual_collection": {
            "state_count": len(training_states),
            "label_count": len(labels),
            "continuation_policy": manifest["continuation_policy"],
            "elapsed_seconds": round(collection_seconds, 6),
            "label_path": str(labels_path.relative_to(ROOT)),
            "diagnostics": first_action_diagnostics(training_states, labels),
        },
        "model": {
            "ladder_position": "regularized linear action-value; no tree/GBDT run",
            "selection": "single predeclared l2=1.0; no HPO",
            "full_training_asset": str(model_path.relative_to(ROOT)),
            "full_training_asset_sha256": file_hash(model_path),
        },
        "controls": controls,
        "oof": {
            "metrics": oof_overall,
            "session_path": str(oof_path.relative_to(ROOT)),
            "behavior": behavior_diagnostics(oof_traces),
            "fold_scores": [round(value, 6) for value in fold_scores],
            "fold_score_mean": round(statistics.fmean(fold_scores), 6),
            "fold_score_standard_deviation": round(statistics.pstdev(fold_scores), 6),
            "worst_fold_score": round(min(fold_scores), 6),
            "scenario_reward_deltas_vs_champion": scenario_deltas,
            "paired_evidence_vs_champion": paired,
        },
        "folds": folds,
        "all_development_refit": {
            "evaluation_label": "all-development refit",
            "metrics": metric_subset(full_result),
            "behavior": full_behavior,
            "deterministic_repeat": deterministic,
        },
        "performance_and_packaging": performance,
        "decision": {
            "status": decision,
            "promotion_rule_passed": promote,
            "reason": (
                "Positive stable OOF evidence passed every declared gate."
                if promote
                else "The predeclared promotion rule was not met; the linear policy does not justify replacing the fixed champion."
            ),
            "shallow_model_justified": bool(
                mean_delta > 0 and interval[1] > 0 and not promote
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "oof": report["oof"],
                "all_development_refit": report["all_development_refit"],
                "performance_and_packaging": performance,
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
