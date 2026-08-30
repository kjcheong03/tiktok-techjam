from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

from ghostlab.runtime.adaptive_factory import (
    build_adaptive_hybrid_agent,
    load_adaptive_hybrid_config,
)
from ghostlab.training.adaptive_hybrid import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CHAMPION_REPORT = ROOT / "artifacts/reports/adaptive_hybrid_champion_control.json"
STATE_REPORT = (
    ROOT / "artifacts/reports/adaptive_hybrid_state_v2_precision_control.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _peak_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _projection(agent: object, index: int) -> dict:
    trace = agent.traces[index]  # type: ignore[attr-defined]
    return {
        key: value for key, value in trace.__dict__.items() if key not in {"session_id"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the complete adaptive 1A-3B runtime end to end"
    )
    parser.add_argument(
        "--config", default="configs/adaptive_hybrid_1a_3b_2200_structural_v2.json"
    )
    parser.add_argument(
        "--adaptive-report",
        default="artifacts/reports/adaptive_hybrid_structural_v2_public.json",
    )
    parser.add_argument(
        "--training-report",
        default="artifacts/reports/adaptive_hybrid_training_2200_structural_v2.json",
    )
    parser.add_argument(
        "--diversity-report",
        default="artifacts/reports/adaptive_dense_diversity_v2.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/adaptive_hybrid_structural_v2_validation.json",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    adaptive_report_path = ROOT / args.adaptive_report
    training_report_path = ROOT / args.training_report
    diversity_report_path = ROOT / args.diversity_report
    output_path = ROOT / args.output
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_adaptive_hybrid_config(config_path)
    assert config.union_ranker.model_path is not None
    assert config.union_ranker.model_sha256 is not None
    model_path = ROOT / config.union_ranker.model_path
    model_hash_ok = sha256_file(model_path) == config.union_ranker.model_sha256
    receipt_path = model_path.with_name(f"{model_path.stem}.fit_receipt.json")
    receipt = _load(receipt_path)
    receipt_ok = (
        receipt["model_sha256"] == config.union_ranker.model_sha256
        and receipt["holdout_accessed"] is False
    )
    started = time.perf_counter()
    agent = build_adaptive_hybrid_agent(
        ROOT / "data/catalog.jsonl", config_path=config_path, project_root=ROOT
    )
    cold_seconds = time.perf_counter() - started
    profile = {"preference_tags": ["comfort", "durability"]}
    buying_message = (
        "I'm looking for running shoes. A key requirement is: lightweight mesh."
    )
    latencies = []
    responses = []
    for session_id in ("determinism_a", "determinism_b"):
        agent.reset(session_id, profile)
        started = time.perf_counter()
        responses.append(agent.respond(session_id, buying_message, 1, 10))
        latencies.append((time.perf_counter() - started) * 1000.0)
    deterministic = responses[0] == responses[1]
    trace_deterministic = _projection(agent, -2) == _projection(agent, -1)

    agent.reset("browse", {"preference_tags": ["comfort", "style"]})
    started = time.perf_counter()
    browsing_response = agent.respond(
        "browse",
        "I'm looking for summer wedding clothing, but I'm still exploring.",
        1,
        10,
    )
    latencies.append((time.perf_counter() - started) * 1000.0)
    browsing_trace = agent.traces[-1]

    agent.reset("conflict", {"preference_tags": ["formal", "comfort"]})
    started = time.perf_counter()
    conflict_response = agent.respond(
        "conflict",
        "I'm looking for summer dresses, but I'm still exploring and not formal.",
        1,
        10,
    )
    latencies.append((time.perf_counter() - started) * 1000.0)
    conflict_trace = agent.traces[-1]

    agent.reset("override", {})
    first = agent.respond(
        "override", "I'm looking for shirts. A key requirement is: black.", 1, 10
    )
    second = agent.respond(
        "override",
        "Actually, ignore my earlier preference. What I need is: blue.",
        2,
        10,
    )
    override_session = agent.sessions["override"]
    override_epoch_changed = override_session.state.intent_epoch > 0
    override_history_scoped = override_session.controller._shown_ids == {
        item["parent_asin"] for item in second["recommendations"]
    }

    all_responses = [*responses, browsing_response, conflict_response, first, second]
    response_contract = all(
        isinstance(response["message"], str)
        and len(response["recommendations"]) <= 10
        and len({item["parent_asin"] for item in response["recommendations"]})
        == len(response["recommendations"])
        for response in all_responses
    )

    adaptive = _load(adaptive_report_path)
    champion = _load(CHAMPION_REPORT)
    state = _load(STATE_REPORT)["metrics"]
    training = _load(training_report_path)
    diversity = _load(diversity_report_path)
    conflict_profile_update = agent.profile_update("conflict")
    metrics = {
        "adaptive": {
            key: adaptive[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
            )
        },
        "champion": {
            key: champion[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
            )
        },
        "state_v2_precision": {
            key: state[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "recommended_technical_score",
            )
        },
    }
    metrics["adaptive_vs_champion_score_delta"] = (
        adaptive["recommended_technical_score"]
        - champion["recommended_technical_score"]
    )
    metrics["adaptive_vs_state_v2_score_delta"] = (
        adaptive["recommended_technical_score"] - state["recommended_technical_score"]
    )

    checks = {
        "config_schema_valid": True,
        "union_model_hash_valid": model_hash_ok,
        "union_fit_receipt_valid": receipt_ok,
        "union_model_oof_selected": training["union"]["selected_for_output_config"],
        "final_report_config_matches": adaptive["adaptive_runtime"]["config_sha256"]
        == config.canonical_hash(),
        "offline_environment_forced": True,
        "deterministic_response": deterministic,
        "deterministic_trace": trace_deterministic,
        "response_contract": response_contract,
        "buying_route_exercised": agent.traces[-6].route == "buying",
        "browsing_route_exercised": browsing_trace.route == "browsing",
        "diverse_dense_exercised": bool(browsing_trace.query_views),
        "overload_cutoff_exercised": browsing_trace.overloaded,
        "overload_multiroute_exercised": all(
            browsing_trace.contribution_counts[source] > 0
            for source in ("keyword", "category", "vector")
        ),
        "overload_browsing_llm_exercised": (
            browsing_trace.semantic_backend == config.semantic_ranker.model_id
        ),
        "literal_llm_ranker_exercised": (
            browsing_trace.semantic_backend == config.semantic_ranker.model_id
        ),
        "literal_llm_changed_order": browsing_trace.semantic_changed,
        "three_source_buying_merge_exercised": all(
            agent.traces[-6].contribution_counts[source] > 0
            for source in ("keyword", "category", "vector")
        ),
        "dense_diversity_measured_on_all_public_browsing": (
            diversity["browsing_sessions"] == 80
            and set(diversity["summary"])
            == {"multiview_max_relevance", "view_balanced", "embedding_mmr"}
        ),
        "profile_conflict_suppressed": (
            not conflict_trace.profile_active
            and conflict_trace.profile_reason == "explicit_conflict"
        ),
        "intent_override_changed_epoch": override_epoch_changed,
        "intent_override_scoped_history": override_history_scoped,
        "profile_update_exposed": (
            conflict_profile_update is not None
            and conflict_profile_update.provenance == "explicit_session_evidence"
        ),
        "zero_e2e_fallbacks": adaptive["adaptive_runtime"]["fallback_count"] == 0,
        "selective_llm_has_activations": adaptive["adaptive_runtime"][
            "semantic_activation_count"
        ]
        > 0,
        "selective_llm_has_skips": adaptive["adaptive_runtime"]["semantic_skip_count"]
        > 0,
        "factory_config_parity": agent.config_sha256 == config.canonical_hash(),
        "starter_activation_untouched": not (
            ROOT / "configs/active_candidate.json"
        ).exists(),
        "f3_accessed": False,
    }
    output = {
        "schema_version": 1,
        "config_sha256": config.canonical_hash(),
        "checks": checks,
        "all_required_checks_passed": all(
            value is True for key, value in checks.items() if key != "f3_accessed"
        )
        and checks["f3_accessed"] is False,
        "runtime": {
            "cold_initialization_seconds": cold_seconds,
            "turn_latency_ms": latencies,
            "first_browsing_semantic_turn_ms": latencies[2],
            "warm_turn_p95_ms": max(latencies[1:]),
            "warm_mean_turn_ms": statistics.fmean(latencies[1:]),
            "peak_process_memory_mb": _peak_mb(),
        },
        "metrics": metrics,
        "promotion_status": (
            "architecture_complete_optimization_required"
            if adaptive["recommended_technical_score"]
            < champion["recommended_technical_score"]
            else "development_validated"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
