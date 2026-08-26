from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.policy.models import RuntimeConfig
from ghostlab.research.firewall import runtime_profile
from ghostlab.research.replay import ReplayEnvironment
from ghostlab.retrieval.constraint_gbdt import ConstraintGBDTFeatureStore
from ghostlab.retrieval.gbdt import LambdaMARTModel
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.runtime.agent import GhostLabRuntime
from scripts.run_gbdt_constraint_override_guard import build_guarded_agent
from scripts.run_gbdt_reranker import sha256_file, summarized_metrics
from starter.agent import Agent as StarterAgent

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = ROOT / "configs/techniques/guarded_constraint_gbdt_v1.json"
DEFAULT_CONFIG = ROOT / "configs/compiled_policy.json"
REPORT_PATH = ROOT / "artifacts/reports/guarded_compiled_parity_v1.json"


def _trace_hash(trace: list[dict[str, object]]) -> str:
    value = json.dumps(trace, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate guarded compiled parity")
    parser.add_argument("--require-default", action="store_true")
    args = parser.parse_args()
    config = RuntimeConfig.model_validate_json(
        CANDIDATE_CONFIG.read_text(encoding="utf-8")
    )
    base_asset = config.techniques.base_model_asset
    constraint_asset = config.techniques.constraint_model_asset
    assert base_asset is not None
    assert constraint_asset is not None
    catalog_path = ROOT / "data/catalog.jsonl"
    _, categories, products = catalog_index(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    features = ConstraintGBDTFeatureStore(catalog_path, quality=quality.quality)
    research, _ = build_guarded_agent(
        quality,
        features,
        LambdaMARTModel.load(ROOT / base_asset.path),
        LambdaMARTModel.load(ROOT / constraint_asset.path),
    )
    compiled = GhostLabRuntime(catalog_path, CANDIDATE_CONFIG)
    starter = StarterAgent(catalog_path) if args.require_default else None
    nested = json.loads(
        (ROOT / "configs/splits/nested_v1.json").read_text(encoding="utf-8")
    )
    sample_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = [
        sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in sample_ids
    ]
    traces: dict[str, list[dict[str, object]]] = {
        "research": [],
        "compiled": [],
        "starter": [],
    }
    mismatches: list[dict[str, object]] = []
    sessions: list[dict] = []
    for sample in sorted(samples, key=lambda item: str(item["sample_id"])):
        environment = ReplayEnvironment(sample, categories, products)
        observation = environment.observe()
        profile = runtime_profile(sample)
        research.reset(observation.session_id, profile)
        compiled.reset(observation.session_id, profile)
        if starter is not None:
            starter.reset(observation.session_id, profile)
        while not environment.done:
            common = {
                "sample_id": str(sample["sample_id"]),
                "turn": observation.turn,
                "user_message": observation.user_message,
            }
            responses = {
                "research": research.respond(
                    observation.session_id,
                    observation.user_message,
                    observation.turn,
                    observation.top_k,
                ),
                "compiled": compiled.respond(
                    observation.session_id,
                    observation.user_message,
                    observation.turn,
                    observation.top_k,
                ),
            }
            if starter is not None:
                responses["starter"] = starter.respond(
                    observation.session_id,
                    observation.user_message,
                    observation.turn,
                    observation.top_k,
                )
            for name, response in responses.items():
                traces[name].append(
                    {
                        **common,
                        "ask_attribute": response["ask_attribute"],
                        "recommendations": response["recommendations"],
                    }
                )
            expected = responses["research"]
            for name in ("compiled", "starter"):
                if name in responses and responses[name] != expected:
                    mismatches.append(
                        {
                            **common,
                            "variant": name,
                            "research": expected,
                            "observed": responses[name],
                        }
                    )
            next_observation = environment.step(responses["compiled"])
            if next_observation is not None:
                observation = next_observation
        sessions.append(environment.session_result())
    trace_hashes = {
        name: _trace_hash(trace)
        for name, trace in traces.items()
        if trace or name != "starter"
    }
    compiled_trace_matches = trace_hashes["compiled"] == trace_hashes["research"]
    starter_trace_matches = (
        trace_hashes.get("starter") == trace_hashes["research"]
        if args.require_default
        else None
    )
    default_config_matches = (
        RuntimeConfig.model_validate_json(
            DEFAULT_CONFIG.read_text(encoding="utf-8")
        ).techniques
        == config.techniques
    )
    if args.require_default and not default_config_matches:
        mismatches.append({"variant": "default_config", "observed": "not candidate"})
    passed = (
        not mismatches
        and compiled_trace_matches
        and (starter_trace_matches is not False)
    )
    report = {
        "schema_version": 1,
        "gate": "guarded_research_compiled_and_adapter_exact_parity",
        "passed": passed,
        "holdout_accessed": False,
        "sample_count": len(samples),
        "turn_count": len(traces["compiled"]),
        "response_fields_compared": [
            "message",
            "ask_attribute",
            "recommendations",
            "usage",
        ],
        "metrics": summarized_metrics(sessions),
        "research_vs_compiled_exact": compiled_trace_matches,
        "starter_adapter_required": args.require_default,
        "starter_vs_research_exact": starter_trace_matches,
        "default_config_matches_candidate_techniques": default_config_matches,
        "trace_sha256": trace_hashes,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "candidate_config": {
            "path": str(CANDIDATE_CONFIG.relative_to(ROOT)),
            "sha256": sha256_file(CANDIDATE_CONFIG),
            "canonical_sha256": config.canonical_hash(),
        },
        "model_assets": {
            "base": {
                "path": base_asset.path,
                "configured_sha256": base_asset.sha256,
                "actual_sha256": sha256_file(ROOT / base_asset.path),
            },
            "constraint": {
                "path": constraint_asset.path,
                "configured_sha256": constraint_asset.sha256,
                "actual_sha256": sha256_file(ROOT / constraint_asset.path),
            },
        },
        "compiled_runtime_has_experiment_trace": any(
            hasattr(compiled._primary, name)
            for name in ("routing_trace", "question_trace")
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("guarded compiled parity failed")


if __name__ == "__main__":
    main()
