from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.evaluation.gbdt_audit import (
    evidence_gates,
    metric_parity,
    packaging_gate,
    session_parity,
)
from ghostlab.retrieval.gbdt import (
    FEATURE_SETS,
    GBDTFeatureStore,
    LambdaMARTReranker,
    fit_lambdamart,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from scripts.measure_gbdt_runtime import TimedAgent
from scripts.run_gbdt_reranker import (
    ROOT,
    SEED,
    build_agent,
    collect_groups,
    ranking_dataset,
    variant_config,
)

AMENDMENT_PATH = ROOT / "configs/experiments/gbdt_reranker_v1_amendment_1.json"
MANIFEST_PATH = ROOT / "configs/experiments/gbdt_reranker_v1.json"
OOF_REPORT_PATH = ROOT / "artifacts/reports/gbdt_reranker_v1.json"
MODEL_PATH = ROOT / "artifacts/models/gbdt_reranker_v2_round56.json"
AUDIT_REPORT_PATH = ROOT / "artifacts/reports/gbdt_deployment_audit_v1.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    started = time.perf_counter()
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    oof_report = json.loads(OOF_REPORT_PATH.read_text(encoding="utf-8"))
    if amendment["holdout_accessed"] is not False:
        raise RuntimeError("audit amendment violates the holdout firewall")
    selected_id = str(amendment["selected_candidate_id"])
    frozen_rounds = int(amendment["frozen_deployable_round_rule"]["rounds"])
    if frozen_rounds != 56:
        raise RuntimeError("audit must use the precommitted 56-round rule")
    observed_rounds = [
        int(fold["inner_selected_rounds"])
        for fold in oof_report["variants"][selected_id]["folds"]
    ]
    if observed_rounds != amendment["outer_selected_rounds_in_frozen_fold_order"]:
        raise RuntimeError("amended round source does not match frozen OOF evidence")
    immutable = amendment["immutable_outer_oof"]
    observed_oof = oof_report["variants"][selected_id]["oof_metrics"]
    for key, expected in immutable.items():
        if observed_oof[key] != expected:
            raise RuntimeError(f"immutable OOF metric changed: {key}")

    nested = json.loads((ROOT / "configs/splits/nested_v1.json").read_text())
    adaptive_ids = {str(value) for value in nested["adaptive_sample_ids"]}
    samples = {
        str(sample["sample_id"]): sample
        for sample in load_jsonl(ROOT / "data/public_set.jsonl")
        if str(sample["sample_id"]) in adaptive_ids
    }
    catalog_path = ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    sparse = SparseIndex(catalog_path)
    quality = CatalogQualityReranker(catalog_path)
    feature_store = GBDTFeatureStore(catalog_path, quality=quality.quality)
    groups, collection = collect_groups(
        samples, categories, products, sparse, quality, feature_store
    )
    config = variant_config(manifest, selected_id)
    feature_names = FEATURE_SETS[str(config["feature_set"])]
    dataset = ranking_dataset(groups, adaptive_ids, feature_names)

    def refit():
        return fit_lambdamart(
            *dataset,
            candidate_id=selected_id,
            feature_names=feature_names,
            max_depth=int(config["max_depth"]),
            num_leaves=int(config["num_leaves"]),
            learning_rate=float(config["learning_rate"]),
            max_rounds=frozen_rounds,
            early_stopping_rounds=int(config["early_stopping_rounds"]),
            validation=None,
            seed=SEED,
        )

    first_model = refit()
    second_model = refit()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    first_model.save(MODEL_PATH)
    with tempfile.TemporaryDirectory() as directory:
        second_path = Path(directory) / "second_model.json"
        second_model.save(second_path)
        first_bytes = MODEL_PATH.read_bytes()
        second_bytes = second_path.read_bytes()

    ordered_samples = [samples[sample_id] for sample_id in sorted(samples)]
    first_agent = TimedAgent(
        build_agent(quality, LambdaMARTReranker(feature_store, first_model))
    )
    first_result = evaluate(
        first_agent, ordered_samples, catalog_ids, categories, products
    )
    second_agent = TimedAgent(
        build_agent(quality, LambdaMARTReranker(feature_store, second_model))
    )
    second_result = evaluate(
        second_agent, ordered_samples, catalog_ids, categories, products
    )

    runtime_process = subprocess.run(
        [sys.executable, "-m", "scripts.measure_gbdt_runtime", str(MODEL_PATH)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(runtime_process.stdout.strip().splitlines()[-1])
    evidence = evidence_gates(
        oof_report, candidate_id=selected_id, scenario_delta_floor=-0.005
    )
    session_check = session_parity(first_result["sessions"], second_result["sessions"])
    deterministic = {
        "model_bytes_identical": first_bytes == second_bytes,
        "first_model_sha256": sha256_bytes(first_bytes),
        "second_model_sha256": sha256_bytes(second_bytes),
        "session_outcomes": session_check,
        "first_response_failure_count": first_agent.failure_count,
        "second_response_failure_count": second_agent.failure_count,
    }
    deterministic["passed"] = bool(
        deterministic["model_bytes_identical"]
        and session_check["passed"]
        and first_agent.failure_count == 0
        and second_agent.failure_count == 0
    )
    deployment_metrics = {
        key: first_result[key]
        for key in (
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "recommended_technical_score",
        )
    }
    parity = metric_parity(deployment_metrics, runtime["metrics"])
    packaging = packaging_gate(runtime, amendment["executable_gates"]["packaging"])
    gates = {
        **evidence,
        "determinism": deterministic,
        "parity": parity,
        "packaging": packaging,
    }
    all_passed = all(bool(gate["passed"]) for gate in gates.values())
    report = {
        "schema_version": 1,
        "audit_id": "gbdt_deployment_audit_v1",
        "amendment_path": str(AMENDMENT_PATH.relative_to(ROOT)),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "amendment_commit": "e001d872011e7104b7648d3c3db10d18dedc468a",
        "source_oof_report": str(OOF_REPORT_PATH.relative_to(ROOT)),
        "source_oof_report_sha256": sha256_file(OOF_REPORT_PATH),
        "provenance": {
            "data_sha256": sha256_file(ROOT / "data/public_set.jsonl"),
            "catalog_sha256": sha256_file(ROOT / "data/catalog.jsonl"),
            "split_sha256": sha256_file(ROOT / "configs/splits/nested_v1.json"),
            "lock_sha256": sha256_file(ROOT / "uv.lock"),
            "code_hashes": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (
                    ROOT / "ghostlab/evaluation/gbdt_audit.py",
                    ROOT / "ghostlab/retrieval/gbdt.py",
                    ROOT / "scripts/measure_gbdt_runtime.py",
                    ROOT / "scripts/resolve_gbdt_deployment_audit.py",
                )
            },
        },
        "holdout_accessed": False,
        "selected_candidate_id": selected_id,
        "deployable_round_rule": amendment["frozen_deployable_round_rule"],
        "alternative_round_aggregations_evaluated": False,
        "collection": collection,
        "immutable_oof_family_evidence": {
            "metrics": immutable,
            "paired_vs_linear": oof_report["variants"][selected_id][
                "paired_vs_two_feature_linear"
            ],
        },
        "deployable_refit": {
            "evidence_label": "all-development refit",
            "training_sample_count": len(adaptive_ids),
            "training_groups": first_model.training_groups,
            "training_rows": first_model.training_rows,
            "rounds": first_model.best_iteration,
            "model_path": str(MODEL_PATH.relative_to(ROOT)),
            "model_sha256": sha256_file(MODEL_PATH),
            "model_asset_bytes": MODEL_PATH.stat().st_size,
            "feature_names": list(first_model.feature_names),
            "feature_importance_split_count": dict(
                sorted(
                    first_model.split_importance().items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "metrics": {
                **deployment_metrics,
                "scenario_metrics": first_result["scenario_metrics"],
            },
            "response_instrumentation": {
                "response_calls": first_agent.response_calls,
                "failure_count": first_agent.failure_count,
                "failure_counts": {
                    "reset_exceptions": first_agent.reset_exception_count,
                    "response_exceptions": first_agent.response_exception_count,
                    "invalid_responses": first_agent.invalid_response_count,
                },
            },
        },
        "isolated_runtime": runtime,
        "gates": gates,
        "all_gates_passed": all_passed,
        "decision": ("INTEGRATION_READY" if all_passed else "DEPLOYMENT_BLOCKED"),
        "decision_rationale": (
            "The immutable OOF family promotion remains valid and the precommitted median-round deployable refit passes every executable audit gate."
            if all_passed
            else "The immutable OOF family evidence is preserved, but the median-round deployable refit failed at least one executable audit gate."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "deployable_refit": report["deployable_refit"],
                "isolated_runtime": runtime,
                "gates": gates,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
