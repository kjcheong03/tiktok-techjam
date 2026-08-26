from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from ghostlab.policy.models import RuntimeConfig
from ghostlab.retrieval.learned import (
    CandidateFeatureStore,
    LearnedLinearReranker,
    LinearRerankerModel,
)
from ghostlab.runtime.agent import GhostLabRuntime
from ghostlab.runtime.experimental import ExperimentalAgent

ROOT = Path(__file__).resolve().parents[1]


def research_agent(catalog_path: Path, config: RuntimeConfig) -> ExperimentalAgent:
    techniques = config.techniques
    if techniques.state_mode != "raw_history":
        raise ValueError("champion parity expects raw_history state")
    if techniques.question_policy != "sequence":
        raise ValueError("champion parity expects the sequence question policy")
    if techniques.reranker != "learned_linear":
        raise ValueError("champion parity expects the learned linear reranker")
    assert techniques.learned_weights is not None
    model = LinearRerankerModel(
        weights=techniques.learned_weights,
        l2=techniques.learned_l2,
        training_pairs=techniques.learned_training_pairs,
    )
    return ExperimentalAgent(
        catalog_path,
        state_variant="raw_history",
        question_variant="sequence",
        question_order=techniques.question_order,
        negative_evidence=techniques.negative_evidence,
        provenance=techniques.provenance,
        override_invalidation=techniques.override_invalidation,
        retrieval_route="keyword",
        sparse_weights=techniques.sparse_field_weights,
        quality_prior_weight=techniques.quality_prior_weight,
        learned_reranker=LearnedLinearReranker(
            CandidateFeatureStore(catalog_path), model
        ),
    )


def main() -> None:
    catalog_path = ROOT / "data/catalog.jsonl"
    policy_path = ROOT / "configs/compiled_policy.json"
    config = RuntimeConfig.model_validate_json(policy_path.read_text(encoding="utf-8"))
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    split = json.loads((ROOT / "configs/splits/adaptive_v1.json").read_text())
    allowed = set(split["sample_ids"])
    samples = [sample for sample in samples if sample["sample_id"] in allowed]
    catalog_ids, categories, products = catalog_index(catalog_path)
    research = evaluate(
        research_agent(catalog_path, config),
        samples,
        catalog_ids,
        categories,
        products,
    )
    compiled = evaluate(
        GhostLabRuntime(catalog_path, policy_path),
        samples,
        catalog_ids,
        categories,
        products,
    )
    parity_keys = (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "recommended_technical_score",
        "scenario_metrics",
        "sessions",
    )
    mismatches = {
        key: [research[key], compiled[key]]
        for key in parity_keys
        if research[key] != compiled[key]
    }
    report = {
        "phase": 22,
        "gate": "champion_compiled_research_parity",
        "split": "adaptive_v1",
        "holdout_accessed": False,
        "passed": not mismatches,
        "policy_id": config.policy_id,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "policy_canonical_sha256": config.canonical_hash(),
        "metrics": {key: compiled[key] for key in parity_keys if key != "sessions"},
        "mismatches": mismatches,
    }
    output = ROOT / "artifacts/reports/phase22_champion_compiled_parity.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit("compiled parity failed")


if __name__ == "__main__":
    main()
