from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.evaluator import OfflineCampaignEvaluator
from ghostlab.campaign.models import CampaignManifest
from ghostlab.campaign.orchestrator import (
    AutonomousCampaign,
    CampaignOptions,
    FrozenInputs,
    verify_frozen_inputs,
)
from ghostlab.research.technique_suite import build_suite_agent
from ghostlab.retrieval.residual import TECHNIQUE_ID as RESIDUAL_TECHNIQUE_ID
from ghostlab.training.campaign import FoldSafeCandidateBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen, resumable F0/F1/F2 autonomous proposal campaign"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--technique-catalog",
        type=Path,
        default=Path("configs/techniques/catalog_v2.json"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--product-catalog", type=Path, default=Path("data/catalog.jsonl")
    )
    parser.add_argument(
        "--adaptive-split",
        type=Path,
        default=Path("configs/splits/adaptive_v1.json"),
    )
    parser.add_argument(
        "--nested-split",
        type=Path,
        default=Path("configs/splits/nested_v1.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/campaign/autonomous_checkpoint.json"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("artifacts/reports/autonomous_campaign.json"),
    )
    parser.add_argument("--f1-candidates", type=int, default=24)
    parser.add_argument("--f2-candidates", type=int, default=6)
    parser.add_argument("--hpo-trials-per-structure", type=int, default=8)
    parser.add_argument("--higher-order-rounds", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    manifest_path = _project_path(args.manifest)
    technique_catalog_path = _project_path(args.technique_catalog)
    dataset_path = _project_path(args.dataset)
    product_catalog_path = _project_path(args.product_catalog)
    adaptive_path = _project_path(args.adaptive_split)
    nested_path = _project_path(args.nested_split)
    manifest = CampaignManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    catalog = load_catalog(technique_catalog_path)
    verified = verify_frozen_inputs(
        manifest,
        catalog,
        FrozenInputs(
            catalog_path=technique_catalog_path,
            dataset_path=dataset_path,
            adaptive_split_path=adaptive_path,
            nested_split_path=nested_path,
        ),
    )
    adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    adaptive_ids = tuple(str(value) for value in adaptive["sample_ids"])
    outer_folds = tuple(
        tuple(str(value) for value in fold) for fold in nested["outer_folds"]
    )

    campaign: AutonomousCampaign
    fitted_builder: FoldSafeCandidateBuilder

    def evaluator_factory(candidates):  # type: ignore[no-untyped-def]
        return OfflineCampaignEvaluator(
            candidates=candidates,
            builder=lambda candidate: build_suite_agent(
                campaign.materialize(candidate), product_catalog_path
            ),
            fitted_builder=fitted_builder,
            dataset_path=dataset_path,
            catalog_path=product_catalog_path,
            adaptive_sample_ids=adaptive_ids,
            outer_folds=outer_folds,
            budgets=manifest.fidelity_budgets,
            search_outer_folds=manifest.search_outer_folds,
            confirmation_outer_folds=manifest.confirmation_outer_folds,
        )

    campaign = AutonomousCampaign(
        manifest=manifest,
        catalog=catalog,
        registry=default_binding_registry(),
        evaluator_factory=evaluator_factory,
        checkpoint_path=_project_path(args.checkpoint),
        evidence_path=_project_path(args.evidence),
        outer_folds=outer_folds,
        project_root=PROJECT_ROOT,
        search_space_path=Path(
            manifest.search_space_path or "configs/search/wave2_weight_space_v1.json"
        ),
        verified_input_hashes=verified,
        fit_capable_techniques=frozenset({RESIDUAL_TECHNIQUE_ID}),
        options=CampaignOptions(
            f1_candidates=args.f1_candidates,
            f2_candidates=args.f2_candidates,
            hpo_trials_per_structure=args.hpo_trials_per_structure,
            higher_order_rounds=args.higher_order_rounds,
            bootstrap_resamples=args.bootstrap_resamples,
        ),
    )
    fitted_builder = FoldSafeCandidateBuilder(
        materialize=campaign.materialize,
        dataset_path=dataset_path,
        catalog_path=product_catalog_path,
        outer_folds=outer_folds,
        search_outer_folds=manifest.search_outer_folds,
        confirmation_outer_folds=manifest.confirmation_outer_folds,
        campaign_id=manifest.campaign_id,
    )
    result = campaign.run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
