from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from ghostlab.campaign.models import CampaignJob, CandidateSpec
from ghostlab.competition.contract import AgentProtocol
from ghostlab.research.technique_suite import (
    UnifiedTechniqueConfig,
    build_suite_agent,
)
from ghostlab.retrieval.residual import TECHNIQUE_ID
from ghostlab.training.protocol import FitReceipt, FitRequest
from ghostlab.training.residual import FoldDispatchAgent, ResidualFoldTrainer

ConfigMaterializer = Callable[[CandidateSpec], UnifiedTechniqueConfig]


class FoldSafeCandidateBuilder:
    """Build ordinary candidates directly and cross-fit trainable candidates.

    Search jobs are evaluated by outer-fold cross-fitting inside the frozen search
    partition. Confirmation jobs are fitted on the complete search partition and
    evaluated only on their declared confirmation fold.
    """

    def __init__(
        self,
        *,
        materialize: ConfigMaterializer,
        dataset_path: Path,
        catalog_path: Path,
        outer_folds: tuple[tuple[str, ...], ...],
        search_outer_folds: tuple[int, ...],
        confirmation_outer_folds: tuple[int, ...],
        campaign_id: str,
        artifact_root: Path = Path("artifacts/campaigns"),
    ) -> None:
        self.materialize = materialize
        self.dataset_path = dataset_path
        self.catalog_path = catalog_path
        self.outer_folds = outer_folds
        self.search_outer_folds = search_outer_folds
        self.confirmation_outer_folds = confirmation_outer_folds
        self.artifact_root = artifact_root / campaign_id / "fold_fits"
        self.trainer = ResidualFoldTrainer(dataset_path, catalog_path)
        self.fold_by_sample = {
            sample_id: index
            for index, fold in enumerate(outer_folds)
            for sample_id in fold
        }

    @staticmethod
    def _requires_residual(candidate: CandidateSpec) -> bool:
        return TECHNIQUE_ID in candidate.techniques

    def _fit_request(
        self,
        candidate: CandidateSpec,
        job: CampaignJob,
        validation_fold: int,
    ) -> FitRequest:
        if job.fidelity == "f2":
            if validation_fold not in self.confirmation_outer_folds:
                raise ValueError("F2 residual fit requires a confirmation fold")
            train_folds = self.search_outer_folds
        else:
            if validation_fold not in self.search_outer_folds:
                raise ValueError("search residual fit requires a search fold")
            train_folds = tuple(
                fold for fold in self.search_outer_folds if fold != validation_fold
            )
        train_ids = tuple(
            sorted(
                sample_id
                for fold in train_folds
                for sample_id in self.outer_folds[fold]
            )
        )
        return FitRequest(
            technique_id=TECHNIQUE_ID,
            outer_fold=validation_fold,
            inner_fold=0,
            train_sample_ids=train_ids,
            validation_sample_ids=tuple(sorted(self.outer_folds[validation_fold])),
            seed=job.seed,
        )

    def _asset_path(
        self, candidate: CandidateSpec, job: CampaignJob, fold: int
    ) -> Path:
        identity = hashlib.sha256(
            f"{candidate.canonical_hash()}:{job.fidelity}:{job.seed}:{fold}".encode()
        ).hexdigest()[:20]
        return self.artifact_root / candidate.canonical_hash() / f"{identity}.joblib"

    def _fit_or_load(
        self,
        candidate: CandidateSpec,
        job: CampaignJob,
        fold: int,
        config: UnifiedTechniqueConfig,
    ) -> FitReceipt:
        request = self._fit_request(candidate, job, fold)
        asset = self._asset_path(candidate, job, fold)
        receipt_path = asset.with_suffix(".receipt.json")
        if asset.is_file() and receipt_path.is_file():
            persisted = FitReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
            expected = FitReceipt.from_fit(request, asset)
            if persisted == expected:
                return persisted
        receipt = self.trainer.fit(request, asset, candidate_config=config)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        temporary.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
        return receipt

    def __call__(
        self,
        candidate: CandidateSpec,
        job: CampaignJob,
        sample_ids: tuple[str, ...],
    ) -> tuple[AgentProtocol, tuple[FitReceipt, ...]]:
        config = self.materialize(candidate)
        if not self._requires_residual(candidate):
            return build_suite_agent(config, self.catalog_path), ()

        validation_folds = tuple(
            sorted({self.fold_by_sample[sample_id] for sample_id in sample_ids})
        )
        receipts: list[FitReceipt] = []
        agents_by_sample: dict[str, AgentProtocol] = {}
        for fold in validation_folds:
            receipt = self._fit_or_load(candidate, job, fold, config)
            receipts.append(receipt)
            fitted = config.model_copy(
                update={"residual_model_asset": receipt.asset_path}
            )
            agent = build_suite_agent(fitted, self.catalog_path)
            for sample_id in sample_ids:
                if self.fold_by_sample[sample_id] == fold:
                    agents_by_sample[sample_id] = agent
        return FoldDispatchAgent(agents_by_sample), tuple(receipts)
