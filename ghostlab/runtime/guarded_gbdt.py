from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import cast

from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.models import ModelAssetConfig, TechniqueConfig
from ghostlab.retrieval.constraint_gbdt import (
    CONSTRAINT_METADATA_FEATURES,
    OVERRIDE_INVALIDATION_REASONS,
    ConstraintAwareLambdaMARTReranker,
    ConstraintGBDTFeatureStore,
)
from ghostlab.retrieval.gbdt import (
    METADATA_FEATURES,
    LambdaMARTModel,
    LambdaMARTReranker,
)
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.memory import ConversationState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_asset(project_root: Path, asset: ModelAssetConfig) -> Path:
    path = (project_root / asset.path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("model asset resolved outside the project root") from error
    return path


def has_observable_override(state: ConversationState) -> bool:
    return any(
        not item.active and item.invalidated_reason in OVERRIDE_INVALIDATION_REASONS
        for item in state.values
    )


class GuardedGBDTModels:
    """Lazy, content-addressed loader for the two offline runtime models."""

    def __init__(
        self,
        project_root: Path,
        base_asset: ModelAssetConfig,
        constraint_asset: ModelAssetConfig,
        features: ConstraintGBDTFeatureStore,
    ) -> None:
        self._base_path = _resolved_asset(project_root, base_asset)
        self._constraint_path = _resolved_asset(project_root, constraint_asset)
        self._base_hash = base_asset.sha256
        self._constraint_hash = constraint_asset.sha256
        self._features = features
        self._lock = Lock()
        self._base: LambdaMARTReranker | None = None
        self._constraint: ConstraintAwareLambdaMARTReranker | None = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._base is not None and self._constraint is not None:
            return
        with self._lock:
            if self._base is not None and self._constraint is not None:
                return
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            try:
                for path, expected in (
                    (self._base_path, self._base_hash),
                    (self._constraint_path, self._constraint_hash),
                ):
                    if not path.is_file():
                        raise FileNotFoundError(f"missing model asset: {path.name}")
                    actual = _sha256(path)
                    if actual != expected:
                        raise ValueError(f"model asset hash mismatch: {path.name}")
                base_model = LambdaMARTModel.load(self._base_path)
                constraint_model = LambdaMARTModel.load(self._constraint_path)
                if base_model.feature_names != METADATA_FEATURES:
                    raise ValueError("base model feature schema mismatch")
                if constraint_model.feature_names != CONSTRAINT_METADATA_FEATURES:
                    raise ValueError("constraint model feature schema mismatch")
                if not 0 < base_model.best_iteration <= len(base_model.trees):
                    raise ValueError("base model tree count is invalid")
                if (
                    not 0
                    < constraint_model.best_iteration
                    <= len(constraint_model.trees)
                ):
                    raise ValueError("constraint model tree count is invalid")
                self._base = LambdaMARTReranker(self._features, base_model)
                self._constraint = ConstraintAwareLambdaMARTReranker(
                    self._features, constraint_model
                )
            except Exception as error:
                self._load_error = str(error)
                raise

    def rerank(
        self,
        query: str,
        ranking: list[str],
        *,
        state: ConversationState,
        turn: int,
        retrieval_scores: list[float],
        rerank_k: int,
    ) -> list[str]:
        self._load()
        assert self._base is not None
        assert self._constraint is not None
        if has_observable_override(state):
            return self._base.rerank(query, ranking, rerank_k=rerank_k)
        return self._constraint.rerank_with_context(
            query,
            ranking,
            state=state,
            turn=turn,
            retrieval_scores=retrieval_scores,
            rerank_k=rerank_k,
        )


@dataclass
class _RuntimeSession:
    state: ConversationState
    lock: Lock


class CompiledGuardedGBDTAgent:
    """Production runtime for the validated guarded constraint GBDT policy."""

    def __init__(
        self,
        catalog_path: str | Path,
        catalog_ids: set[str],
        techniques: TechniqueConfig,
        project_root: Path,
    ) -> None:
        if techniques.reranker != "guarded_constraint_gbdt":
            raise ValueError("guarded runtime received a different reranker")
        assert techniques.sparse_field_weights is not None
        assert techniques.base_model_asset is not None
        assert techniques.constraint_model_asset is not None
        self.catalog_ids = catalog_ids
        self.techniques = techniques
        self._catalog_path = catalog_path
        self._sparse_local = local()
        self._sparse_local.index = SparseIndex(catalog_path)
        self.quality = CatalogQualityReranker(catalog_path)
        features = ConstraintGBDTFeatureStore(
            catalog_path, quality=self.quality.quality
        )
        self.models = GuardedGBDTModels(
            project_root,
            techniques.base_model_asset,
            techniques.constraint_model_asset,
            features,
        )
        self._sessions: dict[str, _RuntimeSession] = {}
        self._sessions_lock = Lock()

    def _sparse(self) -> SparseIndex:
        sparse = getattr(self._sparse_local, "index", None)
        if sparse is None:
            sparse = SparseIndex(self._catalog_path)
            self._sparse_local.index = sparse
        return cast(SparseIndex, sparse)

    def reset(self, session_id: str, user_profile: dict) -> None:
        session = _RuntimeSession(
            ConversationState(
                session_id,
                user_profile,
                multi_value=False,
                negative_evidence=self.techniques.negative_evidence,
                provenance_enabled=self.techniques.provenance,
                override_invalidation=self.techniques.override_invalidation,
            ),
            Lock(),
        )
        with self._sessions_lock:
            self._sessions[session_id] = session

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        with self._sessions_lock:
            session = self._sessions[session_id]
        with session.lock:
            state = session.state
            state.observe(user_message, turn)
            query = ". ".join(state.messages)
            order = self.techniques.question_order
            question = order[turn - 1] if turn <= len(order) else None
            if question is not None and (
                not state.asked_attributes or state.asked_attributes[-1] != question
            ):
                state.asked_attributes.append(question)
            state.last_asked_attribute = question
            assert self.techniques.sparse_field_weights is not None
            scored = self._sparse().search(
                query,
                self.techniques.retrieval_k,
                self.techniques.sparse_field_weights,
            )
            ranking = [item.parent_asin for item in scored.items]
            ranking = self.quality.rerank(
                ranking,
                weight=self.techniques.quality_prior_weight,
                rerank_k=self.techniques.rerank_k,
            )
            ranking = self.models.rerank(
                query,
                ranking,
                state=state,
                turn=turn,
                retrieval_scores=[
                    float(item.raw_score)
                    for item in scored.items
                    if item.raw_score is not None
                ],
                rerank_k=self.techniques.rerank_k,
            )
            ask_attribute = cast(AskAttribute | None, question)
            message = (
                "Here are the closest matches based on what you have shared."
                if ask_attribute is None
                else f"Do you have a preference for {ask_attribute.replace('_', ' ')}?"
            )
            return normalize_response(
                {
                    "message": message,
                    "ask_attribute": ask_attribute,
                    "recommendations": ranking,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                self.catalog_ids,
                top_k,
            )
