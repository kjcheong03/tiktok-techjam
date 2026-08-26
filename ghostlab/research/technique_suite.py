from __future__ import annotations

import json
from itertools import product
from pathlib import Path, PurePath
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ghostlab.competition.contract import AgentProtocol, AskAttribute

PROJECT_ROOT = Path(__file__).resolve().parents[2]

StateVariant = Literal["current", "raw_history", "single", "multi", "compressed"]
QueryVariant = Literal[
    "raw_history",
    "structured_active",
    "category_constraints",
    "raw_plus_active",
    "compressed_raw",
    "negation_safe_hybrid",
]
QuestionVariant = Literal[
    "none",
    "fixed",
    "sequence",
    "missing_priority",
    "feature_first",
    "uncertainty",
    "other_always",
    "adaptive",
    "learned",
]
RetrievalRoute = Literal["keyword", "dense", "rrf", "weighted", "sparse_first_union"]
DenseBackend = Literal["off", "minilm_control", "e5_small_v2"]
Reranker = Literal["none", "linear", "metadata_gbdt"]


class UnifiedTechniqueConfig(BaseModel):
    """A versioned, research-only composition without changing submission config."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    engine: Literal["compiled", "experimental"] = "experimental"
    compiled_config_path: str | None = None

    state_variant: StateVariant = "raw_history"
    query_variant: QueryVariant | None = None
    question_variant: QuestionVariant = "sequence"
    question_order: tuple[AskAttribute, ...] = ()
    learned_question_asset: str | None = None

    retrieval_route: RetrievalRoute = "keyword"
    dense_backend: DenseBackend = "off"
    dense_model_path: str | None = None
    sparse_weights: tuple[float, float, float, float, float, float] | None = None
    sparse_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.25, ge=0.0, le=1.0)

    negative_evidence: bool = True
    provenance: bool = True
    override_invalidation: bool = True
    structured_filter: bool = False
    profile_prior_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_prior_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    reranker: Reranker = "none"
    reranker_model_asset: str | None = None
    rerank_k: int = Field(default=50, ge=10, le=400)
    cross_encoder_enabled: bool = False
    cross_encoder_model_path: str | None = None
    cross_encoder_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_encoder_rerank_k: int = Field(default=20, ge=1, le=200)

    @field_validator(
        "compiled_config_path",
        "learned_question_asset",
        "dense_model_path",
        "reranker_model_asset",
        "cross_encoder_model_path",
    )
    @classmethod
    def safe_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("asset/config paths must stay inside the project")
        return value

    @model_validator(mode="after")
    def compatible(self) -> UnifiedTechniqueConfig:
        if self.engine == "compiled":
            if self.compiled_config_path is None:
                raise ValueError("compiled engine requires compiled_config_path")
            return self
        if self.compiled_config_path is not None:
            raise ValueError("compiled_config_path is only valid for compiled engine")
        needs_dense = self.retrieval_route != "keyword"
        if needs_dense != (self.dense_backend != "off"):
            raise ValueError(
                "non-keyword retrieval requires a dense backend, and keyword-only "
                "retrieval must keep it off"
            )
        if self.dense_backend != "off" and self.dense_model_path is None:
            raise ValueError("dense backend requires dense_model_path")
        if self.dense_backend == "off" and self.dense_model_path is not None:
            raise ValueError("dense_model_path requires a dense backend")
        if (
            self.retrieval_route == "weighted"
            and abs(self.sparse_weight + self.dense_weight - 1.0) > 1e-9
        ):
            raise ValueError("weighted fusion weights must sum to one")
        if self.query_variant is not None and self.state_variant in {
            "current",
            "single",
        }:
            raise ValueError("structured query variants require conversation memory")
        if self.question_variant == "learned" and self.learned_question_asset is None:
            raise ValueError("learned questions require learned_question_asset")
        if (
            self.question_variant != "learned"
            and self.learned_question_asset is not None
        ):
            raise ValueError("learned_question_asset requires learned questions")
        if self.question_variant == "sequence" and not self.question_order:
            raise ValueError("sequence question policy requires question_order")
        if self.reranker == "metadata_gbdt" and self.reranker_model_asset is None:
            raise ValueError("metadata GBDT requires reranker_model_asset")
        if self.reranker != "metadata_gbdt" and self.reranker_model_asset is not None:
            raise ValueError("reranker_model_asset requires metadata GBDT")
        if self.cross_encoder_enabled:
            if self.cross_encoder_model_path is None:
                raise ValueError("cross-encoder requires cross_encoder_model_path")
            if self.cross_encoder_weight <= 0.0:
                raise ValueError("enabled cross-encoder requires a positive weight")
        elif self.cross_encoder_model_path is not None or self.cross_encoder_weight:
            raise ValueError("cross-encoder fields require cross_encoder_enabled")
        return self


def load_suite_config(path: str | Path) -> UnifiedTechniqueConfig:
    return UnifiedTechniqueConfig.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def valid_combinations(
    base: UnifiedTechniqueConfig,
    dimensions: dict[str, list[object]],
) -> tuple[list[UnifiedTechniqueConfig], list[str]]:
    """Enumerate a finite declared space and retain only schema-valid combinations."""

    unknown = set(dimensions) - set(UnifiedTechniqueConfig.model_fields)
    if unknown:
        raise ValueError(f"unknown combination dimensions: {sorted(unknown)}")
    names = sorted(dimensions)
    accepted: list[UnifiedTechniqueConfig] = []
    rejected: list[str] = []
    base_value = base.model_dump(mode="json")
    for values in product(*(dimensions[name] for name in names)):
        candidate = {**base_value, **dict(zip(names, values, strict=True))}
        try:
            accepted.append(UnifiedTechniqueConfig.model_validate(candidate))
        except ValueError as error:
            rejected.append(str(error))
    return accepted, rejected


def _project_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("path resolved outside project root") from error
    return path


def _load_question_model(path: Path):
    from ghostlab.policy.learned_questions import LinearActionValueModel

    value = json.loads(path.read_text(encoding="utf-8"))
    weights = {
        (None if action == "stop" else cast(AskAttribute, action)): tuple(items)
        for action, items in value["action_weights"].items()
    }
    return LinearActionValueModel(
        feature_names=tuple(value["feature_names"]),
        action_weights=weights,
        l2=float(value["l2"]),
        training_states=int(value["training_states"]),
    )


def build_suite_agent(
    config: UnifiedTechniqueConfig, catalog_path: str | Path
) -> AgentProtocol:
    """Build only selected components; optional imports occur inside this function."""

    if config.engine == "compiled":
        from ghostlab.runtime.agent import GhostLabRuntime

        assert config.compiled_config_path is not None
        return GhostLabRuntime(catalog_path, _project_path(config.compiled_config_path))

    from ghostlab.retrieval.cross_encoder import CrossEncoderReranker
    from ghostlab.runtime.unified_experimental import ExperimentalAgent

    dense = None
    if config.dense_backend != "off":
        from ghostlab.retrieval.dense import (
            E5_SMALL_V2,
            MINILM_CONTROL,
            DenseIndex,
        )

        assert config.dense_model_path is not None
        spec = E5_SMALL_V2 if config.dense_backend == "e5_small_v2" else MINILM_CONTROL
        dense = DenseIndex(
            catalog_path,
            spec,
            model_path=_project_path(config.dense_model_path),
            local_files_only=True,
        )

    learned_question_model = None
    if config.learned_question_asset is not None:
        learned_question_model = _load_question_model(
            _project_path(config.learned_question_asset)
        )

    learned_reranker = None
    if config.reranker == "metadata_gbdt":
        from ghostlab.retrieval.gbdt import (
            GBDTFeatureStore,
            LambdaMARTModel,
            LambdaMARTReranker,
        )
        from ghostlab.retrieval.quality import CatalogQualityReranker

        assert config.reranker_model_asset is not None
        quality = CatalogQualityReranker(catalog_path)
        features = GBDTFeatureStore(catalog_path, quality=quality.quality)
        learned_reranker = LambdaMARTReranker(
            features,
            LambdaMARTModel.load(_project_path(config.reranker_model_asset)),
        )

    cross_encoder = None
    if config.cross_encoder_enabled:
        assert config.cross_encoder_model_path is not None
        cross_encoder = CrossEncoderReranker(
            catalog_path,
            model_name=str(_project_path(config.cross_encoder_model_path)),
            revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
            cache_folder=_project_path("artifacts/cache/cross_encoder"),
            local_files_only=True,
        )

    return ExperimentalAgent(
        catalog_path,
        state_variant=config.state_variant,
        query_variant=config.query_variant,
        question_variant=config.question_variant,
        question_order=tuple(config.question_order),
        learned_question_model=learned_question_model,
        retrieval_route=config.retrieval_route,
        dense_retriever=dense,
        sparse_weight=config.sparse_weight,
        dense_weight=config.dense_weight,
        sparse_weights=config.sparse_weights,
        negative_evidence=config.negative_evidence,
        provenance=config.provenance,
        override_invalidation=config.override_invalidation,
        structured_filter=config.structured_filter,
        profile_prior_weight=config.profile_prior_weight,
        quality_prior_weight=config.quality_prior_weight,
        reranker="linear" if config.reranker == "linear" else "none",
        learned_reranker=learned_reranker,
        learned_rerank_k=config.rerank_k,
        cross_encoder_reranker=cross_encoder,
        cross_encoder_weight=config.cross_encoder_weight,
        cross_encoder_rerank_k=config.cross_encoder_rerank_k,
    )
