from __future__ import annotations

import json
from itertools import product
from pathlib import Path, PurePath
from typing import Literal, Protocol, cast

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
RetrievalRoute = Literal[
    "keyword",
    "dense",
    "rrf",
    "weighted",
    "sparse_first_union",
    "learned_sparse_union",
    "late_interaction_union",
]
DenseBackend = Literal["off", "minilm_control", "e5_small_v2"]
Reranker = Literal["none", "linear", "metadata_gbdt"]
QueryExpansion = Literal["off", "prf"]
Diversification = Literal["off", "facet_mmr"]


class CandidateRetriever(Protocol):
    def search(self, query: str, limit: int) -> object: ...


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
    semantic_rescue_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    learned_sparse_asset: str | None = None
    late_interaction_asset: str | None = None

    query_expansion: QueryExpansion = "off"
    expansion_feedback_k: int = Field(default=5, ge=2, le=50)
    expansion_min_support: float = Field(default=0.4, gt=0.0, le=1.0)
    expansion_max_terms: int = Field(default=4, ge=0, le=20)
    expansion_max_added_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

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
    diversification: Diversification = "off"
    diversification_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    diversification_rerank_k: int = Field(default=30, ge=10, le=200)
    diversification_output_k: int = Field(default=10, ge=1, le=50)
    diversification_max_turn: int = Field(default=2, ge=1, le=9)
    diversification_max_constraints: int = Field(default=1, ge=0, le=10)

    @field_validator(
        "compiled_config_path",
        "learned_question_asset",
        "dense_model_path",
        "reranker_model_asset",
        "cross_encoder_model_path",
        "learned_sparse_asset",
        "late_interaction_asset",
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
            if (
                self.learned_sparse_asset is not None
                or self.late_interaction_asset is not None
                or self.query_expansion != "off"
                or self.diversification != "off"
            ):
                raise ValueError(
                    "Wave 2 retrieval switches require the experimental engine"
                )
            return self
        if self.compiled_config_path is not None:
            raise ValueError("compiled_config_path is only valid for compiled engine")
        dense_routes = {"dense", "rrf", "weighted", "sparse_first_union"}
        needs_dense = self.retrieval_route in dense_routes
        if needs_dense != (self.dense_backend != "off"):
            raise ValueError(
                "non-keyword retrieval requires a dense backend, and keyword-only "
                "retrieval must keep it off"
            )
        if self.dense_backend != "off" and self.dense_model_path is None:
            raise ValueError("dense backend requires dense_model_path")
        if self.dense_backend == "off" and self.dense_model_path is not None:
            raise ValueError("dense_model_path requires a dense backend")
        if self.retrieval_route == "learned_sparse_union":
            if self.learned_sparse_asset is None:
                raise ValueError("learned sparse union requires learned_sparse_asset")
        elif self.learned_sparse_asset is not None:
            raise ValueError("learned_sparse_asset requires learned sparse union")
        if self.retrieval_route == "late_interaction_union":
            if self.late_interaction_asset is None:
                raise ValueError(
                    "late interaction union requires late_interaction_asset"
                )
        elif self.late_interaction_asset is not None:
            raise ValueError("late_interaction_asset requires late interaction union")
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
        if self.diversification_output_k > self.diversification_rerank_k:
            raise ValueError("diversification output depth cannot exceed rerank depth")
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

    semantic_rescue: CandidateRetriever | None = None
    if config.learned_sparse_asset is not None:
        from ghostlab.retrieval.learned_sparse import (
            InvertedSparseIndex,
            LearnedSparseAsset,
            LearnedSparseRetriever,
            SpladeEncoder,
            sha256_file,
        )

        learned_manifest = LearnedSparseAsset.load(
            _project_path(config.learned_sparse_asset)
        )
        model_path, index_path = learned_manifest.require_available(PROJECT_ROOT)
        catalog_hash = sha256_file(catalog_path)
        if learned_manifest.catalog_sha256 != catalog_hash:
            raise ValueError("learned-sparse asset was built for another catalog")
        semantic_rescue = LearnedSparseRetriever(
            SpladeEncoder(model_path, max_terms=learned_manifest.max_terms),
            InvertedSparseIndex.load(index_path),
        )
    if config.late_interaction_asset is not None:
        from ghostlab.retrieval.late_interaction import (
            LateInteractionRetriever,
            TokenEmbeddingStore,
            TransformerTokenEncoder,
            load_feasibility_manifest,
            sha256_file,
        )

        manifest_path = _project_path(config.late_interaction_asset)
        late_manifest = load_feasibility_manifest(manifest_path)
        if late_manifest.get("availability") != "available":
            raise RuntimeError(
                "late-interaction technique is unavailable: "
                + str(
                    late_manifest.get("unavailable_reason", "feasibility gate failed")
                )
            )
        model_path = _project_path(str(late_manifest["model_path"]))
        index_path = _project_path(str(late_manifest["index_path"]))
        if not model_path.is_dir() or not index_path.is_file():
            raise FileNotFoundError("late-interaction model or index is missing")
        if sha256_file(index_path) != late_manifest.get("index_sha256"):
            raise ValueError("late-interaction index checksum mismatch")
        if sha256_file(catalog_path) != late_manifest.get("catalog_sha256"):
            raise ValueError("late-interaction asset was built for another catalog")
        late_max_length = late_manifest.get("max_length", 128)
        if not isinstance(late_max_length, int):
            raise TypeError("late-interaction max_length must be an integer")
        semantic_rescue = LateInteractionRetriever(
            TransformerTokenEncoder(
                model_path,
                max_length=late_max_length,
            ),
            TokenEmbeddingStore.load(index_path),
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

    query_expander = None
    if config.query_expansion == "prf":
        from ghostlab.retrieval.pseudo_relevance import (
            CatalogPseudoRelevanceFeedback,
        )

        query_expander = CatalogPseudoRelevanceFeedback(
            catalog_path,
            feedback_k=config.expansion_feedback_k,
            minimum_support=config.expansion_min_support,
            max_terms=config.expansion_max_terms,
            max_added_ratio=config.expansion_max_added_ratio,
        )

    diversifier = None
    if config.diversification == "facet_mmr":
        from ghostlab.retrieval.diversify import FacetMMRDiversifier

        diversifier = FacetMMRDiversifier(
            catalog_path,
            relevance_weight=config.diversification_weight,
            rerank_k=config.diversification_rerank_k,
            output_k=config.diversification_output_k,
            max_turn=config.diversification_max_turn,
            max_active_constraints=config.diversification_max_constraints,
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
        semantic_rescue_retriever=semantic_rescue,
        semantic_rescue_weight=config.semantic_rescue_weight,
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
        query_expander=query_expander,
        diversifier=diversifier,
    )
