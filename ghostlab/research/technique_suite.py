from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path, PurePath
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ghostlab.competition.contract import AgentProtocol, AskAttribute

PROJECT_ROOT = Path(__file__).resolve().parents[2]

StateVariant = Literal[
    "current",
    "raw_history",
    "single",
    "multi",
    "compressed",
    "baseline_v2",
]
QueryVariant = Literal[
    "raw_history",
    "structured_active",
    "category_constraints",
    "raw_plus_active",
    "compressed_raw",
    "negation_safe_hybrid",
    "coverage_adaptive_v2",
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
    "candidate_eig",
    "reward_voi",
    "joint_observable",
    "distilled_joint",
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
Reranker = Literal[
    "none",
    "linear",
    "metadata_gbdt",
    "reward_lambdamart",
    "turn_aware_lambdamart",
    "rank_ensemble",
]
QueryExpansion = Literal["off", "prf"]
Diversification = Literal["off", "facet_mmr"]
Normalizer = Literal["off", "catalog_v1"]
RoutingVariant = Literal["off", "calibrated"]
RecommendationHistory = Literal["off", "correction_scoped"]
ActivationMode = Literal["always", "uncertain"]
ResidualFeatureSet = Literal["rank", "metadata", "full_context"]
ResidualModelVariant = Literal[
    "regularized_logistic",
    "hist_gbdt_d2_lr005",
    "hist_gbdt_d3_lr005",
    "hist_gbdt_d3_lr01",
    "ensemble_logistic_gbdt_d2_lr005",
    "ensemble_logistic_gbdt_d3_lr005",
    "ensemble_logistic_gbdt_d3_lr01",
]


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
    eig_candidate_k: int = Field(default=100, ge=20, le=400)
    question_value_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    question_max_turn: int = Field(default=10, ge=0, le=10)
    joint_policy_asset: str | None = None

    normalizer: Normalizer = "off"
    normalizer_asset: str | None = None
    constraint_confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    retrieval_route: RetrievalRoute = "keyword"
    dense_backend: DenseBackend = "off"
    dense_model_path: str | None = None
    sparse_weights: tuple[float, float, float, float, float, float] | None = None
    sparse_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    retrieval_k: int = Field(default=200, ge=10, le=400)
    rrf_constant: int = Field(default=60, ge=1, le=200)
    dense_activation: ActivationMode = "always"
    dense_activation_min_entropy: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_rescue_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    learned_sparse_asset: str | None = None
    late_interaction_asset: str | None = None

    query_expansion: QueryExpansion = "off"
    expansion_feedback_k: int = Field(default=5, ge=2, le=50)
    expansion_min_support: float = Field(default=0.4, gt=0.0, le=1.0)
    expansion_max_terms: int = Field(default=4, ge=0, le=20)
    expansion_max_added_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    query_expansion_activation: ActivationMode = "always"
    query_expansion_min_entropy: float = Field(default=0.0, ge=0.0, le=1.0)

    negative_evidence: bool = True
    provenance: bool = True
    override_invalidation: bool = True
    recommendation_history: RecommendationHistory = "off"
    structured_filter: bool = False
    profile_prior_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    profile_prior_max_turn: int = Field(default=10, ge=0, le=10)
    quality_prior_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    reranker: Reranker = "none"
    reranker_model_asset: str | None = None
    rerank_k: int = Field(default=50, ge=10, le=400)
    cross_encoder_enabled: bool = False
    cross_encoder_model_path: str | None = None
    cross_encoder_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_encoder_rerank_k: int = Field(default=20, ge=1, le=200)
    cross_encoder_activation: ActivationMode = "always"
    cross_encoder_min_entropy: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_encoder_min_turn: int = Field(default=1, ge=1, le=10)
    diversification: Diversification = "off"
    diversification_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    diversification_rerank_k: int = Field(default=30, ge=10, le=200)
    diversification_output_k: int = Field(default=10, ge=1, le=50)
    diversification_max_turn: int = Field(default=2, ge=1, le=9)
    diversification_max_constraints: int = Field(default=1, ge=0, le=10)

    residual_reranker_enabled: bool = False
    residual_model_asset: str | None = None
    residual_feature_set: ResidualFeatureSet = "full_context"
    residual_model_variant: ResidualModelVariant = "regularized_logistic"
    residual_regularization: float = Field(default=0.2, gt=0.0, le=20.0)
    residual_rerank_depth: int = Field(default=10, ge=2, le=10)
    residual_model_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    residual_minimum_expected_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_minimum_probability_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_maximum_moved_ids: int = Field(default=10, ge=2, le=10)

    routing_variant: RoutingVariant = "off"
    router_asset: str | None = None
    component_fallback: bool = False

    @field_validator(
        "compiled_config_path",
        "learned_question_asset",
        "joint_policy_asset",
        "normalizer_asset",
        "dense_model_path",
        "reranker_model_asset",
        "cross_encoder_model_path",
        "learned_sparse_asset",
        "late_interaction_asset",
        "router_asset",
        "residual_model_asset",
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
                or self.normalizer != "off"
                or self.routing_variant != "off"
            ):
                raise ValueError(
                    "Wave 2 retrieval switches require the experimental engine"
                )
            return self
        if self.compiled_config_path is not None:
            raise ValueError("compiled_config_path is only valid for compiled engine")
        if (
            self.query_variant == "coverage_adaptive_v2"
            and self.state_variant != "baseline_v2"
        ):
            raise ValueError(
                "coverage-adaptive V2 query requires state_variant=baseline_v2"
            )
        if self.state_variant == "baseline_v2" and self.normalizer != "off":
            raise ValueError(
                "State Baseline V2 uses its frozen legacy adapter; catalog "
                "normalization must be tested as a separate state implementation"
            )
        if (
            self.recommendation_history == "correction_scoped"
            and self.state_variant != "baseline_v2"
        ):
            raise ValueError(
                "correction-scoped recommendation history requires "
                "state_variant=baseline_v2"
            )
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
        joint_variants = {"joint_observable", "distilled_joint"}
        if (self.question_variant in joint_variants) != (
            self.joint_policy_asset is not None
        ):
            raise ValueError("joint question policies require exactly one joint asset")
        if self.question_variant == "sequence" and not self.question_order:
            raise ValueError("sequence question policy requires question_order")
        learned_rankers = {
            "metadata_gbdt",
            "reward_lambdamart",
            "turn_aware_lambdamart",
            "rank_ensemble",
        }
        if (self.reranker in learned_rankers) != (
            self.reranker_model_asset is not None
        ):
            raise ValueError("learned rerankers require exactly one model asset")
        if (self.normalizer == "catalog_v1") != (self.normalizer_asset is not None):
            raise ValueError(
                "catalog normalization requires exactly one ontology asset"
            )
        if (self.routing_variant == "calibrated") != (self.router_asset is not None):
            raise ValueError("calibrated routing requires exactly one router asset")
        if self.component_fallback and self.routing_variant != "calibrated":
            raise ValueError("component fallback requires calibrated routing")
        if self.cross_encoder_enabled:
            if self.cross_encoder_model_path is None:
                raise ValueError("cross-encoder requires cross_encoder_model_path")
            if self.cross_encoder_weight <= 0.0:
                raise ValueError("enabled cross-encoder requires a positive weight")
        elif self.cross_encoder_model_path is not None or self.cross_encoder_weight:
            raise ValueError("cross-encoder fields require cross_encoder_enabled")
        if self.diversification_output_k > self.diversification_rerank_k:
            raise ValueError("diversification output depth cannot exceed rerank depth")
        if not self.residual_reranker_enabled and self.residual_model_asset is not None:
            raise ValueError("residual model asset requires residual reranking")
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    from ghostlab.runtime.unified_experimental import (
        CandidateReranker,
        ExperimentalAgent,
        JointCandidatePolicy,
    )

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

    catalog_normalizer = None
    if config.normalizer == "catalog_v1":
        from ghostlab.state.catalog_ontology import CatalogOntology
        from ghostlab.state.normalization import CatalogStateNormalizer

        assert config.normalizer_asset is not None
        ontology = CatalogOntology.from_path(_project_path(config.normalizer_asset))
        if ontology.catalog_sha256 != _sha256_file(catalog_path):
            raise ValueError("catalog ontology was built for another catalog")
        catalog_normalizer = CatalogStateNormalizer(
            ontology, confidence_threshold=config.constraint_confidence
        )

    eig_policy = None
    if config.question_variant in {"candidate_eig", "reward_voi"}:
        from ghostlab.policy.eig_questions import CandidateEIGPolicy

        eig_policy = CandidateEIGPolicy(
            question_value_margin=config.question_value_margin
        )

    joint_policy: JointCandidatePolicy | None = None
    if config.joint_policy_asset is not None:
        if config.question_variant == "distilled_joint":
            from ghostlab.policy.distilled_expert import DistilledExpertPolicy

            joint_policy = DistilledExpertPolicy.from_path(
                _project_path(config.joint_policy_asset)
            )
        else:
            from ghostlab.policy.joint_policy import JointObservablePolicy

            joint_policy = JointObservablePolicy.from_path(
                _project_path(config.joint_policy_asset)
            )

    calibrated_router = None
    component_fallback = None
    if config.routing_variant == "calibrated":
        from ghostlab.policy.calibrated_router import CalibratedRouteModel

        assert config.router_asset is not None
        calibrated_router = CalibratedRouteModel.from_path(
            _project_path(config.router_asset)
        )
        if config.component_fallback:
            from ghostlab.runtime.component_fallback import ComponentFallback

            component_fallback = ComponentFallback()

    learned_reranker: CandidateReranker | None = None
    if config.reranker in {
        "metadata_gbdt",
        "reward_lambdamart",
        "turn_aware_lambdamart",
        "rank_ensemble",
    }:
        from ghostlab.retrieval.gbdt import (
            GBDTFeatureStore,
            LambdaMARTModel,
            LambdaMARTReranker,
        )
        from ghostlab.retrieval.quality import CatalogQualityReranker

        assert config.reranker_model_asset is not None
        quality = CatalogQualityReranker(catalog_path)
        features = GBDTFeatureStore(catalog_path, quality=quality.quality)
        model_asset = _project_path(config.reranker_model_asset)
        if config.reranker == "rank_ensemble":
            from ghostlab.retrieval.ensemble import (
                ModelRankEnsembleReranker,
                RankEnsembleAsset,
            )

            learned_reranker = ModelRankEnsembleReranker.from_asset(
                features,
                RankEnsembleAsset.load(model_asset),
                project_root=PROJECT_ROOT,
            )
        else:
            learned_reranker = LambdaMARTReranker(
                features, LambdaMARTModel.load(model_asset)
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

    agent = ExperimentalAgent(
        catalog_path,
        state_variant=config.state_variant,
        normalizer=config.normalizer,
        catalog_normalizer=catalog_normalizer,
        query_variant=config.query_variant,
        question_variant=config.question_variant,
        question_order=tuple(config.question_order),
        learned_question_model=learned_question_model,
        eig_policy=eig_policy,
        eig_candidate_k=config.eig_candidate_k,
        question_max_turn=config.question_max_turn,
        joint_policy=joint_policy,
        routing_variant=config.routing_variant,
        calibrated_router=calibrated_router,
        component_fallback=component_fallback,
        retrieval_route=config.retrieval_route,
        dense_retriever=dense,
        semantic_rescue_retriever=semantic_rescue,
        semantic_rescue_weight=config.semantic_rescue_weight,
        sparse_weight=config.sparse_weight,
        dense_weight=config.dense_weight,
        retrieval_k=config.retrieval_k,
        rrf_constant=config.rrf_constant,
        dense_activation=config.dense_activation,
        dense_activation_min_entropy=config.dense_activation_min_entropy,
        sparse_weights=config.sparse_weights,
        negative_evidence=config.negative_evidence,
        provenance=config.provenance,
        override_invalidation=config.override_invalidation,
        structured_filter=config.structured_filter,
        profile_prior_weight=config.profile_prior_weight,
        profile_prior_max_turn=config.profile_prior_max_turn,
        quality_prior_weight=config.quality_prior_weight,
        reranker="linear" if config.reranker == "linear" else "none",
        learned_reranker=learned_reranker,
        learned_rerank_k=config.rerank_k,
        cross_encoder_reranker=cross_encoder,
        cross_encoder_weight=config.cross_encoder_weight,
        cross_encoder_rerank_k=config.cross_encoder_rerank_k,
        cross_encoder_activation=config.cross_encoder_activation,
        cross_encoder_min_entropy=config.cross_encoder_min_entropy,
        cross_encoder_min_turn=config.cross_encoder_min_turn,
        query_expander=query_expander,
        query_expansion_activation=config.query_expansion_activation,
        query_expansion_min_entropy=config.query_expansion_min_entropy,
        diversifier=diversifier,
        recommendation_history=config.recommendation_history,
    )
    if not config.residual_reranker_enabled:
        return agent
    if config.residual_model_asset is None:
        raise ValueError("residual reranking requires a fold-fitted model asset")
    from ghostlab.retrieval.residual import (
        MembershipPreservingResidualReranker,
        ResidualAgentAdapter,
        ResidualPolicy,
    )

    residual = MembershipPreservingResidualReranker.from_asset(
        catalog_path,
        _project_path(config.residual_model_asset),
        policy=ResidualPolicy(
            rerank_depth=config.residual_rerank_depth,
            model_weight=config.residual_model_weight,
            minimum_expected_gain=config.residual_minimum_expected_gain,
            minimum_probability_margin=config.residual_minimum_probability_margin,
            maximum_moved_ids=config.residual_maximum_moved_ids,
        ),
    )
    return ResidualAgentAdapter(agent, residual)
