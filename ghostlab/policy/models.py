from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ghostlab.competition.contract import AskAttribute

RetrievalRoute = Literal["keyword", "dense", "rrf", "weighted_fusion"]
StateMode = Literal["off", "single", "multi", "compressed", "raw_history"]
QuestionPolicy = Literal[
    "none",
    "fixed",
    "missing_priority",
    "feature_first",
    "uncertainty",
    "other_always",
    "learned",
    "adaptive",
    "sequence",
]


class ModelAssetConfig(BaseModel):
    """Content-addressed local model asset resolved from the project root."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePath(value)
        if path.is_absolute():
            raise ValueError("model asset paths must be relative")
        if ".." in path.parts:
            raise ValueError("model asset paths cannot leave the project root")
        if not path.name or value.strip() != value:
            raise ValueError("model asset path must name a local file")
        return value


class TechniqueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    retrieval_route: RetrievalRoute = "keyword"
    state_mode: StateMode = "single"
    question_policy: QuestionPolicy = "missing_priority"
    retrieval_k: int = Field(default=200, ge=10, le=1000)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    sparse_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_field_weights: tuple[float, float, float, float, float, float] | None = None
    negative_evidence: bool = True
    provenance: bool = True
    override_invalidation: bool = True
    profile_priors: bool = False
    quality_prior_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    reranker: Literal["none", "linear", "learned_linear", "guarded_constraint_gbdt"] = (
        "none"
    )
    rerank_k: int = Field(default=50, ge=10, le=200)
    learned_weights: tuple[float, float, float, float, float, float, float] | None = (
        None
    )
    learned_l2: float = Field(default=0.1, ge=0.0)
    learned_training_pairs: int = Field(default=0, ge=0)
    base_model_asset: ModelAssetConfig | None = None
    constraint_model_asset: ModelAssetConfig | None = None
    question_order: tuple[AskAttribute, ...] = Field(default=(), max_length=10)
    enabled_filters: tuple[str, ...] = ()

    @field_validator("sparse_field_weights", "learned_weights")
    @classmethod
    def finite_weights(
        cls, value: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        if value is not None and not all(math.isfinite(item) for item in value):
            raise ValueError("runtime weights must be finite")
        return value

    @model_validator(mode="after")
    def validate_combination(self) -> TechniqueConfig:
        if self.retrieval_route == "weighted_fusion" and not math.isclose(
            self.sparse_weight + self.dense_weight, 1.0, abs_tol=1e-9
        ):
            raise ValueError("weighted fusion weights must sum to one")
        if self.state_mode == "off" and self.question_policy in {
            "missing_priority",
            "feature_first",
            "uncertainty",
            "other_always",
            "learned",
            "adaptive",
            "sequence",
        }:
            raise ValueError("the selected question policy requires state")
        if self.sparse_field_weights is not None and any(
            weight < 0.0 for weight in self.sparse_field_weights
        ):
            raise ValueError("sparse field weights cannot be negative")
        if self.question_policy == "sequence" and not self.question_order:
            raise ValueError("sequence question policy requires question_order")
        if self.reranker == "learned_linear" and self.learned_weights is None:
            raise ValueError("learned linear reranker requires learned_weights")
        if self.reranker != "learned_linear" and self.learned_weights is not None:
            raise ValueError("learned_weights require the learned linear reranker")
        model_assets = (self.base_model_asset, self.constraint_model_asset)
        if self.reranker == "guarded_constraint_gbdt":
            if any(asset is None for asset in model_assets):
                raise ValueError(
                    "guarded constraint GBDT requires base and constraint assets"
                )
            if self.retrieval_route != "keyword":
                raise ValueError("guarded constraint GBDT requires keyword retrieval")
            if self.state_mode != "raw_history":
                raise ValueError("guarded constraint GBDT requires raw_history state")
            if self.question_policy != "sequence":
                raise ValueError("guarded constraint GBDT requires a question sequence")
            if self.sparse_field_weights is None:
                raise ValueError(
                    "guarded constraint GBDT requires sparse field weights"
                )
        elif any(asset is not None for asset in model_assets):
            raise ValueError(
                "GBDT model assets require the guarded constraint reranker"
            )
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=1)
    techniques: TechniqueConfig = Field(default_factory=TechniqueConfig)
    fallback_policy_id: str = "keyword_current_turn"
    trace_enabled: bool = False

    def canonical_hash(self) -> str:
        value = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(value.encode()).hexdigest()


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parent_asin: str = Field(min_length=1)
    route: RetrievalRoute
    rank: int = Field(ge=1)
    raw_score: float | None = None
    normalized_score: Annotated[float | None, Field(ge=0.0, le=1.0)] = None

    @field_validator("raw_score", "normalized_score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate scores must be finite")
        return value


class RankedCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    items: tuple[RankedCandidate, ...]
    route: RetrievalRoute
    requested_k: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ranking(self) -> RankedCandidates:
        identifiers = [item.parent_asin for item in self.items]
        ranks = [item.rank for item in self.items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate identifiers must be unique")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be contiguous and one-based")
        return self


Scalar = bool | int | float | str
Operator = Literal[
    "eq", "ne", "lt", "le", "gt", "ge", "contains", "is_missing", "is_not_missing"
]


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    feature: str
    operator: Operator
    value: Scalar | None = None


class JointAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ask_attribute: AskAttribute | None = None
    retrieval_route: RetrievalRoute = "keyword"
    retrieval_k: int = Field(default=200, ge=10, le=1000)
    sparse_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.25, ge=0.0, le=1.0)


class ActionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ask_attribute: AskAttribute | None | Literal["__inherit__"] = "__inherit__"
    retrieval_route: RetrievalRoute | Literal["__inherit__"] = "__inherit__"
    retrieval_k: int | Literal["__inherit__"] = "__inherit__"
    sparse_weight: float | Literal["__inherit__"] = "__inherit__"
    dense_weight: float | Literal["__inherit__"] = "__inherit__"


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: str
    all_conditions: tuple[Predicate, ...] = Field(max_length=4)
    action_patch: ActionPatch


class DecisionList(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rules: tuple[PolicyRule, ...] = Field(max_length=32)
    default_action: JointAction
