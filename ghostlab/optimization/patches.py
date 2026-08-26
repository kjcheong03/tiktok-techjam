from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ghostlab.competition.contract import AskAttribute


class SearchPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state_variant: Literal["current", "raw_history", "single", "multi", "compressed"]
    question_variant: Literal[
        "none",
        "fixed",
        "sequence",
        "missing_priority",
        "feature_first",
        "uncertainty",
        "other_always",
        "adaptive",
    ]
    question_order: tuple[AskAttribute, ...] | None = None
    repeat_last_question: bool = False
    negative_evidence: bool = True
    override_invalidation: bool = True
    retrieval_route: Literal["keyword", "dense", "rrf", "weighted"] = "keyword"
    sparse_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    reranker: Literal["none", "linear"] = "none"

    @model_validator(mode="after")
    def validate_policy(self) -> SearchPolicyConfig:
        if self.question_variant == "sequence" and not self.question_order:
            raise ValueError("sequence policy needs a non-empty question order")
        if (
            self.retrieval_route == "weighted"
            and abs(self.sparse_weight + self.dense_weight - 1.0) > 1e-9
        ):
            raise ValueError("weighted route weights must sum to one")
        return self

    def policy_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


class TypedOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: Literal[
        "state_variant",
        "question_variant",
        "question_order",
        "repeat_last_question",
        "negative_evidence",
        "override_invalidation",
        "retrieval_route",
        "sparse_weight",
        "dense_weight",
        "reranker",
    ]
    value: object


class PolicyPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    patch_id: str
    parent_policy_id: str
    mutation_family: str
    hypothesis: str
    operations: tuple[TypedOperation, ...] = Field(min_length=1, max_length=8)
    cheapest_fidelity: Literal["f0", "f1", "f2"] = "f0"
    falsification_condition: str
    risk: Literal["low", "medium", "high"]


def materialize_patch(
    parent: SearchPolicyConfig, patch: PolicyPatch
) -> SearchPolicyConfig:
    values = parent.model_dump()
    seen: set[str] = set()
    for operation in patch.operations:
        if operation.field in seen:
            raise ValueError(f"patch writes {operation.field} more than once")
        seen.add(operation.field)
        values[operation.field] = operation.value
    return SearchPolicyConfig.model_validate(values)


def compatible(left: PolicyPatch, right: PolicyPatch) -> bool:
    left_fields = {operation.field for operation in left.operations}
    right_fields = {operation.field for operation in right.operations}
    return left_fields.isdisjoint(right_fields)


def crossover(
    parent: SearchPolicyConfig, left: PolicyPatch, right: PolicyPatch
) -> SearchPolicyConfig:
    if not compatible(left, right):
        raise ValueError("patches write conflicting policy fields")
    return materialize_patch(
        parent,
        PolicyPatch(
            patch_id=f"{left.patch_id}+{right.patch_id}",
            parent_policy_id=left.parent_policy_id,
            mutation_family="compatible_crossover",
            hypothesis=f"{left.hypothesis}; {right.hypothesis}",
            operations=(*left.operations, *right.operations),
            cheapest_fidelity=max(left.cheapest_fidelity, right.cheapest_fidelity),
            falsification_condition=(
                f"{left.falsification_condition}; {right.falsification_condition}"
            ),
            risk=max(left.risk, right.risk),
        ),
    )
