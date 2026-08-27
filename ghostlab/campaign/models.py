from __future__ import annotations

import hashlib
import json
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Availability = Literal[
    "planned",
    "implementing",
    "available",
    "evaluated",
    "selected",
    "parked",
    "interaction_reserve",
    "retest_after_dependency",
    "invalid",
]
Fidelity = Literal["f0", "f1", "f2"]
JobState = Literal["pending", "running", "complete", "blocked", "failed"]


class ResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu: int = Field(default=1, ge=0)
    gpu: int = Field(default=0, ge=0)
    memory_gb: float = Field(default=1.0, ge=0.0)
    heavy_model: bool = False


class TechniqueSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    wave: int = Field(default=1, ge=1)
    availability: Availability
    default_enabled: bool = False
    source: str | None = None
    config_binding: str | None = None
    execution_class: str = "core"
    fit_required: bool = False
    assets: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    exclusive_group: str | None = None
    mechanism_tags: tuple[str, ...] = ()
    retest_triggers: tuple[str, ...] = ()
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    evidence_refs: tuple[str, ...] = ()

    @field_validator("source")
    @classmethod
    def safe_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cls._validate_path(value)
        return value

    @field_validator("assets", "evidence_refs")
    @classmethod
    def safe_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            cls._validate_path(value)
        return values

    @staticmethod
    def _validate_path(value: str) -> None:
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("technique paths must stay inside the project")

    @property
    def executable(self) -> bool:
        return self.availability in {
            "available",
            "evaluated",
            "selected",
            "parked",
            "interaction_reserve",
            "retest_after_dependency",
        }


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    baseline_id: str = Field(min_length=1)
    techniques: tuple[str, ...] = ()
    parameters: tuple[tuple[str, str | int | float | bool], ...] = ()
    complexity: int = Field(default=0, ge=0)
    generation: Literal["control", "single", "pair", "triple", "beam", "ablation"]

    @model_validator(mode="after")
    def unique_techniques(self) -> CandidateSpec:
        if len(set(self.techniques)) != len(self.techniques):
            raise ValueError("candidate techniques must be unique")
        return self

    def canonical_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"candidate_id"})
        payload["techniques"] = sorted(payload["techniques"])
        payload["parameters"] = sorted(payload["parameters"])
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class FidelityBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    f0: int = Field(ge=0)
    f1: int = Field(ge=0)
    f2: int = Field(ge=1)


class CampaignResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_jobs: int = Field(default=1, ge=1)
    gpu_jobs: int = Field(default=0, ge=0)
    memory_gb: float = Field(default=8.0, gt=0.0)
    heavy_model_jobs: int = Field(default=1, ge=0)


class CampaignManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    campaign_id: str = Field(min_length=1)
    parent_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    adaptive_split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    nested_split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_holdout_access: Literal["forbidden"] = "forbidden"
    baseline_presets: tuple[str, ...] = Field(min_length=1)
    baseline_techniques: tuple[str, ...] = Field(min_length=1)
    technique_ids: tuple[str, ...] = ()
    max_order: int = Field(default=2, ge=1, le=8)
    candidate_limit: int = Field(default=1000, ge=1)
    fidelity_budgets: FidelityBudget
    exploration_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    seeds: tuple[int, ...] = Field(min_length=1)
    max_wall_seconds: int = Field(gt=0)
    resources: CampaignResources = Field(default_factory=CampaignResources)
    promotion_rule: Literal["proposal_only"] = "proposal_only"

    @model_validator(mode="after")
    def unique_values(self) -> CampaignManifest:
        if len(set(self.technique_ids)) != len(self.technique_ids):
            raise ValueError("manifest technique IDs must be unique")
        if len(set(self.baseline_techniques)) != len(self.baseline_techniques):
            raise ValueError("manifest baseline technique IDs must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("manifest seeds must be unique")
        return self

    def canonical_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


class CampaignJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fidelity: Fidelity
    outer_fold: int | None = Field(default=None, ge=0)
    seed: int
    resources: ResourceRequest = Field(default_factory=ResourceRequest)


class JobOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    state: JobState
    score: float | None = None
    session_rewards: tuple[float, ...] = ()
    scenario_scores: dict[str, float] = Field(default_factory=dict)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None

    @model_validator(mode="after")
    def outcome_consistency(self) -> JobOutcome:
        if self.state == "complete" and self.score is None:
            raise ValueError("complete outcomes require a score")
        if self.state == "failed" and not self.error:
            raise ValueError("failed outcomes require an error")
        return self
