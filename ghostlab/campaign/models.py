from __future__ import annotations

import hashlib
import json
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ghostlab.training.protocol import FitReceipt

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
ExecutionMode = Literal["runtime", "anchor_only", "research_only"]


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
    execution_mode: ExecutionMode = "runtime"
    selection_safe: bool = True
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

    schema_version: Literal[1, 2] = 1
    campaign_id: str = Field(min_length=1)
    parent_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    adaptive_split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    nested_split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_space_path: str | None = None
    search_space_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    protected_holdout_access: Literal["forbidden"] = "forbidden"
    search_outer_folds: tuple[int, ...] = Field(default=(0, 2, 3), min_length=1)
    confirmation_outer_folds: tuple[int, ...] = Field(default=(1, 4), min_length=1)
    baseline_presets: tuple[str, ...] = Field(min_length=1)
    baseline_techniques: tuple[str, ...] = Field(min_length=1)
    baseline_techniques_by_preset: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    baseline_search_modes: dict[str, Literal["composable", "control_only"]] = Field(
        default_factory=dict
    )
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
        if self.schema_version == 2 and (
            self.search_space_path is None or self.search_space_hash is None
        ):
            raise ValueError("schema v2 campaigns must freeze their search space")
        if (self.search_space_path is None) != (self.search_space_hash is None):
            raise ValueError("search-space path and hash must be declared together")
        if self.search_space_path is not None:
            path = PurePath(self.search_space_path)
            if path.is_absolute() or ".." in path.parts or not path.name:
                raise ValueError("search-space path must stay inside the project")
        if len(set(self.technique_ids)) != len(self.technique_ids):
            raise ValueError("manifest technique IDs must be unique")
        if len(set(self.baseline_techniques)) != len(self.baseline_techniques):
            raise ValueError("manifest baseline technique IDs must be unique")
        unknown_presets = (
            set(self.baseline_techniques_by_preset) | set(self.baseline_search_modes)
        ) - set(self.baseline_presets)
        if unknown_presets:
            raise ValueError(
                "baseline technique mapping references unknown presets: "
                f"{sorted(unknown_presets)}"
            )
        for preset, technique_ids in self.baseline_techniques_by_preset.items():
            if len(set(technique_ids)) != len(technique_ids):
                raise ValueError(
                    f"baseline technique IDs must be unique for preset {preset}"
                )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("manifest seeds must be unique")
        if len(set(self.search_outer_folds)) != len(self.search_outer_folds):
            raise ValueError("search outer-fold IDs must be unique")
        if len(set(self.confirmation_outer_folds)) != len(
            self.confirmation_outer_folds
        ):
            raise ValueError("confirmation outer-fold IDs must be unique")
        overlap = set(self.search_outer_folds) & set(self.confirmation_outer_folds)
        if overlap:
            raise ValueError(
                f"search and confirmation outer folds overlap: {sorted(overlap)}"
            )
        return self

    def validate_fold_partition(self, outer_fold_count: int) -> None:
        if outer_fold_count <= 1:
            raise ValueError("campaign requires at least two outer folds")
        declared = set(self.search_outer_folds) | set(self.confirmation_outer_folds)
        expected = set(range(outer_fold_count))
        if declared != expected:
            raise ValueError(
                "search and confirmation folds must partition every nested outer "
                f"fold exactly; expected {sorted(expected)}, got {sorted(declared)}"
            )

    def techniques_for_preset(self, preset: str) -> tuple[str, ...]:
        if preset not in self.baseline_presets:
            raise ValueError(f"unknown baseline preset: {preset}")
        return self.baseline_techniques_by_preset.get(preset, self.baseline_techniques)

    def search_mode_for_preset(
        self, preset: str
    ) -> Literal["composable", "control_only"]:
        if preset not in self.baseline_presets:
            raise ValueError(f"unknown baseline preset: {preset}")
        return self.baseline_search_modes.get(preset, "composable")

    def canonical_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
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
    hit_rate_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    mttc: float | None = Field(default=None, ge=1.0, le=11.0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    latency_p95_ms: float = Field(default=0.0, ge=0.0)
    memory_mb: float = Field(default=0.0, ge=0.0)
    fit_receipts: tuple[FitReceipt, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def outcome_consistency(self) -> JobOutcome:
        if self.state == "complete" and self.score is None:
            raise ValueError("complete outcomes require a score")
        if self.state == "failed" and not self.error:
            raise ValueError("failed outcomes require an error")
        return self


class MetricSnapshot(BaseModel):
    """Comparable aggregate metrics from one frozen evaluation population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    technical_score: float
    hit_rate_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    mttc: float | None = Field(default=None, ge=1.0, le=11.0)


class ChampionComparison(BaseModel):
    """Same-fold evidence shown to a human before any promotion decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    champion_candidate_id: str = Field(min_length=1)
    champion_baseline_id: str = Field(min_length=1)
    candidate_metrics: MetricSnapshot
    champion_metrics: MetricSnapshot
    technical_score_delta: float
    hit_rate_at_10_delta: float | None = None
    mrr_delta: float | None = None
    mttc_delta: float | None = None
    paired_mean_delta: float
    confidence_interval: tuple[float, float]
    randomization_pvalue: float = Field(ge=0.0, le=1.0)
    paired_session_count: int = Field(gt=0)
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)
    scenario_deltas: dict[str, float] = Field(default_factory=dict)
    beats_champion_point_estimate: bool
    statistically_supported: bool
    maximum_scenario_regression: float = Field(default=0.02, ge=0.0)
    no_material_scenario_regression: bool
    fit_receipts_verified: bool
    promotion_recommended: bool
    automatic_promotion: Literal[False] = False

    @model_validator(mode="after")
    def internally_consistent(self) -> ChampionComparison:
        def close(left: float, right: float) -> bool:
            return abs(left - right) <= 1e-9

        expected_score_delta = (
            self.candidate_metrics.technical_score
            - self.champion_metrics.technical_score
        )
        if not close(self.technical_score_delta, expected_score_delta):
            raise ValueError("champion technical-score delta is inconsistent")
        for name, reported in (
            ("hit_rate_at_10", self.hit_rate_at_10_delta),
            ("mrr", self.mrr_delta),
            ("mttc", self.mttc_delta),
        ):
            candidate = getattr(self.candidate_metrics, name)
            champion = getattr(self.champion_metrics, name)
            expected = (
                None if candidate is None or champion is None else candidate - champion
            )
            if (reported is None) != (expected is None) or (
                reported is not None
                and expected is not None
                and not close(reported, expected)
            ):
                raise ValueError(f"champion {name} delta is inconsistent")
        if self.beats_champion_point_estimate != (self.paired_mean_delta > 0.0):
            raise ValueError("champion point-estimate flag is inconsistent")
        if self.confidence_interval[0] > self.confidence_interval[1]:
            raise ValueError("champion confidence interval is reversed")
        if self.wins + self.ties + self.losses != self.paired_session_count:
            raise ValueError("champion paired outcome counts are inconsistent")
        expected_support = (
            self.confidence_interval[0] > 0.0 and self.randomization_pvalue <= 0.05
        )
        if self.statistically_supported != expected_support:
            raise ValueError("champion statistical-support flag is inconsistent")
        expected_scenario_safety = all(
            delta >= -self.maximum_scenario_regression
            for delta in self.scenario_deltas.values()
        )
        if self.no_material_scenario_regression != expected_scenario_safety:
            raise ValueError("champion scenario-safety flag is inconsistent")
        expected_recommendation = (
            self.beats_champion_point_estimate
            and self.statistically_supported
            and self.no_material_scenario_regression
            and self.fit_receipts_verified
        )
        if self.promotion_recommended != expected_recommendation:
            raise ValueError("champion promotion recommendation is inconsistent")
        return self
