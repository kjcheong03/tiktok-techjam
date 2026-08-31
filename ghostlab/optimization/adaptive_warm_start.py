from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig


class AdaptiveWarmStartSpec(BaseModel):
    """A historical candidate translated onto the fixed Adaptive Hybrid contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    warm_start_id: str = Field(min_length=1)
    source_candidate_id: str = Field(min_length=1)
    source_preset: str = Field(min_length=1)
    source_preset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    architecture: Literal["adaptive_hybrid_1a_3b_v1"]
    techniques: tuple[str, ...] = Field(min_length=1)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    inherited_mechanisms: tuple[str, ...] = ()
    excluded_source_techniques: dict[str, str] = Field(default_factory=dict)

    @field_validator("source_preset")
    @classmethod
    def source_stays_inside_repository(cls, value: str) -> str:
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("warm-start source preset must stay inside the repository")
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adaptive_warm_start(
    path: str | Path,
    *,
    project_root: str | Path,
    baseline: AdaptiveHybridConfig,
    registry: AdaptiveTechniqueRegistry,
) -> tuple[AdaptiveWarmStartSpec, CandidateSpec]:
    """Load and preflight a warm seed without executing its historical runtime."""

    root = Path(project_root).resolve()
    warm_path = Path(path)
    if not warm_path.is_absolute():
        warm_path = root / warm_path
    warm_path = warm_path.resolve()
    if root not in warm_path.parents or not warm_path.is_file():
        raise ValueError("warm-start specification must be a repository file")
    spec = AdaptiveWarmStartSpec.model_validate_json(
        warm_path.read_text(encoding="utf-8")
    )
    if baseline.architecture != spec.architecture:
        raise ValueError(
            "warm-start architecture does not match the fixed baseline contract"
        )

    source_path = (root / spec.source_preset).resolve()
    if root not in source_path.parents or not source_path.is_file():
        raise ValueError("warm-start source preset is missing")
    if _sha256(source_path) != spec.source_preset_sha256:
        raise ValueError("warm-start source preset hash mismatch")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if source_payload.get("experiment_id") != spec.source_candidate_id:
        raise ValueError("warm-start source candidate ID mismatch")

    inventory = registry.inventory()
    additions = tuple(sorted(set(spec.techniques)))
    non_promotable = set(additions) - set(inventory.promotable)
    if non_promotable:
        raise ValueError(
            "warm start contains non-promotable techniques: "
            f"{sorted(non_promotable)}"
        )
    techniques = tuple(sorted(set(inventory.compulsory) | set(additions)))
    candidate = CandidateSpec(
        candidate_id=f"warm-start-{spec.warm_start_id}",
        baseline_id=baseline.policy_id,
        techniques=techniques,
        parameters=tuple(sorted(spec.parameters.items())),
        complexity=len(additions),
        generation="beam",
    )
    registry.validate_candidate(candidate)
    materialized = registry.materialize(baseline, candidate)
    if materialized.architecture != baseline.architecture:
        raise ValueError("warm start changed the fixed architecture")
    return spec, candidate
