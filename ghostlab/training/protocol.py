from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _ids_hash(values: tuple[str, ...]) -> str:
    encoded = json.dumps(sorted(values), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class FitRequest(BaseModel):
    """Explicit data boundary supplied to any campaign-fitted component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    technique_id: str = Field(min_length=1)
    outer_fold: int = Field(ge=0)
    inner_fold: int = Field(ge=0)
    train_sample_ids: tuple[str, ...] = Field(min_length=1)
    validation_sample_ids: tuple[str, ...] = Field(min_length=1)
    seed: int

    @model_validator(mode="after")
    def disjoint(self) -> FitRequest:
        assert_disjoint_fit(self.train_sample_ids, self.validation_sample_ids)
        return self


class FitReceipt(BaseModel):
    """Auditable proof binding a fitted asset to its exact fold and samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    technique_id: str
    outer_fold: int
    inner_fold: int
    seed: int
    train_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_path: str
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_fit(cls, request: FitRequest, asset_path: Path) -> FitReceipt:
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        return cls(
            technique_id=request.technique_id,
            outer_fold=request.outer_fold,
            inner_fold=request.inner_fold,
            seed=request.seed,
            train_sample_ids_sha256=_ids_hash(request.train_sample_ids),
            validation_sample_ids_sha256=_ids_hash(request.validation_sample_ids),
            asset_path=asset_path.as_posix(),
            asset_sha256=digest,
        )


class FoldSafeTrainer(Protocol):
    def fit(self, request: FitRequest, output_path: Path) -> FitReceipt: ...


def assert_disjoint_fit(
    train_sample_ids: tuple[str, ...], validation_sample_ids: tuple[str, ...]
) -> None:
    if len(train_sample_ids) != len(set(train_sample_ids)):
        raise ValueError("fit sample IDs must be unique")
    if len(validation_sample_ids) != len(set(validation_sample_ids)):
        raise ValueError("validation sample IDs must be unique")
    overlap = set(train_sample_ids) & set(validation_sample_ids)
    if overlap:
        raise ValueError(f"fit/validation leakage detected: {sorted(overlap)}")
