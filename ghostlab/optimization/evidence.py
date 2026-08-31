from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    policy_id: str
    parent_ids: tuple[str, ...] = ()
    kind: Literal[
        "positive", "negative", "failure", "ablation", "interaction", "observation"
    ]
    claim: str
    conditions: dict[str, str | int | float | bool]
    session_ids: tuple[str, ...]
    delta_technical_score: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mutation_family: str
    trace_refs: tuple[str, ...] = ()


class EvidenceStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: EvidenceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def read(self) -> list[EvidenceRecord]:
        if not self.path.exists():
            return []
        return [
            EvidenceRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


DecisionStatus = Literal[
    "NOT_TESTED",
    "TESTING",
    "PROMOTED",
    "PARKED_STANDALONE",
    "INTERACTION_RESERVE",
    "RETEST_AFTER_DEPENDENCY",
    "INVALID",
    "OUT_OF_SCOPE_WITH_REASON",
]


class DecisionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    split: str
    primary_metric: str = "recommended_technical_score"
    sample_count: int = Field(ge=1)
    candidate_score: float
    baseline_score: float | None = None
    delta: float | None = None
    hit_rate_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    mttc: float | None = Field(default=None, ge=1.0, le=11.0)


class TechniqueDecisionRecord(BaseModel):
    """Durable decision and retest rationale for one versioned technique."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    decision_id: str
    technique_id: str
    family: str
    status: DecisionStatus
    parent_policy_id: str | None = None
    hypothesis: str
    mechanism: str
    metrics: DecisionMetrics | None = None
    diagnosis: str
    dependencies: tuple[str, ...] = ()
    interaction_partners: tuple[str, ...] = ()
    retest_conditions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...]
    decided_at: str


class TechniqueDecisionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[TechniqueDecisionRecord]:
        if not self.path.exists():
            return []
        return [
            TechniqueDecisionRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, record: TechniqueDecisionRecord) -> None:
        existing = self.read()
        if any(item.decision_id == record.decision_id for item in existing):
            raise ValueError(f"duplicate decision_id: {record.decision_id}")
        if any(item.technique_id == record.technique_id for item in existing):
            raise ValueError(f"duplicate technique_id: {record.technique_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")


def family_ucb_allocation(
    gains: dict[str, list[float]], *, exploration: float = 0.2, floor: float = 0.01
) -> dict[str, float]:
    if not gains:
        return {}
    total_trials = sum(len(values) for values in gains.values())
    logits = {}
    for family, values in gains.items():
        mean = sum(values) / len(values) if values else 0.0
        bonus = exploration * math.sqrt(math.log(total_trials + 1) / (len(values) + 1))
        logits[family] = mean + bonus
    maximum = max(logits.values())
    exponentials = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(exponentials.values())
    raw = {key: value / total for key, value in exponentials.items()}
    floored = {key: max(floor, value) for key, value in raw.items()}
    normalization = sum(floored.values())
    return {key: value / normalization for key, value in floored.items()}
