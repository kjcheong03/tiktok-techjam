from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from ghostlab.optimization.evidence import (
    TechniqueDecisionRecord,
    TechniqueDecisionStore,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts/evidence/technique_decisions.jsonl"
TESTED_STATUSES = {
    "PROMOTED",
    "PARKED_STANDALONE",
    "INTERACTION_RESERVE",
    "RETEST_AFTER_DEPENDENCY",
}


def validate_records(
    records: list[TechniqueDecisionRecord], root: Path = ROOT
) -> list[str]:
    errors: list[str] = []
    decision_ids = [record.decision_id for record in records]
    technique_ids = [record.technique_id for record in records]
    known_techniques = set(technique_ids)

    for label, values in (
        ("decision_id", decision_ids),
        ("technique_id", technique_ids),
    ):
        duplicates = sorted(
            value for value, count in Counter(values).items() if count > 1
        )
        if duplicates:
            errors.append(f"duplicate {label}: {', '.join(duplicates)}")

    for record in records:
        prefix = f"{record.decision_id}/{record.technique_id}"
        linked = set(record.dependencies) | set(record.interaction_partners)
        unknown = sorted(linked - known_techniques)
        if unknown:
            errors.append(
                f"{prefix}: unknown linked technique(s): {', '.join(unknown)}"
            )
        if record.technique_id in linked:
            errors.append(f"{prefix}: technique links to itself")
        if record.status == "NOT_TESTED" and record.metrics is not None:
            errors.append(f"{prefix}: NOT_TESTED record has metrics")
        if record.status in TESTED_STATUSES and record.metrics is None:
            errors.append(f"{prefix}: tested decision has no metrics")
        if record.metrics is not None:
            metrics = record.metrics
            if (
                metrics.baseline_score is not None
                and metrics.delta is not None
                and abs(
                    metrics.candidate_score - metrics.baseline_score - metrics.delta
                )
                > 1e-6
            ):
                errors.append(f"{prefix}: candidate - baseline does not equal delta")
        try:
            date.fromisoformat(record.decided_at)
        except ValueError:
            errors.append(f"{prefix}: decided_at is not an ISO date")
        if not record.evidence_refs:
            errors.append(f"{prefix}: no evidence references")
        for reference in record.evidence_refs:
            if reference.startswith("/") or ".." in Path(reference).parts:
                errors.append(f"{prefix}: unsafe evidence reference: {reference}")
                continue
            if "guarded" in Path(reference).parts or "f3_v1" in reference.lower():
                errors.append(f"{prefix}: protected-holdout reference: {reference}")
            if not (root / reference).is_file():
                errors.append(f"{prefix}: missing evidence reference: {reference}")

    promoted = {
        record.technique_id for record in records if record.status == "PROMOTED"
    }
    if "system.champion_v1" not in promoted:
        errors.append("the current champion is not registered as PROMOTED")
    return errors


def main() -> None:
    records = TechniqueDecisionStore(LEDGER).read()
    errors = validate_records(records)
    summary = {
        "ledger": str(LEDGER.relative_to(ROOT)),
        "record_count": len(records),
        "status_counts": dict(sorted(Counter(item.status for item in records).items())),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
