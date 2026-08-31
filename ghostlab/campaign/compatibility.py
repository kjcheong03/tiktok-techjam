from __future__ import annotations

from dataclasses import dataclass

from ghostlab.campaign.catalog import TechniqueCatalog


@dataclass(frozen=True)
class CompatibilityResult:
    valid: bool
    reasons: tuple[str, ...]


def validate_techniques(
    catalog: TechniqueCatalog, technique_ids: tuple[str, ...]
) -> CompatibilityResult:
    reasons: list[str] = []
    selected = set(technique_ids)
    if len(selected) != len(technique_ids):
        reasons.append("duplicate technique ID")
    groups: dict[str, list[str]] = {}
    for technique_id in sorted(selected):
        technique = catalog.techniques.get(technique_id)
        if technique is None:
            reasons.append(f"unknown technique: {technique_id}")
            continue
        if not technique.executable:
            reasons.append(
                f"unavailable technique: {technique_id} ({technique.availability})"
            )
        missing = sorted(set(technique.requires) - selected)
        if missing:
            reasons.append(f"{technique_id} missing requirements: {', '.join(missing)}")
        conflicts = sorted(set(technique.conflicts) & selected)
        if conflicts:
            reasons.append(f"{technique_id} conflicts with: {', '.join(conflicts)}")
        if technique.exclusive_group is not None:
            groups.setdefault(technique.exclusive_group, []).append(technique_id)
    for group, members in sorted(groups.items()):
        if len(members) > 1:
            reasons.append(f"exclusive group {group}: {', '.join(sorted(members))}")
    return CompatibilityResult(not reasons, tuple(reasons))
