from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ghostlab.campaign.bindings import TechniqueBindingRegistry
from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.interaction_search import (
    SearchLimits,
    plan_standalones_and_pairs,
)
from ghostlab.research.technique_suite import load_suite_config

AdmissionStatus = Literal[
    "admitted",
    "blocked_missing_asset",
    "blocked_unavailable",
    "anchor_only",
    "research_only",
    "runnable_not_selected",
]


class TechniqueAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    technique_id: str
    family: str
    status: AdmissionStatus
    selected_for_campaign: bool
    source: str | None
    binding_disposition: str
    description: str
    assets: tuple[str, ...] = ()
    missing_assets: tuple[str, ...] = ()
    minimum_trial: str


class AdmissionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    campaign_id: str
    template_path: str
    complete_catalog_accounting: bool
    campaign_ready: bool
    admitted_count: int
    blocked_count: int
    planned_structure_count: int
    materializable_structure_count: int
    blocked_structure_count: int
    admitted_without_trial: tuple[str, ...]
    records: tuple[TechniqueAdmission, ...]


def load_campaign_selection(template_path: Path) -> tuple[str, set[str]]:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    campaign_id = payload.get("campaign_id")
    technique_ids = payload.get("technique_ids")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise TypeError("campaign template requires campaign_id")
    if not isinstance(technique_ids, list) or not all(
        isinstance(item, str) for item in technique_ids
    ):
        raise TypeError("campaign template requires string technique_ids")
    return campaign_id, set(technique_ids)


def build_admission_report(
    *,
    project_root: Path,
    template_path: Path,
    catalog: TechniqueCatalog,
    registry: TechniqueBindingRegistry,
) -> AdmissionReport:
    """Account for every catalog entry and fail visibly on missing runtime assets."""

    root = project_root.resolve()
    campaign_id, selected = load_campaign_selection(template_path)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    presets = tuple(str(item) for item in template["baseline_presets"])
    modes = dict(template.get("baseline_search_modes", {}))
    discovery = tuple(
        preset for preset in presets if modes.get(preset, "composable") == "composable"
    )
    if len(discovery) != 1:
        raise ValueError("admission audit requires exactly one pure discovery anchor")
    baseline = tuple(str(item) for item in template["baseline_techniques"])
    composable_ids = tuple(
        item
        for item in template["technique_ids"]
        if item in registry.bindings
        and registry.bindings[item].disposition == "composable"
        and item in catalog.techniques
        and catalog.techniques[item].executable
    )
    candidate_budget = int(template["candidate_limit"]) - (len(presets) - 1)
    plan = plan_standalones_and_pairs(
        catalog,
        baseline_id=discovery[0],
        baseline_techniques=baseline,
        technique_ids=composable_ids,
        limits=SearchLimits(
            max_order=int(template["max_order"]),
            max_candidates=candidate_budget,
            max_wall_seconds=float(template["max_wall_seconds"]),
        ),
    )
    base_config = load_suite_config(root / discovery[0])
    runnable = []
    for candidate in plan.candidates:
        additions = tuple(item for item in candidate.techniques if item not in baseline)
        try:
            registry.materialize(
                base_config, candidate.model_copy(update={"techniques": additions})
            )
        except (TypeError, ValueError):
            continue
        runnable.append(candidate)
    covered = {item for candidate in runnable for item in candidate.techniques}
    records: list[TechniqueAdmission] = []
    for technique_id, technique in sorted(catalog.techniques.items()):
        binding = registry.bindings.get(technique_id)
        disposition = binding.disposition if binding is not None else "unavailable"
        assets = tuple(
            sorted({*technique.assets, *(binding.asset_paths if binding else ())})
        )
        missing = tuple(
            value for value in assets if not (root / value).resolve().exists()
        )
        is_selected = technique_id in selected
        if (
            disposition == "research_only"
            or technique.execution_mode == "research_only"
        ):
            status: AdmissionStatus = "research_only"
            trial = "workflow-level evaluation; not a runtime toggle"
        elif disposition == "anchor_only" or technique.execution_mode == "anchor_only":
            status = "anchor_only"
            trial = "complete-preset control; not an additive switch"
        elif disposition != "composable" or not technique.executable:
            status = "blocked_unavailable"
            trial = "not runnable until the recorded blocker is resolved"
        elif missing:
            status = "blocked_missing_asset"
            trial = "admitted after asset preparation and verification"
        elif not is_selected:
            status = "runnable_not_selected"
            trial = "must be added before campaign freeze"
        else:
            status = "admitted"
            eligible = [
                candidate
                for candidate in runnable
                if technique_id in candidate.techniques and technique_id not in baseline
            ]
            if eligible:
                best = min(
                    eligible, key=lambda item: (item.complexity, item.candidate_id)
                )
                trial = f"F0 {best.generation}: " + ", ".join(
                    item for item in best.techniques if item not in baseline
                )
            else:
                trial = "no materializable F0 structure found"
        records.append(
            TechniqueAdmission(
                technique_id=technique_id,
                family=technique.family,
                status=status,
                selected_for_campaign=is_selected,
                source=technique.source,
                binding_disposition=disposition,
                description=(
                    binding.reason if binding is not None else "missing binding"
                ),
                assets=assets,
                missing_assets=missing,
                minimum_trial=trial,
            )
        )
    runtime_omissions = [
        row for row in records if row.status == "runnable_not_selected"
    ]
    blockers = [
        row
        for row in records
        if row.selected_for_campaign
        and row.status in {"blocked_missing_asset", "blocked_unavailable"}
    ]
    admitted = [row for row in records if row.status == "admitted"]
    admitted_without_trial = tuple(
        sorted(
            row.technique_id
            for row in admitted
            if row.technique_id not in covered and row.technique_id not in baseline
        )
    )
    return AdmissionReport(
        campaign_id=campaign_id,
        template_path=template_path.relative_to(root).as_posix(),
        complete_catalog_accounting=(len(records) == len(catalog.techniques)),
        campaign_ready=(
            not blockers and not runtime_omissions and not admitted_without_trial
        ),
        admitted_count=len(admitted),
        blocked_count=len(blockers),
        planned_structure_count=len(plan.candidates),
        materializable_structure_count=len(runnable),
        blocked_structure_count=len(plan.candidates) - len(runnable),
        admitted_without_trial=admitted_without_trial,
        records=tuple(records),
    )
