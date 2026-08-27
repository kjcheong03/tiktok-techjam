from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ghostlab.campaign.models import Availability, ResourceRequest, TechniqueSpec


@dataclass(frozen=True)
class TechniqueCatalog:
    schema_version: int
    techniques: dict[str, TechniqueSpec]
    content_hash: str

    def available(self) -> tuple[TechniqueSpec, ...]:
        return tuple(
            self.techniques[key]
            for key in sorted(self.techniques)
            if self.techniques[key].executable
        )

    def unavailable_reasons(self, technique_ids: tuple[str, ...]) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for technique_id in technique_ids:
            technique = self.techniques.get(technique_id)
            if technique is None:
                reasons[technique_id] = "unknown_technique"
            elif not technique.executable:
                reasons[technique_id] = f"availability={technique.availability}"
        return reasons


def _v1_availability(status: str) -> Availability:
    if status in {"selected", "original_champion"}:
        return "selected"
    if status.startswith("parked") or status == "fallback":
        return "parked"
    if status == "interaction_reserve":
        return "interaction_reserve"
    return "available"


def _load_v1(path: Path) -> dict[str, TechniqueSpec]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, TechniqueSpec] = {}
    for item in value["techniques"]:
        result[item["id"]] = TechniqueSpec(
            id=item["id"],
            family=item["family"],
            wave=1,
            availability=_v1_availability(item["status"]),
            default_enabled=False,
            source=item.get("source"),
            execution_class=item.get("extra", "core"),
            resources=ResourceRequest(
                heavy_model=item.get("extra") in {"neural", "all"}
            ),
        )
    return result


def load_catalog(path: str | Path) -> TechniqueCatalog:
    catalog_path = Path(path)
    raw = catalog_path.read_bytes()
    value = json.loads(raw)
    version = int(value["schema_version"])
    if version not in {1, 2}:
        raise ValueError(f"unsupported catalog schema: {version}")
    techniques: dict[str, TechniqueSpec] = {}
    extends = value.get("extends")
    if extends is not None:
        parent = (catalog_path.parent / str(extends)).resolve()
        if parent.parent != catalog_path.parent.resolve():
            raise ValueError("catalog extension must stay in the catalog directory")
        techniques.update(_load_v1(parent))
    if version == 1:
        techniques.update(_load_v1(catalog_path))
    else:
        declared: set[str] = set()
        for item in value.get("techniques", []):
            spec = TechniqueSpec.model_validate(item)
            if spec.id in declared:
                raise ValueError(f"duplicate technique ID in catalog: {spec.id}")
            declared.add(spec.id)
            techniques[spec.id] = spec
    return TechniqueCatalog(
        schema_version=version,
        techniques=techniques,
        content_hash=hashlib.sha256(raw).hexdigest(),
    )
