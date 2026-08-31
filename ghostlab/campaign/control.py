from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class CampaignControl(BaseModel):
    """Atomic operator requests consumed at safe campaign stage boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    skip_hpo: bool = False
    requested_at: str | None = None


def load_campaign_control(path: Path) -> CampaignControl:
    if not path.exists():
        return CampaignControl()
    return CampaignControl.model_validate_json(path.read_text(encoding="utf-8"))


def save_campaign_control(path: Path, control: CampaignControl) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(control.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def request_skip_hpo(path: Path) -> CampaignControl:
    current = load_campaign_control(path)
    updated = current.model_copy(
        update={
            "skip_hpo": True,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_campaign_control(path, updated)
    return updated
