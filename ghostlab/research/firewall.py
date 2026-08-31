from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

FORBIDDEN_RUNTIME_NAMES = frozenset(
    {
        "ground_truth",
        "target",
        "target_id",
        "intent_card",
        "behavior",
        "scenario_type",
        "difficulty_bucket",
        "disclosed",
        "reward",
        "label",
    }
)


def reject_forbidden_names(names: Iterable[str]) -> None:
    forbidden = {name for name in names if name.casefold() in FORBIDDEN_RUNTIME_NAMES}
    if forbidden:
        raise ValueError(
            f"research-only fields crossed runtime boundary: {sorted(forbidden)}"
        )


def runtime_profile(sample: Mapping[str, object]) -> dict:
    profile = sample.get("user_profile")
    if not isinstance(profile, dict):
        raise TypeError("sample user_profile must be an object")
    reject_forbidden_names(profile)
    return json.loads(json.dumps(profile))


def session_set_hash(sample_ids: Iterable[str]) -> str:
    serialized = "\n".join(sorted(sample_ids))
    return hashlib.sha256(serialized.encode()).hexdigest()
