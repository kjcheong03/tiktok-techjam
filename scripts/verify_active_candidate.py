from __future__ import annotations

import json

from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.runtime.selected import ACTIVE_POINTER, resolve_active_preset


def main() -> None:
    preset = resolve_active_preset(ACTIVE_POINTER)
    if preset is None:
        raise SystemExit(
            "no active pointer; starter.Agent uses compiled champion fallback"
        )
    payload = json.loads(preset.read_text(encoding="utf-8"))
    architecture: str
    if payload.get("architecture") == "adaptive_hybrid_1a_3b_v1":
        config = AdaptiveArchitectureAudit.validate(
            load_adaptive_hybrid_config(preset)
        )
        identifier = config.policy_id
        architecture = config.architecture
    else:
        suite = load_suite_config(preset)
        identifier = suite.experiment_id
        architecture = "legacy_suite"
    print(
        json.dumps(
            {
                "verified": True,
                "candidate_id": identifier,
                "architecture": architecture,
                "preset": str(preset),
                "starter_entrypoint": "starter.Agent",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
