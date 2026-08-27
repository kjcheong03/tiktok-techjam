from __future__ import annotations

import json

from ghostlab.research.technique_suite import load_suite_config
from ghostlab.runtime.selected import ACTIVE_POINTER, resolve_active_preset


def main() -> None:
    preset = resolve_active_preset(ACTIVE_POINTER)
    if preset is None:
        raise SystemExit(
            "no active pointer; starter.Agent uses compiled champion fallback"
        )
    config = load_suite_config(preset)
    print(
        json.dumps(
            {
                "verified": True,
                "experiment_id": config.experiment_id,
                "preset": str(preset),
                "starter_entrypoint": "starter.Agent",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
