from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from ghostlab.runtime.adaptive_components import DiverseDenseTrack
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from ghostlab.state.baseline_v2 import StateBaselineV2
from ghostlab.state.v2_view import V2SessionController

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS: dict[str, dict[str, object]] = {
    "warm_weather_wedding": {
        "request": (
            "comfortable breathable smart-casual outfit and accessories for a "
            "warm-weather wedding guest, avoid formalwear"
        ),
        "category_families": {
            "clothing": ("dress",),
            "footwear": ("sandal", "pump", "flat"),
            "accessory": ("handbag", "earring", "necklace", "bracelet"),
        },
    },
    "sunny_beach_holiday": {
        "request": (
            "things to wear and bring for a sunny beach holiday: swim, sun "
            "protection, and walking comfort"
        ),
        "category_families": {
            "swimwear": ("swimsuit", "cover up"),
            "footwear": ("sandal", "flip-flop"),
            "sun_accessory": ("sunglass", "hat", "cap"),
        },
    },
    "rainy_road_running": {
        "request": "gear for comfortable outdoor road running in cool rainy weather",
        "category_families": {
            "running": ("running", "athletic"),
            "outerwear": ("jacket", "coat", "vest"),
            "accessory": ("sock", "hat", "cap"),
        },
    },
}


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_adaptive_hybrid_config(
        ROOT / "configs/adaptive_hybrid_1a_3b_v1.json"
    )
    catalog_path = ROOT / "data/catalog.jsonl"
    categories: dict[str, str] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            # Ignore universal catalogue roots; validate specific category families.
            leaves = (product.get("categories") or [])[2:]
            categories[str(product["parent_asin"])] = " ".join(leaves).casefold()
    track = DiverseDenseTrack(
        catalog_path, config.browsing, project_root=ROOT
    )
    results: dict[str, object] = {}
    all_passed = True
    for name, definition in SCENARIOS.items():
        request = str(definition["request"])
        state = StateBaselineV2(name, {})
        state.observe(request, 1)
        view = V2SessionController(state).snapshot(query_text=request, turn=1)
        dense = track.search(view)
        families = cast(
            dict[str, tuple[str, ...]], definition["category_families"]
        )
        family_counts = {
            family: sum(
                any(token in categories[identifier] for token in tokens)
                for identifier in dense.identifiers
            )
            for family, tokens in families.items()
        }
        passed = all(count > 0 for count in family_counts.values())
        all_passed = all_passed and passed
        results[name] = {
            "candidate_count": len(dense.identifiers),
            "query_views": dense.query_views,
            "family_counts": family_counts,
            "all_expected_families_recovered": passed,
        }
    report = {
        "schema_version": 1,
        "validation_kind": "category_blind_cross_category_browsing",
        "uses_target_ids": False,
        "uses_official_scenario_labels": False,
        "all_scenarios_passed": all_passed,
        "scenarios": results,
    }
    output = ROOT / "artifacts/reports/cross_category_browsing_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
