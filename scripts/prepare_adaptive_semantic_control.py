from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config
from scripts.fetch_optional_assets import verify as verify_asset

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "smollm2-1.7b-instruct"
MODEL_PATH = "artifacts/cache/models/smollm2-1.7b-instruct"
MODEL_MANIFEST = "configs/assets/smollm2_1_7b_instruct.json"
MODEL_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
CONTROL_WEIGHT = 0.05
CONTROL_DEPTH = 10


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if ".cache" in item.parts:
            continue
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def prepare_control(input_path: Path, output_path: Path) -> dict[str, object]:
    manifest_path = ROOT / MODEL_MANIFEST
    model_path = ROOT / MODEL_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = verify_asset(manifest, model_path)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"SmolLM2 config is missing: {model_path / 'config.json'}"
        )
    if not any(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"SmolLM2 weights are missing under {model_path}")

    source = load_adaptive_hybrid_config(input_path)
    semantic = source.semantic_ranker.model_copy(
        update={
            "backend": "local_causal_relevance",
            "model_id": MODEL_ID,
            "model_path": MODEL_PATH,
            "model_revision": MODEL_REVISION,
            "model_sha256": _tree_sha256(model_path),
            "activation_policy": "browsing_only",
            "activate_for_browsing": True,
            "weight": CONTROL_WEIGHT,
            "rerank_k": CONTROL_DEPTH,
        }
    )
    control = source.model_copy(
        update={
            "policy_id": f"{source.policy_id}_smollm2_control",
            "semantic_ranker": semantic,
        }
    )
    control = AdaptiveArchitectureAudit.validate(
        type(source).model_validate(control.model_dump())
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(control.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "mode": "fixed_semantic_control",
        "development_only": True,
        "protected_holdout_accessed": False,
        "architecture": control.architecture,
        "input_config": str(input_path.relative_to(ROOT)),
        "output_config": str(output_path.relative_to(ROOT)),
        "output_config_sha256": control.canonical_hash(),
        "semantic_policy": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": semantic.model_sha256,
            "activation_policy": semantic.activation_policy,
            "weight": CONTROL_WEIGHT,
            "depth": CONTROL_DEPTH,
            "fallback_model": semantic.fallback_model_path,
        },
        "asset_verification": verification,
        "ghostlab_semantic_grid": {
            "f0_depth": 10,
            "weights": [0.05, 0.10, 0.15, 0.20],
            "f1_survivor_depth": 20,
        },
        "model_family_search_reopened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the fixed SmolLM2 semantic control before GhostLab racing"
    )
    parser.add_argument(
        "--input", default="configs/adaptive_hybrid_1a_3b_1650_final_v1.json"
    )
    parser.add_argument(
        "--output",
        default="configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json",
    )
    parser.add_argument(
        "--report", default="artifacts/reports/adaptive_semantic_control_v1.json"
    )
    args = parser.parse_args()
    report = prepare_control(ROOT / args.input, ROOT / args.output)
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
