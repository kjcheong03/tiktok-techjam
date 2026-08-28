from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODE_TEMPLATES = {
    "full": "configs/campaigns/adaptive_autonomous_full_v1.template.json",
    "discover": "configs/campaigns/adaptive_autonomous_discovery_v1.template.json",
    "augment": "configs/campaigns/adaptive_autonomous_augment_v1.template.json",
}


def _run(*arguments: str) -> None:
    print("+", " ".join(("uv", "run", "python", *arguments)), flush=True)
    try:
        subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"stage failed with exit code {error.returncode}; see the error above"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, freeze, resume, validate, and package one autonomous campaign"
        )
    )
    parser.add_argument("--mode", choices=tuple(MODE_TEMPLATES), default="full")
    parser.add_argument(
        "--template",
        help="Advanced override; otherwise --mode selects a versioned template",
    )
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--f1-candidates", type=int, default=24)
    parser.add_argument("--f2-candidates", type=int, default=6)
    parser.add_argument("--hpo-trials", type=int, default=8)
    parser.add_argument("--higher-order-rounds", type=int, default=2)
    args = parser.parse_args()

    template_argument = args.template or MODE_TEMPLATES[args.mode]
    template = ROOT / template_argument
    template_payload = json.loads(template.read_text(encoding="utf-8"))
    campaign_id = str(template_payload["campaign_id"])
    campaign_root = Path("artifacts/campaigns") / campaign_id
    manifest = campaign_root / "manifest.json"
    checkpoint = campaign_root / "checkpoint.json"
    evidence = campaign_root / "evidence.json"
    plan = campaign_root / "plan.json"
    proposals = Path("artifacts/proposals") / campaign_id

    preflight = [
        "-m",
        "scripts.preflight_autonomous",
        "--template",
        template_argument,
        "--output",
        (campaign_root / "admission.json").as_posix(),
    ]
    if args.prepare_assets:
        preflight.append("--prepare-assets")
    if args.mode == "augment" and args.template is None:
        preflight.append("--allow-unselected")
    _run(*preflight)

    manifest_exists = (ROOT / manifest).is_file()
    if not manifest_exists:
        if args.resume:
            raise FileNotFoundError("--resume requires an existing frozen manifest")
        _run(
            "-m",
            "scripts.freeze_wave2_campaign",
            "--template",
            template_argument,
            "--output",
            manifest.as_posix(),
        )
    else:
        print(f"resuming frozen campaign from {manifest}", flush=True)

    _run(
        "-m",
        "scripts.plan_wave2_campaign",
        "--manifest",
        manifest.as_posix(),
        "--output",
        plan.as_posix(),
    )
    _run(
        "-m",
        "scripts.run_autonomous_campaign",
        "--manifest",
        manifest.as_posix(),
        "--checkpoint",
        checkpoint.as_posix(),
        "--evidence",
        evidence.as_posix(),
        "--f1-candidates",
        str(args.f1_candidates),
        "--f2-candidates",
        str(args.f2_candidates),
        "--hpo-trials-per-structure",
        str(args.hpo_trials),
        "--higher-order-rounds",
        str(args.higher_order_rounds),
    )
    campaign_evidence = json.loads((ROOT / evidence).read_text(encoding="utf-8"))
    confirmed = campaign_evidence.get("confirmed_top3", [])
    if len(confirmed) != 3:
        print(
            json.dumps(
                {
                    "automatic_promotion": False,
                    "campaign_complete": True,
                    "campaign_id": campaign_id,
                    "proposal_count": 0,
                    "safe_candidates_before_packaging": len(confirmed),
                    "retain_current_champion": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    proposal_baseline = str(confirmed[0]["baseline_id"])
    _run(
        "-m",
        "scripts.materialize_campaign_top_three",
        "--manifest",
        manifest.as_posix(),
        "--evidence",
        evidence.as_posix(),
        "--checkpoint",
        checkpoint.as_posix(),
        "--baseline-id",
        proposal_baseline,
        "--output",
        proposals.as_posix(),
    )
    manifest_path = ROOT / proposals / "proposal_manifest.json"
    proposal_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = [
        "uv run python -m scripts.prepare_candidate --preset "
        + str(record["preset"]["path"])
        for record in proposal_manifest["candidates"]
    ]
    print(
        json.dumps(
            {
                "campaign_complete": True,
                "campaign_id": campaign_id,
                "proposal_manifest": manifest_path.relative_to(ROOT).as_posix(),
                "prepare_commands": commands,
                "automatic_promotion": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
