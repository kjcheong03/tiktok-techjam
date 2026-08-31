from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ghostlab.campaign.admission import build_admission_report
from ghostlab.campaign.bindings import default_binding_registry
from ghostlab.campaign.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "configs/campaigns/autonomous_state_v2_v1.template.json"
DEFAULT_REPORT = ROOT / "artifacts/campaigns/autonomous_state_v2_v1/admission.json"


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def prepare_assets() -> None:
    ontology = ROOT / "artifacts/assets/catalog_ontology_v1.json"
    if not ontology.is_file():
        _run(
            "-m",
            "scripts.build_attribute_ontology",
            "--output",
            "artifacts/assets/catalog_ontology_v1.json",
        )
    for asset in ("minilm", "e5", "cross_encoder"):
        _run("-m", "scripts.fetch_optional_assets", asset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare assets and account for every autonomous technique"
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument(
        "--allow-unselected",
        action="store_true",
        help="Allow an explicitly targeted campaign to omit otherwise runnable techniques",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.prepare_assets:
        prepare_assets()
    template = args.template.resolve()
    catalog_path = (
        ROOT / json.loads(template.read_text(encoding="utf-8"))["catalog_path"]
    )
    report = build_admission_report(
        project_root=ROOT,
        template_path=template,
        catalog=load_catalog(catalog_path),
        registry=default_binding_registry(),
        require_all_runnable=not args.allow_unselected,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.verbose:
        print(report.model_dump_json(indent=2))
    else:
        print(
            json.dumps(
                {
                    "campaign_id": report.campaign_id,
                    "campaign_ready": report.campaign_ready,
                    "admitted_count": report.admitted_count,
                    "blocked_count": report.blocked_count,
                    "planned_structure_count": report.planned_structure_count,
                    "materializable_structure_count": (
                        report.materializable_structure_count
                    ),
                    "admitted_without_trial": report.admitted_without_trial,
                    "report": str(output.relative_to(ROOT)),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if not report.campaign_ready and not args.allow_blocked:
        raise SystemExit(
            "preflight failed: resolve the blockers recorded in " + str(output)
        )


if __name__ == "__main__":
    main()
