from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CampaignManifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "freeze requires a clean worktree so HEAD contains every input"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a Wave 2 campaign template")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require_clean_worktree()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    catalog_path = PROJECT_ROOT / template.pop("catalog_path")
    adaptive_path = PROJECT_ROOT / template.pop("adaptive_split_path")
    nested_path = PROJECT_ROOT / template.pop("nested_split_path")
    adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
    manifest = CampaignManifest(
        **template,
        parent_commit=git_head(),
        catalog_hash=load_catalog(catalog_path).content_hash,
        dataset_hash=adaptive["dataset_sha256"],
        adaptive_split_hash=sha256_file(adaptive_path),
        nested_split_hash=sha256_file(nested_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
