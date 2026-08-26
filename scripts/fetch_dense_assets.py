from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ghostlab.retrieval.dense import sha256_file

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/assets/e5_small_v2.json"


def verify_asset(manifest: dict[str, object], destination: Path) -> None:
    files = manifest["files"]
    if not isinstance(files, dict):
        raise TypeError("asset manifest files must be an object")
    errors = []
    for relative, expected_value in files.items():
        expected = dict(expected_value)  # type: ignore[arg-type]
        path = destination / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        if path.stat().st_size != int(expected["bytes"]):
            errors.append(f"size mismatch {relative}")
        if sha256_file(path) != str(expected["sha256"]):
            errors.append(f"hash mismatch {relative}")
    if errors:
        raise RuntimeError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the pinned E5 asset before offline evaluation"
    )
    parser.add_argument("--endpoint")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    destination = ROOT / str(manifest["destination"])
    if not args.verify_only:
        if args.endpoint:
            os.environ["HF_ENDPOINT"] = args.endpoint
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(manifest["model_name"]),
            revision=str(manifest["revision"]),
            local_dir=destination,
            allow_patterns=sorted(dict(manifest["files"])),
        )
    verify_asset(manifest, destination)
    print(
        json.dumps(
            {
                "asset_id": manifest["asset_id"],
                "destination": str(destination.relative_to(ROOT)),
                "file_count": len(dict(manifest["files"])),
                "total_bytes": manifest["total_bytes"],
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
