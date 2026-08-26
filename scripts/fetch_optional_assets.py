from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "minilm": ROOT / "configs/assets/minilm_control.json",
    "e5": ROOT / "configs/assets/e5_small_v2.json",
    "cross_encoder": ROOT / "configs/assets/cross_encoder_minilm.json",
}
RECEIPT = ".ghostlab_asset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest: dict[str, object], destination: Path) -> dict[str, object]:
    expected_files = manifest.get("files")
    if isinstance(expected_files, dict):
        errors: list[str] = []
        for relative, expected_value in expected_files.items():
            expected = dict(expected_value)
            path = destination / relative
            if not path.is_file():
                errors.append(f"missing {relative}")
            elif path.stat().st_size != int(expected["bytes"]):
                errors.append(f"size mismatch {relative}")
            elif sha256_file(path) != str(expected["sha256"]):
                errors.append(f"hash mismatch {relative}")
        if errors:
            raise RuntimeError("; ".join(errors))
        return {"verification": "per-file-sha256", "files": len(expected_files)}
    receipt_path = destination / RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError(f"missing acquisition receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in ("model_name", "revision"):
        if receipt.get(key) != manifest.get(key):
            raise RuntimeError(f"asset receipt {key} mismatch")
    return {"verification": "pinned-revision-receipt", "files": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch or verify pinned optional GhostLab model assets"
    )
    parser.add_argument("asset", choices=sorted(MANIFESTS))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--endpoint")
    args = parser.parse_args()

    manifest = json.loads(MANIFESTS[args.asset].read_text(encoding="utf-8"))
    destination = ROOT / str(manifest["destination"])
    if not args.verify_only:
        if args.endpoint:
            os.environ["HF_ENDPOINT"] = args.endpoint
        from huggingface_hub import snapshot_download

        expected_files = manifest.get("files")
        snapshot_download(
            repo_id=str(manifest["model_name"]),
            revision=str(manifest["revision"]),
            local_dir=destination,
            allow_patterns=(
                sorted(expected_files) if isinstance(expected_files, dict) else None
            ),
        )
        if not isinstance(expected_files, dict):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / RECEIPT).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_name": manifest["model_name"],
                        "revision": manifest["revision"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    result = verify(manifest, destination)
    print(
        json.dumps(
            {
                "asset": args.asset,
                "destination": str(destination.relative_to(ROOT)),
                **result,
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
