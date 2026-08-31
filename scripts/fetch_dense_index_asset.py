from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/assets/dense_indexes_50k_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {path}")
    expected_bytes = int(expected["bytes"])
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != str(expected["sha256"]):
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: {actual_sha256} != {expected['sha256']}"
        )


def expected_archive_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for index in manifest["indexes"]:
        for kind in ("matrix", "metadata"):
            path = str(index[kind]["path"])
            if path in paths:
                raise RuntimeError(f"duplicate dense-index manifest path: {path}")
            paths.add(path)
    return paths


def verify_dense_indexes(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    for index in manifest["indexes"]:
        matrix_path = root / str(index["matrix"]["path"])
        metadata_path = root / str(index["metadata"]["path"])
        _verify_file(matrix_path, dict(index["matrix"]))
        _verify_file(metadata_path, dict(index["metadata"]))

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scalar_fields = (
            "catalog_sha256",
            "document_format_version",
            "dtype",
            "embedding_dimension",
            "identifiers_sha256",
            "model_spec_sha256",
            "normalized",
            "row_count",
        )
        for field in scalar_fields:
            if metadata.get(field) != index.get(field):
                raise RuntimeError(
                    f"{index['key']} metadata {field} mismatch: "
                    f"{metadata.get(field)!r} != {index.get(field)!r}"
                )
        model = dict(metadata.get("model", {}))
        model_expectations = {
            "key": index["key"],
            "model_name": index["model_name"],
            "revision": index["model_revision"],
            "embedding_dimension": index["embedding_dimension"],
        }
        for field, expected in model_expectations.items():
            if model.get(field) != expected:
                raise RuntimeError(
                    f"{index['key']} model {field} mismatch: "
                    f"{model.get(field)!r} != {expected!r}"
                )

        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (
            int(index["row_count"]),
            int(index["embedding_dimension"]),
        )
        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"{index['key']} shape mismatch: {matrix.shape} != {expected_shape}"
            )
        if matrix.dtype.name != str(index["dtype"]):
            raise RuntimeError(
                f"{index['key']} dtype mismatch: {matrix.dtype.name} != {index['dtype']}"
            )
        verified.append(
            {
                "embedding_dimension": matrix.shape[1],
                "key": index["key"],
                "model_revision": index["model_revision"],
                "row_count": matrix.shape[0],
            }
        )
    return {"index_count": len(verified), "indexes": verified}


def verify_archive(manifest: dict[str, Any], archive_path: Path) -> None:
    _verify_file(archive_path, dict(manifest["archive"]))
    expected = expected_archive_paths(manifest)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        actual = {member.name for member in members}
        if actual != expected:
            raise RuntimeError(
                f"archive membership mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        if any(not member.isfile() for member in members):
            raise RuntimeError("dense-index archive may contain regular files only")


def extract_archive(
    manifest: dict[str, Any], archive_path: Path, destination_root: Path
) -> dict[str, Any]:
    verify_archive(manifest, archive_path)
    with tempfile.TemporaryDirectory(prefix="ghostlab-dense-extract-") as temporary:
        staging_root = Path(temporary)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                target = staging_root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                archive_source = archive.extractfile(member)
                if archive_source is None:
                    raise RuntimeError(f"could not read archive member {member.name}")
                with archive_source, target.open("wb") as output:
                    shutil.copyfileobj(archive_source, output)

        result = verify_dense_indexes(manifest, staging_root)
        for relative in sorted(expected_archive_paths(manifest)):
            staged_source = staging_root / relative
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_source, destination)
    return result


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "GhostLab-setup/1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the released GhostLab 50k dense indexes"
    )
    parser.add_argument("--archive", type=Path, help="use an existing local archive")
    parser.add_argument("--destination-root", type=Path, default=ROOT)
    parser.add_argument("--url", help="override the manifest download URL")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    destination_root = args.destination_root.resolve()
    if args.verify_only:
        result = verify_dense_indexes(manifest, destination_root)
        action = "verified"
    else:
        with tempfile.TemporaryDirectory(prefix="ghostlab-dense-download-") as temporary:
            if args.archive:
                archive_path = args.archive.resolve()
            else:
                archive_path = Path(temporary) / str(manifest["asset_name"])
                download(str(args.url or manifest["download_url"]), archive_path)
            result = extract_archive(manifest, archive_path, destination_root)
        action = "installed"

    print(
        json.dumps(
            {
                "action": action,
                "asset_id": manifest["asset_id"],
                "destination": str(destination_root / "artifacts/cache/dense"),
                "verified": True,
                **result,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
