from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

from scripts.fetch_dense_index_asset import (
    extract_archive,
    sha256_file,
    verify_dense_indexes,
)


def _fixture_manifest(root: Path) -> dict[str, object]:
    relative_dir = Path("artifacts/cache/dense")
    matrix_relative = relative_dir / "test-model-catalog-spec.npy"
    metadata_relative = relative_dir / "test-model-catalog-spec.json"
    matrix_path = root / matrix_relative
    metadata_path = root / metadata_relative
    matrix_path.parent.mkdir(parents=True)
    np.save(matrix_path, np.eye(3, dtype=np.float32), allow_pickle=False)
    metadata = {
        "catalog_sha256": "catalog-sha",
        "document_format_version": "test-v1",
        "dtype": "float32",
        "embedding_dimension": 3,
        "identifiers_sha256": "identifiers-sha",
        "model": {
            "embedding_dimension": 3,
            "key": "test_model",
            "model_name": "example/test-model",
            "revision": "model-revision",
        },
        "model_spec_sha256": "model-spec-sha",
        "normalized": True,
        "row_count": 3,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "archive": {"bytes": 0, "sha256": ""},
        "asset_id": "test_dense_asset",
        "indexes": [
            {
                **{key: value for key, value in metadata.items() if key != "model"},
                "key": "test_model",
                "matrix": {
                    "bytes": matrix_path.stat().st_size,
                    "path": matrix_relative.as_posix(),
                    "sha256": sha256_file(matrix_path),
                },
                "metadata": {
                    "bytes": metadata_path.stat().st_size,
                    "path": metadata_relative.as_posix(),
                    "sha256": sha256_file(metadata_path),
                },
                "model_name": "example/test-model",
                "model_revision": "model-revision",
            }
        ],
    }


def test_verify_and_extract_dense_index_archive(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    manifest = _fixture_manifest(source_root)
    verified = verify_dense_indexes(manifest, source_root)  # type: ignore[arg-type]
    assert verified["index_count"] == 1

    archive_path = tmp_path / "dense.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        index = manifest["indexes"][0]  # type: ignore[index]
        for kind in ("matrix", "metadata"):
            relative = Path(index[kind]["path"])  # type: ignore[index]
            archive.add(source_root / relative, arcname=relative.as_posix())
    manifest["archive"] = {
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }

    installed = extract_archive(
        manifest, archive_path, destination_root  # type: ignore[arg-type]
    )
    assert installed["indexes"] == [
        {
            "embedding_dimension": 3,
            "key": "test_model",
            "model_revision": "model-revision",
            "row_count": 3,
        }
    ]
    verify_dense_indexes(manifest, destination_root)  # type: ignore[arg-type]
