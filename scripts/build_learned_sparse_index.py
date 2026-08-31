from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.retrieval import catalog_document
from ghostlab.retrieval.learned_sparse import (
    InvertedSparseIndex,
    LearnedSparseAsset,
    SpladeEncoder,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local SPLADE sparse index")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-template", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    template = LearnedSparseAsset.load(args.manifest_template)
    identifiers: list[str] = []
    documents: list[str] = []
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            identifiers.append(str(product["parent_asin"]))
            documents.append(catalog_document(product))
    encoder = SpladeEncoder(
        args.model, max_terms=template.max_terms, batch_size=args.batch_size
    )
    vectors = encoder.encode(documents)
    InvertedSparseIndex.from_vectors(identifiers, vectors).save(args.index)
    output = {
        **template.__dict__,
        "availability": "built_unvalidated",
        "model_path": str(args.model),
        "index_path": str(args.index),
        "index_sha256": sha256_file(args.index),
        "catalog_sha256": sha256_file(args.catalog),
        "unavailable_reason": "Recall/runtime/license gates have not yet passed.",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
