from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.retrieval import catalog_document
from ghostlab.retrieval.late_interaction import (
    TokenEmbeddingStore,
    TransformerTokenEncoder,
    load_feasibility_manifest,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local reference late-interaction token index"
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-template", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    template = load_feasibility_manifest(args.manifest_template)
    identifiers: list[str] = []
    documents: list[str] = []
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            identifiers.append(str(product["parent_asin"]))
            documents.append(catalog_document(product))
    max_length = template.get("max_length", 128)
    if not isinstance(max_length, int):
        raise TypeError("manifest max_length must be an integer")
    encoder = TransformerTokenEncoder(
        args.model,
        max_length=max_length,
        batch_size=args.batch_size,
    )
    TokenEmbeddingStore.from_documents(
        identifiers, encoder.encode_tokens(documents)
    ).save(args.index)
    output = {
        **template,
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
