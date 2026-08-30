from __future__ import annotations

import json
from pathlib import Path

from ghostlab.optimization.racing import lineage_cluster_means, racing_decide
from ghostlab.training.adaptive_datasets import load_adaptive_training_corpus
from ghostlab.training.adaptive_lineage import (
    build_lineage_manifest,
    load_lineage_manifest,
    manifest_outer_folds,
    subset_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def test_lineage_manifest_has_exact_group_safe_partitions_and_folds(
    tmp_path: Path,
) -> None:
    corpus = load_adaptive_training_corpus(ROOT, DATASETS)
    payload, audit = build_lineage_manifest(corpus, seed=20260831, fold_count=5)
    assert audit["status"] == "verified"
    assert audit["candidate_group_count"] == 400
    assert len(payload["partitions"]["development"]["sample_ids"]) == 1650
    assert len(payload["partitions"]["holdout"]["sample_ids"]) == 550
    assert payload["partitions"]["development"]["source_counts"] == {
        "data/independent_template_1000.jsonl": 750,
        "data/public_set.jsonl": 150,
        "data/synthetic_1000_public_like.jsonl": 750,
    }
    assert payload["partitions"]["holdout"]["source_counts"] == {
        "data/independent_template_1000.jsonl": 250,
        "data/public_set.jsonl": 50,
        "data/synthetic_1000_public_like.jsonl": 250,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_lineage_manifest(path, corpus)
    development = subset_corpus(corpus, manifest, "development")
    folds = manifest_outer_folds(manifest)
    assert set().union(*(set(fold) for fold in folds)) == development.sample_ids
    group_by_sample = manifest.group_by_sample
    owners: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for sample_id in fold:
            owner = owners.setdefault(group_by_sample[sample_id], index)
            assert owner == index


def test_lineage_cluster_statistics_do_not_treat_variants_as_independent() -> None:
    values = [1.0] * 5 + [-1.0]
    clusters = ["related"] * 5 + ["independent"]
    assert lineage_cluster_means(values, clusters) == [-1.0, 1.0]
    assert (
        racing_decide(
            values,
            fidelity="f2",
            cluster_ids=clusters,
            material_delta=0.01,
        )
        == "HOLD_MORE_DATA"
    )
