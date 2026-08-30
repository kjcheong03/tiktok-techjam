from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ghostlab.training.adaptive_datasets import (
    load_adaptive_training_corpus,
    progressive_stratified_samples,
    stratified_outer_folds,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "data/public_set.jsonl",
    "data/synthetic_1000_public_like.jsonl",
    "data/independent_template_1000.jsonl",
)


def test_real_adaptive_training_corpus_has_2200_unique_samples_and_targets() -> None:
    corpus = load_adaptive_training_corpus(ROOT, DATASETS)
    targets = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in corpus.samples.values()
    }
    assert len(corpus.samples) == 2200
    assert len(targets) == 2200
    assert [source.sample_count for source in corpus.sources] == [200, 1000, 1000]


def test_nested_folds_are_complete_disjoint_and_source_scenario_stratified() -> None:
    corpus = load_adaptive_training_corpus(ROOT, DATASETS)
    folds = stratified_outer_folds(corpus, fold_count=5, seed=20260830)
    assert len(folds) == 5
    assert set().union(*(set(fold) for fold in folds)) == corpus.sample_ids
    assert sum(len(fold) for fold in folds) == 2200
    assert all(len(fold) == 440 for fold in folds)
    distributions = []
    for fold in folds:
        distributions.append(
            Counter(
                (
                    corpus.origins[sample_id],
                    str(corpus.samples[sample_id]["scenario_type"]),
                )
                for sample_id in fold
            )
        )
    assert all(item == distributions[0] for item in distributions[1:])


def test_progressive_order_gives_nested_balanced_fidelity_prefixes() -> None:
    corpus = load_adaptive_training_corpus(ROOT, DATASETS)
    ordered = progressive_stratified_samples(corpus, seed=20260826)
    assert len(ordered) == 2200
    for count in (440, 1100, 2200):
        scenarios = Counter(str(item["scenario_type"]) for item in ordered[:count])
        assert scenarios == {
            "buying": round(count * 0.4),
            "browsing": round(count * 0.4),
            "intent_override": round(count * 0.15),
            "boundary": round(count * 0.05),
        }


def test_loader_rejects_duplicate_targets_across_sources(tmp_path: Path) -> None:
    def write(name: str, sample_id: str) -> str:
        path = tmp_path / name
        payload = {
            "sample_id": sample_id,
            "scenario_type": "buying",
            "ground_truth": {"parent_asin": "same-target"},
            "user_profile": {},
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return str(path)

    with pytest.raises(ValueError, match="duplicate target"):
        load_adaptive_training_corpus(
            ROOT,
            (write("first.jsonl", "first"), write("second.jsonl", "second")),
        )
