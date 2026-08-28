"""Fold-safe contracts for campaign-fitted components."""

from ghostlab.training.protocol import (
    FitReceipt,
    FitRequest,
    FoldSafeTrainer,
    assert_disjoint_fit,
)

__all__ = [
    "FitReceipt",
    "FitRequest",
    "FoldSafeTrainer",
    "assert_disjoint_fit",
]
