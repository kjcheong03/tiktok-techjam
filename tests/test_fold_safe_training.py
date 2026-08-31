from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ghostlab.training.protocol import FitRequest
from ghostlab.training.residual import MeanProbabilityModel, _fit_probability_model


def test_fit_request_rejects_session_leakage() -> None:
    with pytest.raises(ValidationError, match="leakage"):
        FitRequest(
            technique_id="ranking.test",
            outer_fold=0,
            inner_fold=1,
            train_sample_ids=("a", "b"),
            validation_sample_ids=("b", "c"),
            seed=7,
        )


def test_fit_request_accepts_disjoint_sessions() -> None:
    request = FitRequest(
        technique_id="ranking.test",
        outer_fold=0,
        inner_fold=1,
        train_sample_ids=("a", "b"),
        validation_sample_ids=("c",),
        seed=7,
    )
    assert request.train_sample_ids == ("a", "b")


@pytest.mark.parametrize(
    ("variant", "ensemble"),
    (
        ("regularized_logistic", False),
        ("hist_gbdt_d2_lr005", False),
        ("ensemble_logistic_gbdt_d3_lr01", True),
    ),
)
def test_residual_model_variants_fit_probability_contract(
    variant: str, ensemble: bool
) -> None:
    features = np.asarray(
        [[index, index % 3, index / 10] for index in range(40)], dtype=np.float64
    )
    labels = np.asarray([index % 2 for index in range(40)], dtype=np.int64)
    weights = np.ones(40, dtype=np.float64)

    model = _fit_probability_model(
        variant,
        features,
        labels,
        weights,
        regularization=0.2,
        seed=7,
    )
    probabilities = model.predict_proba(features)  # type: ignore[attr-defined]

    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert isinstance(model, MeanProbabilityModel) is ensemble
