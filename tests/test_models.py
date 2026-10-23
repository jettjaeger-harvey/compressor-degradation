"""Tests for degradation.models."""

import numpy as np
import pandas as pd
import pytest

from degradation.models import build_models, cross_validate_models, regression_metrics
from degradation.splitting import make_cv_folds


def test_metrics_match_hand_calculation():
    """MAE, RMSE and R-squared must equal values worked out by hand."""
    # errors are 0, 0, -1, 1 ; mean of y_true is 2.5
    scores = regression_metrics([1, 2, 3, 4], [1, 2, 4, 3])
    assert scores["mae"] == pytest.approx(0.5)            # (0+0+1+1)/4
    assert scores["rmse"] == pytest.approx(np.sqrt(0.5))   # sqrt((0+0+1+1)/4)
    assert scores["r2"] == pytest.approx(0.6)              # 1 - 2/5


def test_perfect_prediction_scores():
    """A perfect prediction has zero error and R-squared of 1."""
    scores = regression_metrics([0.95, 0.97, 1.0], [0.95, 0.97, 1.0])
    assert scores == pytest.approx({"mae": 0.0, "rmse": 0.0, "r2": 1.0})


def test_metrics_reject_mismatched_lengths():
    """Inputs of different lengths must raise ValueError."""
    with pytest.raises(ValueError):
        regression_metrics([1, 2, 3], [1, 2])


def test_baseline_predicts_training_mean():
    """The mean baseline must predict the training mean for every row."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.95, 0.96, 0.98, 1.0])
    model = build_models()["mean_baseline"].fit(X, y)
    assert np.allclose(model.predict(X), y.mean())


def test_linear_recovers_exact_line():
    """On data that is exactly a straight line, linear regression is perfect."""
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) ** 2})
    y = pd.Series(1.0 - 0.001 * X["a"] + 0.00001 * X["b"])
    model = build_models()["linear"].fit(X, y)
    assert regression_metrics(y, model.predict(X))["rmse"] < 1e-10


def test_cross_validation_table_shape():
    """One row per model, with mean and std of each metric and fit time."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 0.01 + 0.97)
    table = cross_validate_models(build_models(), X, y, make_cv_folds(3))
    assert list(table.index) == ["mean_baseline", "linear", "tree", "svr"]
    for col in ["mae", "rmse", "r2", "fit_s"]:
        assert f"{col}_mean" in table.columns and f"{col}_std" in table.columns
