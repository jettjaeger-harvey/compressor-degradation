"""Tests for degradation.performance."""

import numpy as np
import pandas as pd
import pytest

from degradation.models import build_models
from degradation.performance import (least_squares_stability, model_size_bytes,
                                     noise_robustness, scaling_study, time_model)


def _line():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(1, 2, 200), "b": rng.uniform(1, 2, 200)})
    y = pd.Series(0.95 + 0.02 * X["a"] + 0.01 * X["b"])
    return X, y


def test_time_model_returns_positive_times():
    """Fit and prediction times must be positive and the model must be fitted."""
    X, y = _line()
    r = time_model(build_models()["linear"], X, y, X, repeats=2)
    assert r["fit_s"] > 0 and r["predict_us_per_case"] > 0
    assert len(r["model"].predict(X)) == len(X)


def test_bigger_tree_needs_more_memory():
    """A fully grown tree must take more bytes to store than a stump."""
    X, y = _line()
    stump = build_models()["tree"].set_params(max_depth=1).fit(X, y)
    full = build_models()["tree"].fit(X, y)
    assert model_size_bytes(full) > model_size_bytes(stump)


def test_scaling_rejects_too_large_size():
    """Asking for more rows than exist must raise ValueError."""
    X, y = _line()
    with pytest.raises(ValueError):
        scaling_study(build_models()["linear"], X, y, [50, 500])


def test_zero_noise_gives_clean_error():
    """With zero noise, the MAE must equal the clean-data MAE (0 for an exact fit)."""
    X, y = _line()
    m = build_models()["linear"].fit(X, y)
    res = noise_robustness({"linear": m}, X, y, [0.0, 0.01])
    assert res.loc[0, "linear"] < 1e-12
    assert res.loc[1, "linear"] > res.loc[0, "linear"]


def test_normal_equations_square_the_condition_number():
    """cond(X^T X) must be about cond(X) squared, and both solvers must agree."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    X = pd.DataFrame({"a": a, "b": a + 0.1 * rng.normal(size=300)})
    y = pd.Series(1.0 + 2.0 * X["a"] - X["b"])
    r = least_squares_stability(X, y)
    assert r["cond_XtX"] == pytest.approx(r["cond_X"] ** 2, rel=0.2)
    assert r["max_coef_difference"] < 1e-8
