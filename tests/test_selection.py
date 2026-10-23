"""Tests for degradation.selection."""

import numpy as np
import pandas as pd
import pytest

from degradation.models import build_models
from degradation.selection import condition_number, forward_selection
from degradation.splitting import make_cv_folds


def test_independent_columns_condition_number_is_one():
    """Two uncorrelated (orthogonal) columns give a condition number of 1."""
    X = pd.DataFrame({"a": [1.0, -1.0, 1.0, -1.0], "b": [1.0, 1.0, -1.0, -1.0]})
    assert condition_number(X) == pytest.approx(1.0)


def test_nearly_duplicate_columns_are_ill_conditioned():
    """A column that is almost a copy of another gives a huge condition number."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "b": a + 1e-8 * rng.normal(size=200)})
    assert condition_number(X) > 1e6


def test_constant_column_rejected():
    """A constant column cannot be standardised and must raise ValueError."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 4.0, 4.0]})
    with pytest.raises(ValueError):
        condition_number(X)


def test_forward_selection_picks_informative_sensor_first():
    """When only one column drives the target, it must be selected first."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["noise1", "signal", "noise2"])
    y = pd.Series(0.97 + 0.01 * X["signal"])
    table = forward_selection(build_models()["linear"], X, y, make_cv_folds(3), n_jobs=1)
    assert table.loc[0, "added"] == "signal"
    assert list(table["n_sensors"]) == [1, 2, 3]


def test_nested_subsets_follow_given_order():
    """Row k must use the first k sensors of the given order."""
    from degradation.selection import evaluate_nested_subsets

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(90, 3)), columns=["a", "b", "c"])
    y = pd.Series(0.97 + 0.01 * X["b"] - 0.005 * X["c"])
    table = evaluate_nested_subsets(build_models()["linear"], X, y, make_cv_folds(3),
                                    ["b", "c", "a"], n_jobs=1)
    assert list(table["added"]) == ["b", "c", "a"]
    assert table.loc[1, "mae_mean"] < table.loc[0, "mae_mean"]  # adding c helps
