"""Tests for degradation.splitting."""

import numpy as np
import pandas as pd
import pytest

from degradation.splitting import make_cv_folds, make_test_split


@pytest.fixture
def toy_data():
    """100 rows with a unique id in column 'a' so rows can be tracked."""
    X = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2.0})
    y = pd.Series(np.linspace(0.95, 1.0, 100))
    return X, y


def test_test_split_sizes(toy_data):
    """A 20% test split of 100 rows gives 80 development and 20 test rows."""
    X, y = toy_data
    X_dev, X_test, y_dev, y_test = make_test_split(X, y, test_size=0.2)
    assert (len(X_dev), len(X_test)) == (80, 20)
    assert (len(y_dev), len(y_test)) == (80, 20)


def test_no_row_in_both_sets(toy_data):
    """No row may appear in both the development and the test set."""
    X, y = toy_data
    X_dev, X_test, _, _ = make_test_split(X, y)
    assert set(X_dev["a"]).isdisjoint(X_test["a"])
    assert len(X_dev) + len(X_test) == len(X)


def test_split_is_reproducible(toy_data):
    """The same seed must always give the same test rows."""
    X, y = toy_data
    first = make_test_split(X, y)[1]
    second = make_test_split(X, y)[1]
    pd.testing.assert_frame_equal(first, second)


def test_inputs_and_target_stay_aligned(toy_data):
    """Each input row must keep its own target value after shuffling."""
    X, y = toy_data
    X_dev, _, y_dev, _ = make_test_split(X, y)
    assert (X_dev.index == y_dev.index).all()


def test_bad_test_size_raises_error(toy_data):
    """A test_size outside (0, 1) must raise ValueError."""
    X, y = toy_data
    with pytest.raises(ValueError):
        make_test_split(X, y, test_size=1.5)


def test_every_row_validated_exactly_once(toy_data):
    """Across all folds, each development row is in a validation fold once."""
    X, _ = toy_data
    seen = np.concatenate([val for _, val in make_cv_folds(5).split(X)])
    assert sorted(seen) == list(range(len(X)))
