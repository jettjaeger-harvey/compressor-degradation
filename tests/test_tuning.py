"""Tests for degradation.tuning."""

import numpy as np
import pandas as pd

from degradation.models import build_models
from degradation.splitting import make_cv_folds
from degradation.tuning import tune_model


def _toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(size=(80, 2)), columns=["a", "b"])
    y = pd.Series(0.95 + 0.05 * X["a"] ** 2)
    return X, y


def test_one_row_per_combination():
    """The results table must have one row for every combination tried."""
    X, y = _toy()
    grid = {"max_depth": [1, 2, 3], "min_samples_leaf": [1, 5]}
    _, table = tune_model(build_models()["tree"], grid, X, y, make_cv_folds(3), n_jobs=1)
    assert len(table) == 3 * 2


def test_best_setting_has_lowest_mae():
    """The returned best setting must be the first (lowest-MAE) row."""
    X, y = _toy()
    grid = {"max_depth": [1, 2, 3, 4]}
    best, table = tune_model(build_models()["tree"], grid, X, y, make_cv_folds(3), n_jobs=1)
    assert table["mae_mean"].is_monotonic_increasing
    assert best["max_depth"] == table.loc[0, "max_depth"]


def test_deeper_tree_fits_curve_better():
    """On a smooth curve, a depth-4 tree must beat a depth-1 tree."""
    X, y = _toy()
    _, table = tune_model(build_models()["tree"], {"max_depth": [1, 4]}, X, y,
                          make_cv_folds(3), n_jobs=1)
    mae = table.set_index("max_depth")["mae_mean"]
    assert mae[4] < mae[1]
