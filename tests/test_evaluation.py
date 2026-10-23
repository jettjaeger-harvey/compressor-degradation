"""Tests for degradation.evaluation."""

import numpy as np
import pandas as pd

from degradation.evaluation import (TUNED_PARAMS, build_tuned_models,
                                    evaluate_on_test, extrapolation_test)
from degradation.models import build_models


def test_tuned_settings_are_applied():
    """build_tuned_models must apply the settings chosen by tuning."""
    tuned = build_tuned_models(build_models())
    assert tuned["svr"].get_params()["svr__epsilon"] == TUNED_PARAMS["svr"]["svr__epsilon"]
    assert tuned["tree"].get_params()["max_depth"] is None


def test_test_table_has_one_row_per_model():
    """The test-set table must contain every model and the timing columns."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    y = pd.Series(0.97 + 0.01 * X["a"])
    models = {"mean_baseline": build_models()["mean_baseline"], "linear": build_models()["linear"]}
    table, preds = evaluate_on_test(models, X[:40], y[:40], X[40:], y[40:])
    assert list(table.index) == ["mean_baseline", "linear"]
    assert {"mae", "rmse", "r2", "fit_s", "predict_us_per_case"} <= set(table.columns)
    assert len(preds["linear"]) == 20


def test_tree_cannot_extrapolate_but_linear_can():
    """On a straight line, the tree fails at the edge bands; linear does not."""
    x = np.linspace(0, 1, 200)
    X = pd.DataFrame({"x": x})
    y = pd.Series(0.95 + 0.05 * x)
    models = {"linear": build_models()["linear"], "tree": build_models()["tree"]}
    res = extrapolation_test(models, X, y, n_bands=5)
    edge = res[res["edge_band"]].groupby("model")["mae"].mean()
    assert edge["linear"] < 1e-10
    assert edge["tree"] > 0.003
