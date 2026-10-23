"""Final evaluation on the locked test set and the extrapolation check.

The test set is used exactly once, after all settings have been chosen by
cross-validation on the development set.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from degradation.models import regression_metrics

#: Settings chosen by 5-fold cross-validation (see results/*_tuning.csv).
TUNED_PARAMS = {
    "mean_baseline": {},
    "linear": {},
    "tree": {"max_depth": None, "min_samples_leaf": 1},
    "svr": {"svr__C": 100, "svr__epsilon": 0.0001, "svr__gamma": "scale"},
}


def build_tuned_models(models):
    """Apply :data:`TUNED_PARAMS` to fresh copies of the given models.

    Parameters
    ----------
    models : dict
        Name -> unfitted estimator, e.g. from ``models.build_models()``.

    Returns
    -------
    dict
        Name -> unfitted estimator with the tuned settings.
    """
    return {name: clone(m).set_params(**TUNED_PARAMS.get(name, {}))
            for name, m in models.items()}


def evaluate_on_test(models, X_dev, y_dev, X_test, y_test):
    """Train each model on the whole development set and score it on the test set.

    Parameters
    ----------
    models : dict
        Name -> unfitted estimator (already tuned).
    X_dev, y_dev : pandas.DataFrame, pandas.Series
        Development data used for training.
    X_test, y_test : pandas.DataFrame, pandas.Series
        Locked test data, used only for scoring.

    Returns
    -------
    table : pandas.DataFrame
        One row per model: test MAE, RMSE, R-squared, fit time (s) and
        prediction time per test case (microseconds).
    predictions : dict
        Name -> array of test-set predictions (for plots).
    """
    rows, predictions = [], {}
    for name, model in models.items():
        m = clone(model)
        start = time.perf_counter()
        m.fit(X_dev, y_dev)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = m.predict(X_test)
        predict_us = (time.perf_counter() - start) / len(X_test) * 1e6
        predictions[name] = pred
        rows.append({"model": name, **regression_metrics(y_test, pred),
                     "fit_s": fit_s, "predict_us_per_case": predict_us})
    return pd.DataFrame(rows).set_index("model"), predictions


def extrapolation_test(models, X, y, n_bands=5):
    """Test each model on degradation levels outside its training range.

    The decay range is cut into ``n_bands`` equal bands.  For each band, the
    model is trained on all other bands and tested on that band, so the
    tested degradation states are never inside the training range for the
    two outer bands and are bracketed by training data for the inner bands.

    Parameters
    ----------
    models : dict
        Name -> unfitted estimator.
    X : pandas.DataFrame
        Development-set inputs.
    y : pandas.Series
        Development-set compressor decay coefficients.
    n_bands : int
        Number of equal-width decay bands.

    Returns
    -------
    pandas.DataFrame
        One row per (model, band): the band limits, MAE and whether the band
        lies at the edge of the range (true extrapolation).
    """
    edges = np.linspace(y.min(), y.max(), n_bands + 1)
    band = np.clip(np.digitize(y, edges[1:-1]), 0, n_bands - 1)
    rows = []
    for name, model in models.items():
        for b in range(n_bands):
            test = band == b
            m = clone(model).fit(X[~test], y[~test])
            mae = regression_metrics(y[test], m.predict(X[test]))["mae"]
            rows.append({"model": name, "band": b,
                         "decay_low": edges[b], "decay_high": edges[b + 1],
                         "edge_band": b in (0, n_bands - 1), "mae": mae})
    return pd.DataFrame(rows)
