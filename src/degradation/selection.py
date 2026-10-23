"""Sensor selection: how accuracy, runtime and conditioning depend on which
sensors are used.

Greedy forward selection starts with no sensors and, at each step, adds the
single sensor that most reduces the cross-validated MAE.  This ranks the
sensors and shows how few are needed to detect compressor degradation.

Forward selection retrains the model many times (66 subsets for 12 sensors at
the first few steps alone), so it is run with the fast decision tree.  The
slower SVR is then scored on the nested sensor subsets in the tree's order
with :func:`evaluate_nested_subsets`.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_validate


def condition_number(X):
    """2-norm condition number of the standardised input matrix.

    Each column is scaled to mean 0 and standard deviation 1 first, so the
    result reflects how close the columns are to being linearly dependent,
    not their units.  A value near 1 means independent columns; a very large
    value means some columns are (nearly) combinations of others, which makes
    least-squares solutions sensitive to rounding and noise.

    Parameters
    ----------
    X : pandas.DataFrame
        Input columns (at least one row and one non-constant column).

    Returns
    -------
    float
        Ratio of largest to smallest singular value.

    Raises
    ------
    ValueError
        If any column is constant (it cannot be standardised).
    """
    std = X.std(ddof=0)
    if (std == 0).any():
        raise ValueError(f"Constant columns: {list(std[std == 0].index)}")
    Z = (X - X.mean()) / std
    s = np.linalg.svd(Z.to_numpy(dtype=float), compute_uv=False)
    return np.inf if s[-1] == 0 else s[0] / s[-1]


def forward_selection(model, X, y, cv, max_sensors=None, n_jobs=-1):
    """Greedy forward selection of sensors by cross-validated MAE.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model to evaluate (e.g. the tuned SVR pipeline).
    X : pandas.DataFrame
        Development-set inputs; each column is one sensor.
    y : pandas.Series
        Development-set targets.
    cv : sklearn.model_selection.KFold
        Fold splitter.
    max_sensors : int or None
        Stop after this many sensors (default: all of them).
    n_jobs : int
        CPU cores for cross-validation (-1 means all).

    Returns
    -------
    pandas.DataFrame
        One row per step: the sensor added, number of sensors, mean and std
        of validation MAE, mean R-squared, the time to fit the model once on
        all development rows (seconds), and the condition number of the
        selected inputs.
    """
    remaining = list(X.columns)
    chosen = []
    rows = []
    max_sensors = max_sensors or len(remaining)
    while remaining and len(chosen) < max_sensors:
        best = None
        for sensor in remaining:
            cols = chosen + [sensor]
            mae = -cross_val_score(clone(model), X[cols], y, cv=cv,
                                   scoring="neg_mean_absolute_error", n_jobs=n_jobs)
            if best is None or mae.mean() < best[1].mean():
                best = (sensor, mae)
        sensor, mae = best
        chosen.append(sensor)
        remaining.remove(sensor)
        r2 = cross_val_score(clone(model), X[chosen], y, cv=cv, scoring="r2", n_jobs=n_jobs)
        start = time.perf_counter()
        clone(model).fit(X[chosen], y)
        fit_s = time.perf_counter() - start
        rows.append({
            "added": sensor,
            "n_sensors": len(chosen),
            "mae_mean": mae.mean(),
            "mae_std": mae.std(),
            "r2_mean": r2.mean(),
            "fit_s": fit_s,
            "condition_number": condition_number(X[chosen]),
        })
    return pd.DataFrame(rows)


def evaluate_nested_subsets(model, X, y, cv, order, n_jobs=-1):
    """Score a model on the first 1, 2, ..., k sensors of a given ranking.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model to evaluate.
    X : pandas.DataFrame
        Development-set inputs.
    y : pandas.Series
        Development-set targets.
    cv : sklearn.model_selection.KFold
        Fold splitter.
    order : list of str
        Sensor names, most useful first (e.g. the ``added`` column returned
        by :func:`forward_selection`).
    n_jobs : int
        CPU cores for cross-validation (-1 means all).

    Returns
    -------
    pandas.DataFrame
        Same columns as :func:`forward_selection`, one row per subset size.
    """
    rows = []
    for k in range(1, len(order) + 1):
        cols = list(order[:k])
        scores = cross_validate(clone(model), X[cols], y, cv=cv, n_jobs=n_jobs,
                                scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
        mae, r2 = -scores["test_mae"], scores["test_r2"]
        start = time.perf_counter()
        fitted = clone(model).fit(X[cols], y)
        fit_s = time.perf_counter() - start
        row = {
            "added": order[k - 1],
            "n_sensors": k,
            "mae_mean": mae.mean(),
            "mae_std": mae.std(),
            "r2_mean": r2.mean(),
            "fit_s": fit_s,
            "condition_number": condition_number(X[cols]),
        }
        steps = getattr(fitted, "steps", None)
        last = steps[-1][1] if steps else fitted
        if hasattr(last, "support_"):
            row["n_support_vectors"] = last.support_.size
        rows.append(row)
    return pd.DataFrame(rows)
