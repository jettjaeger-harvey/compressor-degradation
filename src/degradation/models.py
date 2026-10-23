"""The four regression models and their cross-validated evaluation.

Models
------
- ``mean_baseline``: always predicts the mean of the training targets.
- ``linear``: ordinary least-squares linear regression.
- ``tree``: decision-tree regression.
- ``svr``: support-vector regression with a radial basis function kernel.

Linear regression and SVR are sensitive to the very different scales of the
sensors (torque up to ~72,000 kN m, pressures ~1-23 bar), so their inputs are
standardised inside a pipeline.  The pipeline re-learns the scaling from the
training rows of every fold, so no information leaks from validation rows.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from degradation.splitting import RANDOM_STATE


def build_models():
    """Create the four untrained models with default settings.

    Returns
    -------
    dict
        Maps a short model name to an unfitted scikit-learn estimator.
    """
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "svr": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
    }


def regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE and R-squared.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted compressor decay coefficients (same length).

    Returns
    -------
    dict
        ``mae`` and ``rmse`` (same units as the target) and ``r2``
        (dimensionless; 1 is perfect, 0 equals predicting the mean).

    Raises
    ------
    ValueError
        If the inputs have different lengths, are empty, or contain NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs contain NaN")

    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def cross_validate_models(models, X, y, cv):
    """Score every model with k-fold cross-validation.

    For each fold, a fresh copy of the model is trained on the training rows
    and scored on the validation rows; fit time is measured with a wall clock.

    Parameters
    ----------
    models : dict
        Name -> unfitted estimator, e.g. from :func:`build_models`.
    X : pandas.DataFrame
        Development-set inputs.
    y : pandas.Series
        Development-set targets.
    cv : sklearn.model_selection.KFold
        Fold splitter, e.g. from ``splitting.make_cv_folds``.

    Returns
    -------
    pandas.DataFrame
        One row per model with the mean and standard deviation over folds of
        MAE, RMSE, R-squared and fit time (seconds).
    """
    rows = []
    for name, model in models.items():
        fold_scores = []
        for train_idx, val_idx in cv.split(X):
            m = clone(model)
            start = time.perf_counter()
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_s = time.perf_counter() - start
            scores = regression_metrics(y.iloc[val_idx], m.predict(X.iloc[val_idx]))
            scores["fit_s"] = fit_s
            fold_scores.append(scores)
        folds = pd.DataFrame(fold_scores)
        row = {"model": name}
        for col in folds:
            row[f"{col}_mean"] = folds[col].mean()
            row[f"{col}_std"] = folds[col].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
