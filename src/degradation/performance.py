"""Performance analysis: timing, profiling, memory, scalability, robustness,
repeatability and the numerical stability of least squares.

Every function here works on the development or test data produced by
``splitting.make_test_split`` and never changes a model's settings.
"""

import cProfile
import io
import pickle
import pstats
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from degradation.models import regression_metrics


def time_model(model, X_train, y_train, X_test, repeats=5):
    """Median wall-clock time to fit a model and to predict, over several repeats.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data.
    X_test : pandas.DataFrame
        Inputs to predict.
    repeats : int
        Number of repeats; the median is reported to reduce timing noise.

    Returns
    -------
    dict
        ``fit_s`` (seconds), ``predict_us_per_case`` (microseconds) and the
        fitted model (``model``) from the last repeat.
    """
    fits, preds = [], []
    for _ in range(repeats):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        fits.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        m.predict(X_test)
        preds.append((time.perf_counter() - t0) / len(X_test) * 1e6)
    return {"fit_s": float(np.median(fits)),
            "predict_us_per_case": float(np.median(preds)), "model": m}


def profile_fit(model, X, y, top=8):
    """Profile one fit with cProfile and return the most expensive functions.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    X, y : pandas.DataFrame, pandas.Series
        Training data.
    top : int
        Number of functions to return, by cumulative time.

    Returns
    -------
    pandas.DataFrame
        Function name, number of calls, own time and cumulative time (s).
    """
    prof = cProfile.Profile()
    prof.enable()
    clone(model).fit(X, y)
    prof.disable()
    stats = pstats.Stats(prof, stream=io.StringIO())
    rows = []
    for (file, line, name), (cc, nc, tt, ct, _) in stats.stats.items():
        rows.append({"function": f"{name} ({file.split('/')[-1].split(chr(92))[-1]}:{line})",
                     "calls": nc, "own_s": tt, "cumulative_s": ct})
    return (pd.DataFrame(rows).sort_values("cumulative_s", ascending=False)
            .head(top).reset_index(drop=True))


def model_size_bytes(fitted_model):
    """Memory needed to store a fitted model, measured as its pickled size (bytes)."""
    return len(pickle.dumps(fitted_model))


def scaling_study(model, X, y, sizes, repeats=3, random_state=42):
    """Fit time against number of training cases, with a fitted power law.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    X, y : pandas.DataFrame, pandas.Series
        Pool of training data to sample from.
    sizes : list of int
        Training-set sizes to time (each at most ``len(X)``).
    repeats : int
        Timing repeats per size (median reported).
    random_state : int
        Seed for choosing the subsets.

    Returns
    -------
    table : pandas.DataFrame
        ``n`` and median ``fit_s`` for each size.
    exponent : float
        Slope k of log(fit time) against log(n), so fit time is roughly
        proportional to n to the power k.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for n in sizes:
        if n > len(X):
            raise ValueError(f"size {n} exceeds the {len(X)} available rows")
        idx = rng.choice(len(X), size=n, replace=False)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            clone(model).fit(X.iloc[idx], y.iloc[idx])
            times.append(time.perf_counter() - t0)
        rows.append({"n": n, "fit_s": float(np.median(times))})
    table = pd.DataFrame(rows)
    exponent = float(np.polyfit(np.log(table["n"]), np.log(table["fit_s"]), 1)[0])
    return table, exponent


def noise_robustness(fitted_models, X_test, y_test, noise_levels, random_state=42):
    """Test-set MAE when every sensor reading is perturbed by random noise.

    Each reading is multiplied by (1 + e), with e drawn from a normal
    distribution with standard deviation equal to the noise level, so a level
    of 0.01 is a 1% (one standard deviation) relative measurement error.
    Models are trained on clean data, as in practice.

    Parameters
    ----------
    fitted_models : dict
        Name -> model already fitted on clean development data.
    X_test, y_test : pandas.DataFrame, pandas.Series
        Clean test data.
    noise_levels : list of float
        Relative noise standard deviations, e.g. [0, 0.001, 0.01].
    random_state : int
        Seed for the noise, so every model sees identical noisy inputs.

    Returns
    -------
    pandas.DataFrame
        One row per noise level, one MAE column per model.
    """
    rows = []
    for level in noise_levels:
        rng = np.random.default_rng(random_state)
        noisy = X_test * (1 + rng.normal(0.0, level, size=X_test.shape))
        row = {"noise_pct": 100 * level}
        for name, m in fitted_models.items():
            row[name] = regression_metrics(y_test, m.predict(noisy))["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def least_squares_stability(X, y):
    """Compare two ways of solving linear least squares on standardised inputs.

    The normal equations solve (X^T X) b = X^T y, whose condition number is
    the square of X's; the SVD-based solver (numpy.linalg.lstsq) works with X
    directly.  The result shows how conditioning affects each.

    Parameters
    ----------
    X : pandas.DataFrame
        Inputs (non-constant columns).
    y : pandas.Series
        Targets.

    Returns
    -------
    dict
        Condition numbers of X and X^T X, the largest coefficient magnitude
        from each method, the largest difference between the two coefficient
        vectors, and the training RMSE of each.
    """
    Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(Z)), Z])
    yv = y.to_numpy(dtype=float)
    b_svd = np.linalg.lstsq(A, yv, rcond=None)[0]
    AtA = A.T @ A
    try:
        b_ne = np.linalg.solve(AtA, A.T @ yv)
    except np.linalg.LinAlgError:
        b_ne = np.full_like(b_svd, np.nan)
    s = np.linalg.svd(Z, compute_uv=False)
    rmse = lambda b: float(np.sqrt(np.mean((A @ b - yv) ** 2)))
    failed = float("nan")  # normal equations could not be solved (singular matrix)
    return {
        "cond_X": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
        "cond_XtX": float(np.linalg.cond(AtA)),
        "max_coef_svd": float(np.max(np.abs(b_svd))),
        "max_coef_normal_eq": failed if np.isnan(b_ne).all() else float(np.max(np.abs(b_ne))),
        "max_coef_difference": failed if np.isnan(b_ne).all() else float(np.max(np.abs(b_svd - b_ne))),
        "rmse_svd": rmse(b_svd),
        "rmse_normal_eq": rmse(b_ne),
    }


def repeatability(models, X, y, seeds, n_splits=5):
    """Cross-validated MAE of each model for several different random splits.

    Parameters
    ----------
    models : dict
        Name -> unfitted model.
    X, y : pandas.DataFrame, pandas.Series
        Full dataset (a new test split is drawn for each seed).
    seeds : list of int
        Random seeds for the test split and the folds.
    n_splits : int
        Number of cross-validation folds.

    Returns
    -------
    pandas.DataFrame
        One row per (seed, model): cross-validated MAE and test MAE.
    """
    from sklearn.model_selection import cross_val_score

    from degradation.splitting import make_cv_folds, make_test_split

    rows = []
    for seed in seeds:
        Xd, Xt, yd, yt = make_test_split(X, y, random_state=seed)
        for name, model in models.items():
            cv = -cross_val_score(clone(model), Xd, yd, cv=make_cv_folds(n_splits, seed),
                                  scoring="neg_mean_absolute_error", n_jobs=-1).mean()
            test = regression_metrics(yt, clone(model).fit(Xd, yd).predict(Xt))["mae"]
            rows.append({"seed": seed, "model": name, "cv_mae": cv, "test_mae": test})
    return pd.DataFrame(rows)
