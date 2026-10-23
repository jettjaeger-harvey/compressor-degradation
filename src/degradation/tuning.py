"""Hyperparameter tuning by grid search with k-fold cross-validation.

Every combination of settings in a grid is trained and scored on each fold of
the development set.  The full table of results is kept (not just the best
setting) so the effect of each setting on accuracy and runtime can be studied.
The test set is never used here.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV

#: Settings searched for each model.  Settings inside a pipeline are written
#: ``<step>__<setting>``; ``svr`` is the SVR step of the svr pipeline.
PARAM_GRIDS = {
    "tree": {
        "max_depth": [2, 4, 6, 8, 10, 12, 15, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
    },
    "svr": {
        "svr__epsilon": [0.1, 0.01, 0.001, 0.0001],
        "svr__C": [0.1, 1, 10, 100],
        "svr__gamma": ["scale", 0.01, 0.1, 1.0],
    },
}


def tune_model(model, param_grid, X, y, cv, n_jobs=-1):
    """Grid-search one model's settings with cross-validation.

    The best setting is the one with the lowest mean validation MAE.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model, e.g. from ``models.build_models()``.
    param_grid : dict
        Setting name -> list of values to try.
    X : pandas.DataFrame
        Development-set inputs.
    y : pandas.Series
        Development-set targets.
    cv : sklearn.model_selection.KFold
        Fold splitter.
    n_jobs : int
        Number of CPU cores to use (-1 means all).

    Returns
    -------
    best_params : dict
        Setting values with the lowest mean validation MAE.
    table : pandas.DataFrame
        One row per combination, sorted best first: the settings, mean and std
        of validation MAE, RMSE and R-squared, and mean fit time (seconds).
    """
    search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        refit="mae",
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    res = search.cv_results_

    table = pd.DataFrame(res["params"])
    table.columns = [c.split("__")[-1] for c in table.columns]
    for metric, sign in [("mae", -1), ("rmse", -1), ("r2", 1)]:
        table[f"{metric}_mean"] = sign * res[f"mean_test_{metric}"]
        table[f"{metric}_std"] = res[f"std_test_{metric}"]
    table["fit_s_mean"] = res["mean_fit_time"]
    return search.best_params_, table.sort_values("mae_mean").reset_index(drop=True)
