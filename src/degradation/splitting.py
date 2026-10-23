"""Train/test splitting and cross-validation folds.

Workflow
--------
1. :func:`make_test_split` locks away a test set, used once for final scores.
2. :func:`make_cv_folds` defines k-fold cross-validation on the remaining
   development data; each fold acts as the validation set once while the
   other folds are used for training.
"""

from sklearn.model_selection import KFold, train_test_split

#: Seed for every random choice, so results can be reproduced exactly.
RANDOM_STATE = 42


def make_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Randomly split the data into a development set and a locked test set.

    Parameters
    ----------
    X : pandas.DataFrame
        Sensor inputs, one row per operating case.
    y : pandas.Series
        Compressor decay coefficient for each row.
    test_size : float
        Fraction of rows placed in the test set; must be between 0 and 1.
    random_state : int
        Seed for the random shuffle.

    Returns
    -------
    X_dev, X_test, y_dev, y_test
        Development data (for training and cross-validation) and test data.

    Raises
    ------
    ValueError
        If ``test_size`` is not strictly between 0 and 1, or ``X`` and ``y``
        have different numbers of rows.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_cv_folds(n_splits=5, random_state=RANDOM_STATE):
    """Create the k-fold cross-validation splitter for the development set.

    Parameters
    ----------
    n_splits : int
        Number of folds (at least 2).
    random_state : int
        Seed for shuffling rows before they are divided into folds.

    Returns
    -------
    sklearn.model_selection.KFold
        Splitter that yields (training indices, validation indices) pairs.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
