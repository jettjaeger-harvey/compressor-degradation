"""Loading and preparing the UCI naval propulsion (gas-turbine) dataset.

Source: Coraddu et al. (2014), "Machine Learning Approaches for Improving
Condition-Based Maintenance of Naval Propulsion Plants", UCI Machine Learning
Repository dataset 316.

Each row of ``data.txt`` is one steady-state operating case from a frigate
propulsion simulator: 16 sensor measurements followed by the compressor and
turbine decay coefficients.
"""

from pathlib import Path

import pandas as pd

#: Column names in file order, with units in the documentation below.
#: Temperatures are documented as degrees C, but the inlet temperature is a
#: constant 288, which is only physical in kelvin (15 degrees C), so all
#: temperatures are treated as kelvin.
COLUMNS = [
    "lever_position",          # lp, dimensionless
    "ship_speed_kn",           # v, knots
    "gt_shaft_torque_kNm",     # GTT, kN m
    "gt_speed_rpm",            # GTn, rpm
    "gg_speed_rpm",            # GGn, rpm
    "stbd_prop_torque_kN",     # Ts, kN
    "port_prop_torque_kN",     # Tp, kN
    "hp_turbine_exit_temp_K",  # T48
    "comp_inlet_temp_K",       # T1
    "comp_outlet_temp_K",      # T2
    "hp_turbine_exit_pres_bar",  # P48
    "comp_inlet_pres_bar",     # P1
    "comp_outlet_pres_bar",    # P2
    "exhaust_pres_bar",        # Pexh
    "turbine_injection_pct",   # TIC, %
    "fuel_flow_kgs",           # mf, kg/s
    "comp_decay",              # kMc, target
    "turbine_decay",           # kMt, not used as an input
]

#: Name of the target column (compressor decay state coefficient).
TARGET = "comp_decay"

#: Valid range of the target given by the dataset documentation.
TARGET_RANGE = (0.95, 1.0)


def load_dataset(path):
    """Read ``data.txt`` and check that it is valid.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the whitespace-separated ``data.txt`` file.

    Returns
    -------
    pandas.DataFrame
        One row per operating case and the 18 columns in :data:`COLUMNS`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a row does not have 18 numeric values, a value is missing, or a
        decay coefficient lies outside its documented range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {df.shape[1]}")
    df.columns = COLUMNS

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("File contains missing or non-numeric values")

    low, high = TARGET_RANGE
    if not df[TARGET].between(low, high).all():
        raise ValueError(f"{TARGET} outside documented range {TARGET_RANGE}")
    if not df["turbine_decay"].between(0.975, 1.0).all():
        raise ValueError("turbine_decay outside documented range (0.975, 1.0)")

    return df


def find_redundant_columns(X):
    """Find input columns that carry no extra information.

    A column is redundant if it is constant, or if it is an exact copy or an
    exact one-to-one relabelling of a column that appears earlier.

    Parameters
    ----------
    X : pandas.DataFrame
        Input (sensor) columns only.

    Returns
    -------
    dict
        Maps each redundant column name to the reason it is redundant.
    """
    reasons = {}
    for i, col in enumerate(X.columns):
        if X[col].nunique() == 1:
            reasons[col] = "constant"
            continue
        for earlier in X.columns[:i]:
            if earlier in reasons:
                continue
            pairs = X[[earlier, col]].drop_duplicates()
            one_to_one = (
                len(pairs) == X[earlier].nunique() == X[col].nunique()
            )
            if one_to_one:
                reasons[col] = f"one-to-one with {earlier}"
                break
    return reasons


def split_inputs_target(df):
    """Separate the sensor inputs from the target and drop redundant sensors.

    The turbine decay coefficient is removed because it would not be known
    in practice.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_dataset`.

    Returns
    -------
    X : pandas.DataFrame
        Informative sensor inputs.
    y : pandas.Series
        Compressor decay coefficient (dimensionless, 0.95 to 1.0).
    dropped : dict
        Redundant columns that were removed and why.
    """
    X = df.drop(columns=[TARGET, "turbine_decay"])
    dropped = find_redundant_columns(X)
    return X.drop(columns=list(dropped)), df[TARGET], dropped
