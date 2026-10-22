"""Tests for degradation.data.

Each test builds a small input where the correct answer is known in advance
and checks that the code gives that answer.
"""

import pandas as pd
import pytest

from degradation.data import find_redundant_columns, load_dataset


# ---------------------------------------------------------------------------
# Finished examples
# ---------------------------------------------------------------------------

def test_constant_column_is_flagged():
    """A column whose value never changes must be flagged as constant."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
    assert find_redundant_columns(X) == {"b": "constant"}


def test_missing_file_raises_error(tmp_path):
    """Loading a file that does not exist must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.txt")


# ---------------------------------------------------------------------------
# Redundant-sensor detection
# ---------------------------------------------------------------------------

def test_copied_column_is_flagged():
    """A column that is an exact copy of an earlier column must be flagged."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    assert find_redundant_columns(X) == {"b": "one-to-one with a"}


def test_independent_columns_are_kept():
    """Columns that carry different information must not be flagged."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]})
    assert find_redundant_columns(X) == {}


# ---------------------------------------------------------------------------
# Invalid input files
# ---------------------------------------------------------------------------

def test_wrong_number_of_columns_raises_error(tmp_path):
    """A file whose rows do not have 18 values must raise ValueError."""
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="Expected 18 columns"):
        load_dataset(path)


def test_target_out_of_range_raises_error(tmp_path):
    """A compressor decay value outside 0.95-1.0 must raise ValueError."""
    row = ["1"] * 16 + ["0.90", "0.98"]  # 16 sensors, bad kMc, valid kMt
    path = tmp_path / "bad.txt"
    path.write_text(" ".join(row) + "\n")
    with pytest.raises(ValueError, match="comp_decay outside"):
        load_dataset(path)
