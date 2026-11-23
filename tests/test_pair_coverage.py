"""Coverage of every ordered operator pair (T2-A3, D86).

The table this script writes becomes T1-A's label, so the two things that could bias it
silently are pinned: which side the threshold comes from, and that the matched cell is the
project's own split-conformal rule and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pair_coverage_by_split as pairs  # noqa: E402

from caliber.calibration.conformal import split_threshold  # noqa: E402


def _frame(shift_b: float = -1.0, n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("calib", "test"):
        for p in range(10):
            for op, shift in (("a", 0.0), ("b", shift_b)):
                for _ in range(n):
                    rows.append({"operator_id": op, "split": split, "patient_id": f"{split}{p}",
                                 "label": bool(rng.random() < 0.5),
                                 "score": float(rng.normal(shift, 1.0))})
    return pd.DataFrame(rows)


def test_the_threshold_is_the_projects_split_conformal_rule():
    df = _frame()
    cal = df[(df.operator_id == "a") & (df.split == "calib")]
    expected = split_threshold(cal["score"], cal["label"], alpha=0.1).threshold
    by = pairs.positives_by_cluster(df, "patient_id")
    cell = next(c for c in pairs.pair_cells(by, ["a", "b"], 0.1)
                if c["calibration"] == "a" and c["deployment"] == "a")
    assert cell["tau"] == pytest.approx(expected)


def test_too_few_positives_flags_everything_and_does_not_invent_a_threshold():
    assert pairs.threshold(np.array([0.3, 0.7]), 0.1) == float("-inf")


def test_the_threshold_comes_from_the_calibration_operator_and_coverage_from_the_other():
    """Operator b scores one unit lower. A threshold fitted at a and used at b loses
    coverage; fitted at b and used at a it over-covers. Swapping the two sides in the code
    would swap these, and the pair table would say the opposite of the bank."""
    by = pairs.positives_by_cluster(_frame(), "patient_id")
    cells = {(c["calibration"], c["deployment"]): c["coverage"]
             for c in pairs.pair_cells(by, ["a", "b"], 0.1)}
    assert cells[("a", "b")] < cells[("a", "a")] - 0.15
    assert cells[("b", "a")] > cells[("b", "b")] + 0.03
    assert cells[("a", "a")] == pytest.approx(0.9, abs=0.08)
    assert cells[("b", "b")] == pytest.approx(0.9, abs=0.08)


def test_only_positives_and_only_the_right_split_are_counted():
    df = _frame()
    by = pairs.positives_by_cluster(df, "patient_id")
    n_cal = int(((df.operator_id == "a") & (df.split == "calib") & df.label).sum())
    n_test = int(((df.operator_id == "b") & (df.split == "test") & df.label).sum())
    cell = next(c for c in pairs.pair_cells(by, ["a", "b"], 0.1)
                if c["calibration"] == "a" and c["deployment"] == "b")
    assert (cell["n_positives_calib"], cell["n_positives_test"]) == (n_cal, n_test)
