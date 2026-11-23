"""The spread-ratio table behind D92's frozen prediction (T2.11).

``transfer_spread_ratio.py`` puts a measured ratio beside a predicted one for every ordered
pair. These pin what each side is made of, on a case where the answer is known: positives
at ``b`` are the positives at ``a`` scaled by a constant, so the density ratio at the
threshold is the reciprocal of that constant.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.data.splits import split_for


@pytest.fixture(scope="module")
def spread():
    path = Path(__file__).resolve().parents[1] / "scripts" / "transfer_spread_ratio.py"
    spec = importlib.util.spec_from_file_location("transfer_spread_ratio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(scores, n_calib, alpha, n_splits, rng):
    """Per-split cells drawn the way the bank's are: one threshold per calibration draw."""
    rows = []
    k = int(math.floor(alpha * (n_calib + 1)))
    for split in range(n_splits):
        for a, sa in scores.items():
            tau = np.sort(rng.choice(sa, size=n_calib, replace=False))[k - 1]
            for b, sb in scores.items():
                rows.append({"salt": f"s{split}", "calibration": a, "deployment": b,
                             "coverage": float((sb >= tau).mean()),
                             "n_positives_test": sb.size})
    return pd.DataFrame(rows)


def test_predicted_ratio_is_the_density_ratio_at_the_threshold(spread):
    rng = np.random.default_rng(0)
    base = rng.exponential(size=40000)
    scores = {"a": base, "b": 2.0 * rng.exponential(size=40000)}
    table = spread.spread_ratios(_cells(scores, 400, 0.10, 4, rng), scores,
                                 {"a": 400, "b": 400}, 0.10).set_index(["calibration", "deployment"])
    # exponential(scale s): F_b(F_a^{-1}(u)) = 1 - (1 - u)^(1/s) for a at scale 1, so the
    # slope at u is (1/s) (1 - u)^(1/s - 1)
    assert table.loc[("a", "b"), "predicted_ratio"] == pytest.approx(0.5 * 0.9 ** -0.5, rel=0.05)
    assert table.loc[("b", "a"), "predicted_ratio"] == pytest.approx(2.0 * 0.9, rel=0.05)


def test_measured_ratio_is_the_transfer_spread_over_the_calibration_operators_matched_spread(spread):
    rng = np.random.default_rng(1)
    scores = {"a": rng.exponential(size=20000), "b": 2.0 * rng.exponential(size=20000)}
    cells = _cells(scores, 300, 0.10, 6, rng)
    table = spread.spread_ratios(cells, scores, {"a": 300, "b": 300}, 0.10)
    sd = cells.groupby(["calibration", "deployment"])["coverage"].std(ddof=1)
    row = table[(table["calibration"] == "a") & (table["deployment"] == "b")].iloc[0]
    assert row["measured_ratio"] == pytest.approx(sd[("a", "b")] / sd[("a", "a")])
    assert len(table) == 2 and (table["calibration"] != table["deployment"]).all()


def test_the_law_holds_on_cells_drawn_from_it(spread):
    rng = np.random.default_rng(2)
    scales = {"a": 1.0, "b": 0.6, "c": 1.8}
    scores = {k: s * rng.exponential(size=60000) for k, s in scales.items()}
    table = spread.spread_ratios(_cells(scores, 2000, 0.10, 400, rng), scores,
                                 dict.fromkeys(scores, 2000), 0.10)
    a = spread.agreement(table)
    assert a["rho"] > 0.9
    assert a["median_abs_log_error"] < 0.15
    assert a["same_side_of_one"] == 1.0


def test_agreement_declines_on_too_few_pairs(spread):
    table = pd.DataFrame({"measured_ratio": [1.0, 2.0], "predicted_ratio": [1.0, 2.0]})
    assert spread.agreement(table) == {"n": 2.0}


def test_a_subjects_side_does_not_depend_on_the_operator(spread):
    subjects = [f"s{i:04d}" for i in range(40)]
    raw = pd.DataFrame([{"sample_id": s, "operator_id": op} for s in subjects for op in "xyz"])
    same, size = spread.shared_calibration_cohort(raw, "sample_id", 0.5, 0.10)
    expected = sum(split_for(s, calib_fraction=0.5, design_fraction=0.10,
                             salt=spread.PROJECT_SALT) == "calib" for s in subjects)
    assert same and size == expected
    # an operator missing a calibration subject no longer shares the cohort
    calib = next(s for s in subjects if split_for(s, calib_fraction=0.5, design_fraction=0.10,
                                                  salt=spread.PROJECT_SALT) == "calib")
    short = raw[~((raw["sample_id"] == calib) & (raw["operator_id"] == "z"))]
    assert not spread.shared_calibration_cohort(short, "sample_id", 0.5, 0.10)[0]
