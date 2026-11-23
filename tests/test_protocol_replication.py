"""The literature's validation protocol and its null control (T2-D, D-070 point 5).

The script's claim is a comparison: a detector with no posterior information against the
posterior spread, on one label, in one pass. These tests pin the three things that could
make that comparison say something it should not -- a label valued on the wrong rows, an
orientation that flips, and an interval that holds the label fixed while the rows move.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def protocol():
    path = Path(__file__).resolve().parents[1] / "scripts" / "protocol_replication.py"
    spec = importlib.util.spec_from_file_location("protocol_replication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(n=400, seed=0):
    """Regions whose error scales with their content, and a spread that knows nothing
    more than the content does: the circular case in miniature."""
    rng = np.random.default_rng(seed)
    content = rng.lognormal(0.0, 1.0, n)
    return pd.DataFrame({
        "patient_id": np.repeat(np.arange(n // 20), 20),
        "err_total": content * rng.lognormal(0.0, 0.2, n),
        "s_disagree": content * rng.lognormal(0.0, 0.2, n),
        "s_free": content,
        "s_range": rng.lognormal(0.0, 1.0, n),
    })


def test_spearman_is_a_rank_statistic(protocol):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert protocol.spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert protocol.spearman(x, -x ** 3) == pytest.approx(-1.0)
    assert np.isnan(protocol.spearman(x, np.ones(5)))


def test_the_high_error_label_is_the_top_tenth_of_these_rows(protocol):
    err = np.arange(100.0)
    label = protocol.high_error(err, 0.90)
    assert label.sum() == 10 and label[-10:].all()
    # valued on the rows it is given: a second operator with ten times the error still
    # has a tenth of its regions labelled, not all of them
    assert protocol.high_error(err * 10, 0.90).sum() == 10


def test_orientation_is_fixed_and_a_backwards_detector_scores_below_half(protocol):
    """A larger value predicts a larger error. A detector that runs the other way is
    reported below 0.5, never flipped to look informative."""
    df = _rows()
    df["backwards"] = -df["err_total"]
    row = protocol.protocol_row(df, ["s_free", "backwards"])
    assert row["auroc90_backwards"] == pytest.approx(0.0)
    assert row["rho_backwards"] == pytest.approx(-1.0)
    assert row["auroc90_s_free"] > 0.9


def test_a_scale_only_control_matches_a_spread_that_only_knows_scale(protocol):
    """The case the script exists to expose. The control has no posterior information and
    the protocol cannot tell it from the spread."""
    row = protocol.protocol_row(_rows(), ["s_disagree", "s_free", "s_range"])
    assert abs(row["auroc90_s_free"] - row["auroc90_s_disagree"]) < protocol.MATCH_MARGIN
    assert row["auroc90_s_range"] == pytest.approx(0.5, abs=0.12)   # unrelated to the error


def test_the_interval_revalues_the_label_inside_each_resample(protocol):
    """One cluster holds every high-error region. With the label fixed from the full
    table, a resample that misses that cluster has no positives and its AUROC is
    undefined; re-valued, every resample has its own top tenth and every draw counts."""
    df = _rows(n=200)
    df.loc[df["patient_id"] == 0, "err_total"] += 1e3
    lo, hi = protocol.cluster_interval(df, "patient_id", "s_free", "s_disagree",
                                       n_boot=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    # same seed, same rows: the interval is reproducible
    assert (lo, hi) == protocol.cluster_interval(df, "patient_id", "s_free", "s_disagree",
                                                 n_boot=200, seed=0)


def test_the_control_columns_are_ones_a_bank_has_and_need_no_draws(protocol):
    """`s_free` and `s_range` are functions of the mean reconstruction. If a control
    were ever swapped for a column that reads the draws, it would stop being a control."""
    from caliber.scorebank.schema import COLUMNS

    assert set(protocol.CONTROLS) == {"s_free", "s_range"}
    assert set(protocol.CONTROLS) <= set(COLUMNS)
