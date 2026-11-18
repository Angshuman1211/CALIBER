"""The CT distance table behind Figure 5 (T2.8, T2.11).

`ct_distances.py --all-pairs` writes each unordered pair once. The figure plots naive
transfer in both directions, so it needs both orders, and a pair it cannot find is
silently left off the plot. These pin that every direction is supplied and that a
measured value is never replaced by a derived one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def figures():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(tmp_path, rows):
    path = tmp_path / "distances.csv"
    pd.DataFrame(rows, columns=["calibration", "deployment", "measure", "value"]).to_csv(
        path, index=False)
    return path


def test_a_pair_measured_once_is_available_in_both_directions(figures, tmp_path):
    csv = _table(tmp_path, [("v180", "v30", "null_space_growth", 0.60),
                            ("v180", "v30", "null_space_projection", 0.38)])
    dist = figures.ct_distances_from(csv)
    assert dist[("v180", "v30")] == pytest.approx(0.60)
    assert dist[("v30", "v180")] == pytest.approx(-0.60)   # signed: the reverse loses it


def test_a_measured_direction_is_not_overwritten_by_the_derived_one(figures, tmp_path):
    csv = _table(tmp_path, [("v180", "v30", "null_space_growth", 0.6016),
                            ("v30", "v180", "null_space_growth", -0.6011)])
    dist = figures.ct_distances_from(csv)
    assert dist[("v30", "v180")] == pytest.approx(-0.6011)
    assert dist[("v180", "v30")] == pytest.approx(0.6016)


def test_only_the_signed_measure_is_read(figures, tmp_path):
    """The projection distance is symmetric and unsigned; flipping its sign would put
    half the points at negative distances that mean nothing."""
    csv = _table(tmp_path, [("v180", "v30", "null_space_projection", 0.38)])
    assert figures.ct_distances_from(csv) == {}
