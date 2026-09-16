"""Deriving scores and labels from a bank, and being honest about precision (T2.4).

The precision half exists because of P-23: regions are clustered within volumes and
volumes within patients, so an interval computed over regions treats thousands of
correlated observations as independent and comes out far too narrow. The first test here
measures that gap rather than asserting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation import (
    cluster_bootstrap_coverage,
    clusters_for_precision,
    derive_score_and_label,
)
from caliber.scoring.score import ScoreWeights


def _bank(n_volumes=12, regions=64, seed=0):
    """A bank-shaped frame whose regions are correlated within each volume."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_volumes):
        volume_effect = rng.normal(0, 1.0)          # what makes regions non-independent
        for r in range(regions):
            free = abs(volume_effect + rng.normal(0, 0.2)) + 0.1
            rows.append({
                "operator_id": "op", "sample_id": f"vol{v:02d}", "region_id": r,
                "s_free": free, "s_disagree": free * 0.5, "s_residual": 0.05,
                "err_free": free * 2.2, "free_content": free,
                "err_total": free * 2.3, "ann_coverage": 0.0,
            })
    return pd.DataFrame(rows)


# -- deriving score and label ---------------------------------------------------


def test_score_and_label_are_added_from_cached_columns():
    df = derive_score_and_label(_bank(), "op", threshold=1.75)
    assert {"score", "label"} <= set(df.columns)
    assert df["score"].notna().all()
    assert df["label"].dtype == bool


def test_an_unknown_calibration_operator_is_refused():
    with pytest.raises(ValueError, match="not in this bank"):
        derive_score_and_label(_bank(), "no-such-operator")


def test_scales_come_from_the_calibration_operator_only(monkeypatch):
    """D27: re-estimating scales per operator would shrink the gap being measured."""
    df = _bank()
    other = df.copy()
    other["operator_id"] = "other"
    other["s_free"] = other["s_free"] * 100          # a wildly different deployment operator
    both = pd.concat([df, other], ignore_index=True)

    derived = derive_score_and_label(both, "op", threshold=1.75)
    # The calibration operator's own scores must not depend on what else is in the bank.
    alone = derive_score_and_label(df, "op", threshold=1.75)
    assert np.allclose(
        derived[derived["operator_id"] == "op"]["score"].to_numpy(),
        alone["score"].to_numpy(),
    )


def test_the_default_threshold_is_track_1s_ratified_tau():
    import inspect

    assert inspect.signature(derive_score_and_label).parameters["threshold"].default == 1.75


def test_weights_are_honoured():
    df = _bank()
    a = derive_score_and_label(df, "op", weights=ScoreWeights.only("free"))
    b = derive_score_and_label(df, "op", weights=ScoreWeights.only("residual"))
    assert not np.allclose(a["score"].to_numpy(), b["score"].to_numpy())


# -- the interval that matters --------------------------------------------------


def test_clustering_widens_the_interval_a_lot():
    """The whole reason this module exists (P-23).

    Treating each region as its own cluster is the naive interval. Clustering by volume
    is the honest one. On data with a volume effect, the honest interval must be
    substantially wider — otherwise a bank of correlated regions looks like a large
    independent sample and every coverage claim is overconfident.
    """
    df = derive_score_and_label(_bank(), "op", threshold=1.75)
    flags = df["score"] > df["score"].median()

    by_region = cluster_bootstrap_coverage(flags, df["label"], np.arange(len(df)))
    by_volume = cluster_bootstrap_coverage(flags, df["label"], df["sample_id"])

    assert by_volume["se"] > 3 * by_region["se"], (by_volume["se"], by_region["se"])
    assert by_volume["n_clusters"] == 12
    assert by_region["n_clusters"] == len(df)


def test_coverage_point_estimate_is_the_pooled_ratio():
    df = derive_score_and_label(_bank(), "op", threshold=1.75)
    flags = df["score"] > df["score"].median()
    stats = cluster_bootstrap_coverage(flags, df["label"], df["sample_id"])
    expected = float((flags & df["label"]).sum() / df["label"].sum())
    assert stats["coverage"] == pytest.approx(expected)
    assert stats["lo"] <= stats["coverage"] <= stats["hi"]


def test_a_bank_with_no_positives_reports_nan_rather_than_a_number():
    df = _bank()
    stats = cluster_bootstrap_coverage(
        np.ones(len(df), bool), np.zeros(len(df), bool), df["sample_id"]
    )
    assert np.isnan(stats["coverage"]) and stats["n_positive"] == 0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_coverage([True], [True, False], ["a", "b"])


def test_the_bootstrap_is_deterministic():
    df = derive_score_and_label(_bank(), "op", threshold=1.75)
    flags = df["score"] > df["score"].median()
    a = cluster_bootstrap_coverage(flags, df["label"], df["sample_id"], seed=3)
    b = cluster_bootstrap_coverage(flags, df["label"], df["sample_id"], seed=3)
    assert a == b


# -- extrapolating to a sample size ---------------------------------------------


def test_precision_scales_as_one_over_root_n():
    """Halving the interval costs four times the patients — the Week 4 arithmetic."""
    wide = clusters_for_precision(0.12, 12, 0.10)
    narrow = clusters_for_precision(0.12, 12, 0.05)
    assert narrow == pytest.approx(4 * wide, rel=0.05)


def test_more_clusters_are_needed_for_a_tighter_interval():
    assert clusters_for_precision(0.12, 12, 0.02) > clusters_for_precision(0.12, 12, 0.10)


@pytest.mark.parametrize(("se", "target"), [(0.0, 0.1), (float("nan"), 0.1), (0.1, 0.0)])
def test_degenerate_inputs_are_refused(se, target):
    with pytest.raises(ValueError, match="positive"):
        clusters_for_precision(se, 12, target)
