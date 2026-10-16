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
    CANDIDACY_QUANTILE,
    LABEL_QUANTILE,
    candidacy_gate,
    cluster_bootstrap_coverage,
    clusters_for_precision,
    derive_score_and_label,
    design_quantile,
    label_spec,
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


# -- the candidacy gate (D-060) --------------------------------------------------


def test_the_candidacy_gate_unlabels_regions_that_hide_too_little():
    """A region whose free_content is at or below the gate cannot host a null-space
    hallucination, so it is never labelled, however large its ratio (D-060)."""
    df = _bank()
    gate = float(df["free_content"].median())
    gated = derive_score_and_label(df, "op", min_free_content=gate)
    ungated = derive_score_and_label(df, "op")
    assert ungated["label"].all()                      # every ratio is 2.2 > 1.75
    assert not gated.loc[df["free_content"] <= gate, "label"].any()
    assert gated.loc[df["free_content"] > gate, "label"].all()


def test_the_gate_does_not_touch_the_score():
    df = _bank()
    gated = derive_score_and_label(df, "op", min_free_content=float(df["free_content"].max()))
    ungated = derive_score_and_label(df, "op")
    assert np.array_equal(gated["score"].to_numpy(), ungated["score"].to_numpy())


def _split_bank():
    df = _bank()
    sides = {f"vol{v:02d}": side for v, side in
             zip(range(12), ["design"] * 2 + ["calib"] * 5 + ["test"] * 5, strict=True)}
    df["split"] = df["sample_id"].map(sides)
    return df


def test_the_gate_value_comes_from_the_design_split_only():
    """D-051: a label fitted on the patients it is about to score is in-sample."""
    df = _split_bank()
    expected = df.loc[df["split"] == "design", "free_content"].quantile(0.33)
    before = design_quantile(df, "free_content", 0.33)
    assert before == pytest.approx(expected)
    df.loc[df["split"] != "design", "free_content"] *= 100   # calib and test cannot move it
    assert design_quantile(df, "free_content", 0.33) == pytest.approx(before)


def test_a_bank_without_a_design_side_is_refused():
    df = _split_bank()
    df["split"] = df["split"].replace("design", "calib")
    with pytest.raises(ValueError, match="design split"):
        design_quantile(df, "free_content", 0.33)
    with pytest.raises(ValueError, match="design split"):
        design_quantile(df.drop(columns="split"), "free_content", 0.33)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 33])
def test_a_quantile_outside_the_open_unit_interval_is_refused(q):
    with pytest.raises(ValueError, match="quantile"):
        design_quantile(_split_bank(), "free_content", q)


def test_the_ruled_candidacy_quantile_is_d060s():
    """Every reporting script defaults to this constant, so it is pinned once, here."""
    assert CANDIDACY_QUANTILE == 0.33


def test_the_candidacy_gate_is_the_design_quantile_of_free_content():
    df = _split_bank()
    assert candidacy_gate(df) == design_quantile(df, "free_content", CANDIDACY_QUANTILE)
    assert candidacy_gate(df, 0.5) == design_quantile(df, "free_content", 0.5)


def test_quantile_zero_opts_out_of_the_gate():
    """The explicit route to the old label, which is how its numbers are reproduced."""
    assert candidacy_gate(_split_bank(), 0) is None
    assert candidacy_gate(_bank(), 0) is None            # no design side needed to opt out


def test_a_bank_without_a_design_side_is_told_how_to_opt_out():
    with pytest.raises(ValueError, match="candidacy quantile of 0"):
        candidacy_gate(_bank())


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


# -- D-061's absolute label --------------------------------------------------------


def test_the_reported_label_quantile_is_pinned():
    """D-061 names no quantile; q75 is the one its table tested. Pinned once, here."""
    assert LABEL_QUANTILE == 0.75


def test_the_absolute_label_is_err_free_above_its_design_quantile_and_nothing_else():
    df = _split_bank()
    spec = label_spec(df)
    eps = df.loc[df["split"] == "design", "err_free"].quantile(0.75)
    assert spec.name == "absolute" and spec.kwargs == {"threshold": eps, "relative": False}
    labelled = derive_score_and_label(df, "op", **spec.kwargs)
    assert (labelled["label"] == (df["err_free"] > eps)).all()   # no division, no gate
    df.loc[df["split"] != "design", "err_free"] *= 100           # calib and test cannot move it
    assert label_spec(df).kwargs["threshold"] == pytest.approx(eps)


def test_the_superseded_labels_reproduce_for_comparison():
    df = _split_bank()
    gate = design_quantile(df, "free_content", CANDIDACY_QUANTILE)
    assert label_spec(df, "candidacy").kwargs == {"threshold": 1.75, "relative": True,
                                                  "min_free_content": gate}
    assert label_spec(_bank(), "ratio").kwargs == {"threshold": 1.75, "relative": True}


def test_the_annotation_reading_keeps_only_annotated_positives():
    df = _split_bank()
    df["ann_coverage"] = 0.0
    df.loc[df.index[:100], "ann_coverage"] = 0.5
    spec = label_spec(df, require_annotation=True)
    assert spec.name == "absolute+annotation"
    labelled = derive_score_and_label(df, "op", **spec.kwargs)
    assert not labelled.loc[df.index[100:], "label"].any()
    plain = derive_score_and_label(df, "op", **label_spec(df).kwargs)
    assert (labelled["label"] <= plain["label"]).all()


def test_a_bank_without_a_design_side_is_told_to_read_the_ratio_label():
    with pytest.raises(ValueError, match="--label ratio"):
        label_spec(_bank())


def test_an_unknown_label_is_refused():
    with pytest.raises(ValueError, match="unknown label"):
        label_spec(_split_bank(), "nonsense")
