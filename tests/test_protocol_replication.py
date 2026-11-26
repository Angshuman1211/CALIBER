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


# -- T1-E: the circularity section's own arithmetic (D105) ---------------------
#
# The section's reading rests on two things that are algebra, not measurement, and on a
# 2x2 of margins. All three are recomputed here so the prose cannot drift from them.

#: T2.11 section 3a, medians over the nine operators, q90 label. cos theta from T2.10
#: section 6a (face +0.073 / BraTS +0.596 on M4Raw) and T2-C (+0.882 on the full LIDC bank).
#: (prior_class, cos_theta, spread_auroc, s_free_auroc, s_range_auroc)
_T1E_BANKS = {
    "fastmri_face": ("face", 0.073, 0.932, 0.978, 0.427),
    "ct_face": ("face", 0.074, 0.994, 0.997, 0.371),
    "m4raw_brats": ("medical", 0.596, 0.937, 0.895, 0.847),
    "ct_lidc": ("medical", 0.882, 0.920, 0.801, 0.759),
}


def test_the_criterion_reduces_to_the_score_exactly_at_zero_alignment():
    """The identity is why circularity is structural rather than a bad label choice."""
    rng = np.random.default_rng(11)
    s = rng.uniform(0.1, 3.0, 500)
    c = rng.uniform(0.1, 3.0, 500)
    cos_t = rng.uniform(-1.0, 1.0, 500)

    err = np.sqrt(np.maximum(s**2 + c**2 - 2.0 * s * c * cos_t, 0.0))
    assert np.all(np.isfinite(err))

    # At cos theta = 0 and only there, the criterion is hypot(score, content).
    err0 = np.sqrt(s**2 + c**2 - 2.0 * s * c * 0.0)
    assert np.allclose(err0, np.hypot(s, c), atol=1e-12)
    far = np.abs(cos_t) > 0.5
    assert np.all(np.abs(err[far] - np.hypot(s[far], c[far])) > 1e-6), (
        "away from zero the identity must NOT hold, or the test proves nothing")


def test_the_identity_predicts_which_null_wins_and_both_face_banks_agree():
    """At cos theta ~ 0 the error IS the null-space magnitude, so brightness carries
    nothing. Measured: s_free near-perfect, s_range below chance. Both face banks."""
    for name, (cls, cos_t, _spread, s_free, s_range) in _T1E_BANKS.items():
        if cls != "face":
            continue
        assert cos_t < 0.10, name
        assert s_free > 0.97, name
        assert s_range < 0.50, (
            "%s: the measured component must be below chance where the identity holds" % name)


def test_the_margin_sign_is_set_by_the_prior_and_not_the_modality():
    """Two modalities per prior class, and the sign follows the class both times."""
    signs = {}
    for name, (cls, _cos, spread, s_free, _sr) in _T1E_BANKS.items():
        signs.setdefault(cls, []).append(round(s_free - spread, 4))

    assert len(signs["face"]) == 2 and len(signs["medical"]) == 2
    assert all(m > 0 for m in signs["face"]), signs["face"]
    assert all(m < 0 for m in signs["medical"]), signs["medical"]
    assert max(signs["face"]) == pytest.approx(0.046, abs=5e-4)
    assert min(signs["medical"]) == pytest.approx(-0.119, abs=5e-4)


def test_the_posterior_is_credited_with_the_margin_and_not_the_level():
    """Quoting 0.937 or 0.920 credits the posterior with what a scale already delivers."""
    for name in ("m4raw_brats", "ct_lidc"):
        _cls, _cos, spread, s_free, s_range = _T1E_BANKS[name]
        best_null = max(s_free, s_range)
        contribution = spread - best_null
        assert 0.0 < contribution < 0.13, name
        assert contribution < spread / 4.0, (
            "%s: the level is at least four times the contribution, so the level is not "
            "the result" % name)
    m4 = _T1E_BANKS["m4raw_brats"]
    lidc = _T1E_BANKS["ct_lidc"]
    assert m4[2] - max(m4[3], m4[4]) == pytest.approx(0.042, abs=5e-4)
    assert lidc[2] - max(lidc[3], lidc[4]) == pytest.approx(0.119, abs=5e-4)


def test_no_correlation_across_the_four_banks_is_claimed():
    """Four banks, every covariate moving together: the section says so and must keep
    saying so. This is D102's lesson applied before the statistic is computed."""
    doc = (Path(__file__).resolve().parents[1] / "Documentation"
           / "T1.7_manuscript_method_and_analysis.md").read_text(encoding="utf-8")
    assert "No correlation is computed across the four banks, deliberately" in doc
    assert "almost no resolution" in doc
    assert len(_T1E_BANKS) == 4, "if a fifth bank lands, revisit the refusal rather than "\
        "the wording"


def test_the_full_image_spread_limit_is_stated_on_the_stamped_row():
    """The literature's own quantity has not been run on the bank that carries claims."""
    doc = (Path(__file__).resolve().parents[1] / "Documentation"
           / "T1.7_manuscript_method_and_analysis.md").read_text(encoding="utf-8")
    assert "has not been run\n   on the stamped bank" in doc
    assert "0.008 AUROC" in doc, "the substitution's validation must travel with the limit"
    assert "single largest limit on the headline row" in doc


def test_weakening_the_circularity_did_not_deliver_detection():
    """The section must not read as rescuing the score. Two failures, not one."""
    doc = (Path(__file__).resolve().parents[1] / "Documentation"
           / "T1.7_manuscript_method_and_analysis.md").read_text(encoding="utf-8")
    assert "1 of 8** operators on\n  M4Raw and **0 of 9** on CT" in doc
    assert "two separate failures" in doc
