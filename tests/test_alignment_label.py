"""Gates for the direction-based label (D74).

The label exists because of one measured fact: by D38's identity, at ``cos θ ≈ 0``,
``err_free ≈ sqrt(s_free² + free_content²)``, so every magnitude label is a function of
the score's own dominant component. The tests therefore have to pin two things:

* the **geometry** -- that a planted ``cos θ`` comes back, with the right sign, and that
  a recovered-anatomy region is not labelled while an invented one is;
* the **trap** -- that restoring a magnitude factor to the label puts the tautology back.
  That one is a test rather than a comment because it is the mistake a reader of the
  module would most plausibly make while "improving" it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation import derive_score_and_label, label_spec
from caliber.scoring.attribution import (
    ALIGNMENT_QUANTILE,
    MIN_ALIGNMENT_SPREAD,
    alignment_threshold,
    label_misaligned,
    rank_auc,
    spearman,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import label_comparison  # noqa: E402


def _from_geometry(s: np.ndarray, c: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """``err_free`` built forwards from the identity, so ``cos θ`` is known exactly."""
    return np.sqrt(np.maximum(s**2 + c**2 - 2.0 * s * c * cos, 0.0))


# -- the geometry round-trips --------------------------------------------------------


def test_a_planted_alignment_comes_back():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.5, 2.0, 400)
    c = rng.uniform(0.5, 2.0, 400)
    cos = rng.uniform(-0.9, 0.9, 400)
    e = _from_geometry(s, c, cos)
    out = label_misaligned(s, c, e, threshold=0.0)
    assert np.allclose(out.cos_theta, cos, atol=1e-9)
    assert np.array_equal(out.label, cos < 0.0)


def test_recovered_anatomy_is_not_labelled_and_invention_is():
    """The project's own definition (Sec. 4): content in the free component that matches
    the truth is reconstruction, not hallucination, and a flag firing on it is a false
    positive. The label has to encode that distinction, which is the thing a magnitude
    label cannot do -- both cases can carry identical ``err_free``."""
    s = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0])
    recovered, invented = 0.95, -0.95
    e = _from_geometry(s, c, np.array([recovered, invented]))
    out = label_misaligned(s, c, e, threshold=0.0)
    assert not out.label[0], "aligned content is recovered anatomy"
    assert out.label[1], "anti-aligned content is invention"


def test_an_undefined_alignment_is_negative_and_counted_not_dropped():
    """Dropping them would make the label's support depend on score magnitude, which is
    the same confound by another route."""
    s = np.array([1.0, 0.0, 1.0])
    c = np.array([1.0, 1.0, 0.0])
    e = np.array([0.5, 1.0, 1.0])
    out = label_misaligned(s, c, e, threshold=0.9)
    assert out.n_total == 3 and out.n_defined == 1
    assert out.label.sum() == 1 and not out.label[1] and not out.label[2]
    assert out.defined_fraction == pytest.approx(1 / 3)


def test_a_flat_alignment_is_called_degenerate():
    """If every region has the same ``cos θ``, a quantile cut on it slices noise."""
    s = np.full(200, 1.0)
    c = np.full(200, 1.0)
    e = _from_geometry(s, c, np.full(200, 0.05))
    out = label_misaligned(s, c, e, threshold=0.05)
    assert out.spread < MIN_ALIGNMENT_SPREAD
    assert out.degenerate


def test_a_bank_with_real_spread_is_not_degenerate():
    rng = np.random.default_rng(1)
    s, c = rng.uniform(0.5, 2.0, 300), rng.uniform(0.5, 2.0, 300)
    cos = rng.normal(0.07, 0.09, 300).clip(-0.99, 0.99)
    out = label_misaligned(s, c, cos * 0 + _from_geometry(s, c, cos), threshold=0.0)
    assert not out.degenerate


# -- the trap this label exists to avoid, and the one that would undo it ------------


def test_a_magnitude_label_is_near_determined_by_the_score_and_the_direction_label_is_not():
    """The measurement behind D67 and D74, as a gate.

    With ``cos θ`` concentrated near zero -- which is what the mismatched prior produces,
    and what the M4Raw bank shows (sd 0.089) -- ``err_free`` is almost exactly
    ``hypot(s_free, free_content)``. The direction label must be much less tied to
    ``s_free`` than the magnitude label is, or it is not an improvement.
    """
    rng = np.random.default_rng(2)
    n = 2000
    s, c = rng.uniform(0.3, 2.5, n), rng.uniform(0.3, 2.5, n)
    cos = rng.normal(0.07, 0.09, n).clip(-0.99, 0.99)
    e = _from_geometry(s, c, cos)

    magnitude = e > np.quantile(e, 0.75)
    alignment = label_misaligned(s, c, e, np.quantile(cos, ALIGNMENT_QUANTILE)).label

    rho_magnitude = abs(spearman(e, np.hypot(s, c)))
    rho_direction = abs(spearman(cos, s))
    assert rho_magnitude > 0.95, f"the premise of D67 should hold here: {rho_magnitude}"
    assert rho_direction < rho_magnitude / 2.0, (
        f"the direction label is no less score-determined: {rho_direction} vs {rho_magnitude}"
    )
    assert rank_auc(s, magnitude) > rank_auc(s, alignment), (
        "s_free should rank the magnitude label better than the direction label"
    )


def test_restoring_the_magnitude_puts_the_tautology_back():
    """``s_free * sin θ`` -- invented *mass* -- looks more physical and is a trap:
    ``sin θ ≈ 1`` wherever ``cos θ ≈ 0``, so the product collapses to ``s_free``.

    This is a test and not a comment because it is the plausible 'improvement' that would
    silently undo the label."""
    rng = np.random.default_rng(3)
    n = 2000
    s, c = rng.uniform(0.3, 2.5, n), rng.uniform(0.3, 2.5, n)
    cos = rng.normal(0.07, 0.09, n).clip(-0.99, 0.99)

    invented_mass = s * np.sqrt(1.0 - cos**2)
    assert abs(spearman(invented_mass, s)) > 0.95, (
        "sin theta is near 1 throughout, so the product is s_free again"
    )
    assert abs(spearman(cos, s)) < 0.2, "while the direction alone is not"


# -- the design-split discipline (D-051, P-19, P-23) -------------------------------


def test_the_cut_is_refused_without_a_design_split():
    df = _frame(split="calib")
    with pytest.raises(ValueError, match="design split"):
        label_spec(df, "alignment")


def test_the_cut_comes_from_the_design_rows_only():
    """Fitting it on calib or test rows lets the label see the patients whose coverage it
    is about to score. Shifting only the design rows must move the cut."""
    base = _frame()
    shifted = base.copy()
    design = shifted["split"] == "design"
    shifted.loc[design, "err_free"] = shifted.loc[design, "err_free"] * 1.5
    assert label_spec(base, "alignment").kwargs["alignment_threshold"] != pytest.approx(
        label_spec(shifted, "alignment").kwargs["alignment_threshold"]
    )


def test_a_quantile_outside_the_unit_interval_is_refused():
    df = _frame()
    with pytest.raises(ValueError, match="quantile"):
        alignment_threshold(df["s_free"], df["free_content"], df["err_free"], 1.5)


def test_a_bank_missing_s_free_is_refused_with_the_reason():
    df = _frame().drop(columns=["s_free"])
    with pytest.raises(ValueError, match="s_free"):
        label_spec(df, "alignment")


# -- it reaches the frame through the one derivation path (P-19) --------------------


def _frame(split: str | None = None, n: int = 600, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s, c = rng.uniform(0.3, 2.5, n), rng.uniform(0.3, 2.5, n)
    cos = rng.normal(0.07, 0.12, n).clip(-0.99, 0.99)
    df = pd.DataFrame({
        "operator_id": np.where(rng.random(n) < 0.5, "op.a", "op.b"),
        "sample_id": [f"v{i % 20}" for i in range(n)],
        "patient_id": [f"p{i % 20}" for i in range(n)],
        "region_id": np.arange(n) % 30,
        "slice_idx": 0,
        "s_free": s,
        "free_content": c,
        "err_free": _from_geometry(s, c, cos),
        "err_total": _from_geometry(s, c, cos) + 0.1,
        "s_disagree": s * 0.5 + rng.normal(0, 0.1, n),
        "s_residual": rng.normal(0, 0.1, n),
        "ann_coverage": 1.0,
    })
    df["split"] = split if split else np.where(
        np.arange(n) % 5 == 0, "design", np.where(np.arange(n) % 5 < 3, "calib", "test")
    )
    return df


def test_the_label_reaches_the_frame_and_carries_cos_theta():
    df = _frame()
    spec = label_spec(df, "alignment")
    out = derive_score_and_label(df, "op.a", **spec.kwargs)
    assert out["label"].dtype == bool
    assert "cos_theta" in out.columns
    assert 0 < out["label"].sum() < len(out)
    assert "cos theta" in spec.text and "D74" in spec.text


def test_the_positive_rate_is_the_quantile_by_construction():
    """P-02's failure was a fixed constant that flagged every region. A quantile cut
    cannot, and the rate is known before the bank is read."""
    df = _frame(n=2000)
    spec = label_spec(df, "alignment")
    out = derive_score_and_label(df, "op.a", **spec.kwargs)
    assert out["label"].mean() == pytest.approx(ALIGNMENT_QUANTILE, abs=0.06)


def test_the_alignment_label_is_listed_and_selectable_by_name():
    from caliber.evaluation import LABELS

    assert "alignment" in LABELS
    assert "absolute" in LABELS, "D-061's label stays the reported one until D74 is ruled"


# -- the entry point ---------------------------------------------------------------


def test_the_script_compares_both_labels_per_operator(tmp_path, capsys):
    bank = tmp_path / "bank.parquet"
    _frame(n=3000, seed=5).to_parquet(bank, index=False)
    out_csv = tmp_path / "out.csv"
    code = label_comparison.main([
        str(bank), "--calibration-operator", "op.a", "--out", str(out_csv),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "identity check" in out
    assert "absolute" in out and "alignment" in out
    table = pd.read_csv(out_csv)
    assert set(table["label"]) == {"absolute", "alignment"}
    assert table["operator_id"].nunique() >= 2, "one row per operator, never pooled"


def test_the_script_refuses_a_bank_that_fails_the_identity(tmp_path, capsys):
    """cos theta is only recoverable if the three columns share one normalisation. A bank
    whose truth and score columns were scaled differently must be refused, not read."""
    df = _frame(n=600, seed=6)
    df["err_free"] = df["err_free"] * 3.0 + 1.0
    bank = tmp_path / "bad.parquet"
    df.to_parquet(bank, index=False)
    code = label_comparison.main([str(bank), "--calibration-operator", "op.a"])
    assert code == 1
    assert "identity" in capsys.readouterr().err


def test_the_script_refuses_a_bank_without_the_null_space_columns(tmp_path, capsys):
    bank = tmp_path / "thin.parquet"
    _frame(n=200).drop(columns=["s_free"]).to_parquet(bank, index=False)
    code = label_comparison.main([str(bank), "--calibration-operator", "op.a"])
    assert code == 1
    assert "s_free" in capsys.readouterr().err
