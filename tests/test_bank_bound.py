"""T3.6: the bound-versus-measurement pipeline, on a hand-built crossed bank with known structure.

The harness-built banks are exercised by ``scripts/t3_6_bound_vs_measurement.py`` itself (and
tested below by running it); these gates use a table whose answer is known in advance, so a
change to the pairing, the split handling or the bound arithmetic fails here rather than
producing a plausible-looking number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation.bank import label_spec
from caliber.theory.bank_bound import KEYS, max_density, pair_frame, pair_report

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def make_bank(shift: float = 0.0, n_items: int = 60, r: int = 16, seed: int = 0) -> pd.DataFrame:
    """Two operators over the same items. ``dst`` = ``src`` with scores/errors shifted."""
    rng = np.random.default_rng(seed)
    splits = ["design"] * 9 + ["calib"] * 27 + ["test"] * (n_items - 36)
    rows = []
    for i in range(n_items):
        base = rng.normal(1.0, 0.3, r) + rng.normal(0, 0.5)  # shared item effect
        err = np.abs(base + rng.normal(0, 0.2, r))
        for op, sh in (("mri.a", 0.0), ("mri.b", shift)):
            rows.append(pd.DataFrame({
                "operator_id": op, "split": splits[i], "sample_id": f"i{i}",
                "patient_id": f"p{i}", "slice_idx": 0, "region_id": range(r),
                "s_free": np.abs(base + sh) + 0.05, "s_disagree": np.abs(base + sh) * 0.5 + 0.05,
                "s_residual": 0.0, "err_total": err, "err_free": err + sh,
                "free_content": 0.5, "ann_coverage": 0.0,
            }))
    return pd.concat(rows, ignore_index=True)


def report(df, d_op=0.5, **kw):
    return pair_report(df, "mri.a", "mri.b", label_spec(df), d_op, n_boot=40, **kw)


def test_pair_frame_couples_the_same_item_and_region():
    df = make_bank(0.3)
    d = df.assign(score=df["s_free"], label=True)
    pf = pair_frame(d, "mri.a", "mri.b")
    assert len(pf) == 60 * 16
    assert set(KEYS) <= set(pf.columns)
    assert (pf["score_b"] - pf["score_a"]).round(9).eq(0.3).mean() > 0.9  # |.| keeps most signs


def test_pair_frame_refuses_a_bank_that_is_not_crossed():
    df = make_bank(0.0)
    df.loc[df.operator_id == "mri.b", "sample_id"] = "other" + df["sample_id"]
    d = df.assign(score=df["s_free"], label=True)
    with pytest.raises(ValueError, match="not crossed"):
        pair_frame(d, "mri.a", "mri.b")


def test_identical_operators_have_nothing_to_bound():
    r = report(make_bank(0.0))
    assert r.ks_cond == pytest.approx(0.0, abs=1e-9)
    assert r.p_flip == 0.0 and r.crossing == pytest.approx(0.0, abs=1e-9)
    assert r.algebra_bound == pytest.approx(0.0, abs=1e-9)
    assert abs(r.gap) < 0.15  # only held-out sampling noise (27 calib / 24 test items)


def test_a_downward_shift_undercovers_and_the_bound_holds():
    r = report(make_bank(-0.6))  # dst scores sit below src's: the src threshold misses at dst
    assert r.ks_cond > 0.2
    assert r.gap > 0 and r.holds_ks


def test_an_upward_shift_overcovers_and_the_bound_is_not_about_it():
    r = report(make_bank(0.6))  # dst scores sit above src's: flag area grows, coverage rises
    assert r.gap < 0
    assert r.ks_cond > 0.2  # the distance is symmetric; the gap is not (D47)


def test_the_label_decomposition_bound_holds_in_both_directions():
    df = make_bank(-0.6)
    for src, dst in (("mri.a", "mri.b"), ("mri.b", "mri.a")):
        r = pair_report(df, src, dst, label_spec(df), 0.5, n_boot=20)
        assert r.ks_cond <= r.algebra_bound + 1e-9


def test_design_rows_set_the_label_but_not_the_laws():
    df = make_bank(0.0)
    df.loc[df.split == "design", ["s_free", "s_disagree"]] += 50.0  # absurd design scores
    clean = report(make_bank(0.0))
    contaminated = report(df)
    assert contaminated.ks_cond == pytest.approx(clean.ks_cond, abs=1e-9)


def test_max_density_of_a_standard_normal_is_near_the_known_peak():
    x = np.random.default_rng(1).normal(size=20_000)
    assert max_density(x) == pytest.approx(0.3989, abs=0.04)


def test_the_script_runs_end_to_end_on_a_harness_bank(capsys, tmp_path):
    import t3_6_bound_vs_measurement as script

    out = tmp_path / "t.csv"
    script.main([
        "--synthetic", "mri", "--solver", "noise", "--items", "14", "--size", "32",
        "--n-boot", "10", "--calibration-operator", "mri.random1d.R2.c0.04.s1",
        "--out", str(out),
    ])
    text = capsys.readouterr().out
    assert "label: err_free >" in text and "gap vs d_K^+" in text
    assert len(pd.read_csv(out)) == 3
