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

from caliber.calibration.conformal import split_threshold
from caliber.evaluation.bank import derive_score_and_label, label_spec
from caliber.theory.bank_bound import (
    HIST_BINS,
    KEYS,
    _bootstrap,
    conformal_threshold,
    coupled_w1_cond_bound,
    distance_correlations,
    histogram_tv,
    max_density,
    pair_frame,
    pair_report,
)
from caliber.theory.stats import ks_distance, w1_distance

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


# -- T3-A: the refit bootstrap, the matched gap, the bound check and rows 3-6 ------------


def test_conformal_threshold_is_split_thresholds_rule_exactly():
    rng = np.random.default_rng(3)
    for n in (0, 1, 5, 8, 9, 10, 37, 400):
        x = rng.normal(size=n)
        want = split_threshold(x, np.ones(n, bool), 0.1).threshold
        assert conformal_threshold(x, 0.1) == want


def test_every_bootstrap_replicate_refits_both_thresholds_on_its_own_patients():
    """Replicate 0 rebuilt by hand from the same draws: both sides move, both taus refit."""
    df = make_bank(-0.3, seed=5)
    lab = label_spec(df)
    d = derive_score_and_label(df, "mri.a", **lab.kwargs)
    pf = pair_frame(d, "mri.a", "mri.b")
    fit, read = pf[pf.split == "calib"], pf[pf.split == "test"]
    laws = pf[pf.split != "design"]
    reps = _bootstrap(fit, read, laws, "patient_id", 0.1, 0.9, 3, 11, tau_fixed=0.0)

    rng = np.random.default_rng(11)
    cal_keys, tst_keys = sorted(fit.patient_id.unique()), sorted(read.patient_id.unique())
    ci = rng.integers(0, len(cal_keys), size=(3, len(cal_keys)))
    ti = rng.integers(0, len(tst_keys), size=(3, len(tst_keys)))
    cal = pd.concat([fit[fit.patient_id == cal_keys[i]] for i in ci[0]])
    tst = pd.concat([read[read.patient_id == tst_keys[i]] for i in ti[0]])
    tau_a = split_threshold(cal.score_a, cal.label_a.astype(bool), 0.1).threshold
    tau_b = split_threshold(cal.score_b, cal.label_b.astype(bool), 0.1).threshold
    pos = tst.loc[tst.label_b.astype(bool), "score_b"]
    cov_ab, cov_bb = (pos >= tau_a).mean(), (pos >= tau_b).mean()
    assert reps["gap"][0] == pytest.approx(0.9 - cov_ab, abs=1e-12)
    assert reps["gap_matched"][0] == pytest.approx(cov_bb - cov_ab, abs=1e-12)
    assert reps["gap_fixed"][0] == pytest.approx(0.9 - (pos >= 0.0).mean(), abs=1e-12)


def test_null_fixture_the_refit_interval_covers_and_the_fixed_one_does_not():
    """D73's rule: run the null fixture before trusting a new interval.

    Identical operators, 40 banks: the truth of the nominal gap is 0 to within 1/(n+1). The
    refit interval contains it at about its level; holding the threshold fixed contains it
    less often, because it leaves out the calibration draw (measured: 38 vs 33 of 40, widths
    in the ratio ~1.4, i.e. sqrt(2) at balanced split sizes).
    """
    refit = fixed = 0
    ratios = []
    for seed in range(40):
        df = make_bank(0.0, n_items=60, seed=seed)
        r = pair_report(df, "mri.a", "mri.b", label_spec(df), 0.5, n_boot=300, seed=seed)
        refit += r.gap_lo <= 0 <= r.gap_hi
        fixed += r.gap_fixed_lo <= 0 <= r.gap_fixed_hi
        ratios.append(r.interval_width_ratio)
    assert refit >= 34 and refit > fixed
    assert 1.15 < np.median(ratios) < 1.8


def test_matched_gap_cancels_a_shift_common_to_both_operators():
    """Same operator twice: tau_src = tau_dst, so the matched gap is exactly zero."""
    r = report(make_bank(0.0))
    assert r.gap_matched == 0.0 and r.tau_src == r.tau_dst
    assert r.cov_matched == r.cov_dst
    assert r.slack == pytest.approx(1 / (r.n_pos_calib_dst + 1))
    assert r.holds_slack and r.tightness == 0.0


def test_transfer_cells_reproduce_track_2s_pair_table_under_one_scale_operator():
    """With scales frozen on one operator, cov_dst and cov_matched are T2-A3's cells."""
    import pair_coverage_by_split as t2

    df = make_bank(-0.4, seed=2)
    lab = label_spec(df)
    d = derive_score_and_label(df, "mri.a", **lab.kwargs)
    cells = pd.DataFrame(t2.pair_cells(t2.positives_by_cluster(d, "patient_id"),
                                       ["mri.a", "mri.b"], 0.1))
    cov = cells.set_index(["calibration", "deployment"])["coverage"]
    r = pair_report(df, "mri.b", "mri.a", lab, 0.5, n_boot=20, scale_operator="mri.a")
    assert r.cov_dst == pytest.approx(cov[("mri.b", "mri.a")], abs=1e-12)
    assert r.cov_matched == pytest.approx(cov[("mri.a", "mri.a")], abs=1e-12)


def test_the_coupled_bounds_are_theorems_on_arbitrary_couplings():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 300))
        sa = rng.normal(size=n) * rng.uniform(0.2, 3)
        sb = sa * rng.uniform(0.5, 1.5) + rng.normal(size=n) * rng.uniform(0, 2)
        la = rng.random(n) < rng.uniform(0.1, 0.9)
        flip = rng.random(n) < rng.uniform(0, 0.5)
        lb = np.where(flip, ~la, la)
        if not la.any() or not lb.any():
            continue
        assert w1_distance(sa, sb) <= np.abs(sa - sb).mean() + 1e-12
        assert w1_distance(sa[la], sb[lb]) <= coupled_w1_cond_bound(sa, la, sb, lb) + 1e-12


def test_the_conditional_coupled_bound_is_exact_for_a_pure_shift_with_no_flips():
    sa = np.random.default_rng(1).normal(size=500)
    lab = np.ones(500, bool)
    assert coupled_w1_cond_bound(sa, lab, sa + 0.3, lab) == pytest.approx(0.3)
    assert w1_distance(sa, sa + 0.3) == pytest.approx(0.3)


def test_rows_three_four_and_six_on_a_shifted_bank():
    r = report(make_bank(-0.6))
    assert r.holds_coupled and r.holds_coupled_cond
    assert 0 < r.w1_uncond <= r.w1_coupled
    assert r.w1_cond_lo <= r.w1_cond <= r.w1_cond_hi
    assert len(r.tv_hist) == len(HIST_BINS)
    assert all(t >= r.ks_cond * 0.5 for t in r.tv_hist)  # coarse partitions still see it
    row = r.as_row()
    assert {f"tv_h{b}" for b in HIST_BINS} <= set(row) and "tv_hist" not in row


def test_histogram_tv_extremes():
    x = np.linspace(0, 1, 101)
    assert histogram_tv(x, x, 20) == 0.0
    assert histogram_tv(np.zeros(10), np.ones(10), 20) == 1.0


def test_distance_correlations_are_reported_per_cell_and_pooled():
    ops = {"f.a1": "f", "f.a2": "f", "f.a3": "f", "g.b1": "g", "g.b2": "g", "g.b3": "g"}
    rng = np.random.default_rng(4)
    rows = []
    for a in ops:
        for b in ops:
            if a == b:
                continue
            d = float(rng.uniform(0.01, 1.0))
            sign = 1.0 if ops[a] == ops[b] else -1.0  # opposite trend across families
            rows.append({"src": a, "dst": b, "d_op": d, "ks_cond": sign * d})
    corr = distance_correlations(pd.DataFrame(rows), ops, xs=("d_op",), ys=("ks_cond",),
                                 n_boot=50)
    by = corr.set_index("scope")
    assert set(by.index) == {"pooled", "f->f", "f->g", "g->f", "g->g"}
    assert by.loc["f->f", "rho"] == pytest.approx(1.0)
    assert by.loc["f->g", "rho"] == pytest.approx(-1.0)
    assert by.loc["pooled", "n_pairs"] == 30 and by.loc["f->f", "n_pairs"] == 6
    agree = by["cells_agree_in_sign"].iloc[0]
    assert agree is not None and not agree  # four finite cells, two signs
    assert np.isfinite(by.loc["pooled", "lo"]) and np.isfinite(by.loc["pooled", "hi"])


def test_distances_csv_fills_the_reverse_order_by_symmetry(tmp_path):
    import t3_6_bound_vs_measurement as script

    path = tmp_path / "d.csv"
    pd.DataFrame([
        {"calibration": "a", "deployment": "b", "measure": "null_space_growth", "value": 0.3},
        {"calibration": "a", "deployment": "b", "measure": "null_space_projection",
         "value": 0.5},
        {"calibration": "a", "deployment": "c", "measure": "null_space_growth", "value": "nan"},
    ]).to_csv(path, index=False)
    d = script.load_distances(path)
    assert d["null_space_growth"][("b", "a")] == -0.3
    assert d["null_space_projection"][("b", "a")] == 0.5
    assert ("a", "c") not in d["null_space_growth"]


def test_the_script_runs_end_to_end_on_a_harness_bank(capsys, tmp_path):
    import t3_6_bound_vs_measurement as script

    out, corr = tmp_path / "t.csv", tmp_path / "c.csv"
    script.main([
        "--synthetic", "mri", "--solver", "noise", "--items", "14", "--size", "32",
        "--n-boot", "20", "--calibration-operator", "mri.random1d.R2.c0.04.s1",
        "--both-directions", "--out", str(out), "--out-corr", str(corr),
    ])
    text = capsys.readouterr().out
    assert "label: err_free >" in text and "rows 1-2" in text and "row 5" in text
    table = pd.read_csv(out)
    assert len(table) == 6
    assert table["holds_coupled"].all() and table["holds_coupled_cond"].all()
    assert {"gap_matched", "gap_matched_lo", "tightness", "w1_coupled"} <= set(table.columns)
    assert len(pd.read_csv(corr))


def test_quantile_bins_survive_an_outlier_that_collapses_equal_width_bins():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 4000), rng.normal(0.8, 1, 4000)
    b[0] = 1e4  # one far score stretches the pooled range
    ks = ks_distance(a, b)
    assert histogram_tv(a, b, 50, edges="uniform") < ks  # the defect the LIDC run showed
    assert histogram_tv(a, b, 50) >= 0.9 * ks
    with pytest.raises(ValueError):
        histogram_tv(a, b, 10, edges="log")
