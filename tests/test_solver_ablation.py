"""Gates for D68's solver ablation.

This is the experiment that decides whether the score gets a positive claim at all, so the
tests are built around fixtures with planted answers in **both** directions -- a bank where
the decomposition earns its keep on the low-share solver, and one where it does not. A
verdict that cannot come back negative is not a verdict.

The share computation gets its own tests because it is the ablation's premise and because
the obvious way to compute it is wrong: ``P_N`` is non-local, so the orthogonal decomposition
does not hold region by region and the per-region ratio exceeds 1 on real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import solver_ablation  # noqa: E402
from solver_ablation import MIN_SHARE_GAP, null_error_share  # noqa: E402

CALIB_OP = "op.low"
OPS = (CALIB_OP, "op.high")


def _bank(
    solver_id: str,
    null_share: float,
    score_beats_baseline: bool,
    n_patients: int = 16,
    n_regions: int = 24,
    seed: int = 0,
) -> pd.DataFrame:
    """One solver arm with a planted null-error share and a planted detection advantage.

    ``err_free`` is built from ``err_total`` and the requested share, so the premise section
    has something definite to recover. ``s_free`` carries the hallucination signal only when
    ``score_beats_baseline``; otherwise the signal sits in ``s_disagree`` as well and the two
    tie, which is the DDNM situation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        split = "design" if p < 3 else "calib" if p < 8 else "test"
        for op in OPS:
            for r in range(n_regions):
                hallucinated = bool(rng.random() < 0.35)
                err_total = (2.0 if hallucinated else 0.4) + abs(rng.normal(0, 0.05))
                signal = 2.0 if hallucinated else 0.0
                rows.append({
                    "solver_id": solver_id,
                    "solver_family": "diffusion",
                    "patient_id": f"p{p:02d}",
                    "sample_id": f"p{p:02d}_v0",
                    "slice_idx": 0,
                    "region_id": r,
                    "region_area": 64.0,
                    "operator_id": op,
                    "split": split,
                    "err_total": err_total,
                    "err_free": err_total * np.sqrt(null_share),
                    "free_content": 1.0 + rng.normal(0, 0.05),
                    "ann_coverage": 1.0,
                    "s_free": signal + rng.normal(0, 0.6),
                    "s_disagree": (rng.normal(0, 1.0) if score_beats_baseline
                                   else signal + rng.normal(0, 0.6)),
                    "s_residual": rng.normal(0, 0.1),
                })
    return pd.DataFrame(rows)


# -- the share, and why it is taken at image level ---------------------------------


def test_the_share_recovers_a_planted_value():
    df = _bank("dps", null_share=0.84, score_beats_baseline=True)
    table = null_error_share(df)
    assert table["share"].median() == pytest.approx(0.84, abs=0.01)


def test_the_share_is_area_weighted_so_unequal_regions_do_not_vote_equally():
    """A large region's error has to count for more than a small one's, or the ratio is not
    the image's MSE ratio."""
    df = _bank("dps", null_share=0.5, score_beats_baseline=True, n_patients=4, n_regions=2)
    df.loc[df.index % 2 == 0, "region_area"] = 1.0
    df.loc[df.index % 2 == 0, "err_free"] = df.loc[df.index % 2 == 0, "err_total"]
    weighted = null_error_share(df)["share"].median()
    unweighted = null_error_share(df.assign(region_area=1.0))["share"].median()
    assert weighted != pytest.approx(unweighted), "area weighting had no effect"


def test_a_per_region_ratio_above_one_is_counted_and_not_an_error():
    """``P_N`` is non-local, so projecting then restricting is not restricting then
    projecting: energy crosses region boundaries and the per-region ratio can exceed 1. On
    the M4Raw bank it does so in 97 of 384 regions. It must be reported, not refused."""
    df = _bank("dps", null_share=0.95, score_beats_baseline=True, n_patients=4)
    df.loc[df.index[:10], "err_free"] = df.loc[df.index[:10], "err_total"] * 1.02
    table = null_error_share(df)
    assert table["n_over"].sum() == 10
    assert np.isfinite(table["share"]).all()


def test_a_bank_without_the_error_columns_is_refused():
    df = _bank("dps", 0.9, True).drop(columns=["err_total"])
    with pytest.raises(KeyError, match="err_total"):
        null_error_share(df)


# -- the premise gate --------------------------------------------------------------


def _write(tmp_path: Path, *frames: pd.DataFrame) -> list[str]:
    paths = []
    for i, f in enumerate(frames):
        p = tmp_path / f"bank{i}.parquet"
        f.to_parquet(p, index=False)
        paths.append(str(p))
    return paths


def test_two_solvers_with_the_same_share_are_flagged_as_having_no_premise(tmp_path, capsys):
    """If the solvers do not differ in how much error they leave in the null space, D68's
    rationale is simply absent and the script must say so rather than proceed quietly."""
    a = _bank("ddnm", null_share=0.95, score_beats_baseline=False, seed=1)
    b = _bank("dps", null_share=0.95 - MIN_SHARE_GAP / 2, score_beats_baseline=True, seed=2)
    code = solver_ablation.main(
        [*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP]
    )
    assert code == 0
    assert "PREMISE ABSENT" in capsys.readouterr().err


def test_a_real_share_gap_is_reported_as_holding(tmp_path, capsys):
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=3)
    b = _bank("dps", null_share=0.84, score_beats_baseline=True, seed=4)
    code = solver_ablation.main(
        [*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "premise holds" in out
    assert "dps" in out


def test_one_solver_is_refused_because_the_ablation_is_a_contrast(tmp_path, capsys):
    only = _bank("ddnm", 0.99, False)
    code = solver_ablation.main(
        [*_write(tmp_path, only), "--calibration-operator", CALIB_OP]
    )
    assert code == 1
    assert "needs two" in capsys.readouterr().err


def test_a_bank_without_solver_id_is_refused(tmp_path, capsys):
    df = _bank("ddnm", 0.99, False).drop(columns=["solver_id"])
    code = solver_ablation.main(
        [*_write(tmp_path, df), "--calibration-operator", CALIB_OP]
    )
    assert code == 1
    assert "solver_id" in capsys.readouterr().err


# -- the verdict, planted both ways ------------------------------------------------


def test_the_verdict_is_positive_when_the_low_share_solver_wins(tmp_path, capsys):
    """Planted: on the low-share arm the signal lives in ``s_free`` and the baseline is
    noise, so the score should beat it."""
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=5)
    b = _bank("dps", null_share=0.82, score_beats_baseline=True, seed=6)
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert "D68 ANSWERED" in out
    assert "NEGATIVE" not in out


def test_the_verdict_is_negative_when_the_low_share_solver_ties_its_baseline(tmp_path, capsys):
    """The outcome this project should expect, and the one the verdict must be able to
    reach: the decomposition does not earn its keep even where the mechanism allows it."""
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=7)
    b = _bank("dps", null_share=0.82, score_beats_baseline=False, seed=8)
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert "D68 ANSWERED, NEGATIVE" in out or "D68 PARTIAL" in out


def test_the_verdict_carries_d67s_caveat_either_way(tmp_path, capsys):
    """An excess against a magnitude label can be the identity rather than a detector, so
    the caveat travels with the number."""
    a = _bank("ddnm", 0.99, False, seed=9)
    b = _bank("dps", 0.82, True, seed=10)
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    assert "D67" in capsys.readouterr().out


def test_thin_operators_are_noted_rather_than_scored(tmp_path, capsys):
    a = _bank("ddnm", 0.99, False, seed=11)
    b = _bank("dps", 0.82, True, seed=12)
    # make one operator almost entirely negative on the dps arm
    mask = (b["operator_id"] == "op.high") & (b["split"] == "test")
    b.loc[mask, "err_free"] = 0.01
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert "floor (D32)" in out or "op.high" in out


def test_the_table_is_written_with_one_row_per_solver_operator(tmp_path):
    a = _bank("ddnm", 0.99, False, seed=13)
    b = _bank("dps", 0.82, True, seed=14)
    out_csv = tmp_path / "ablation.csv"
    solver_ablation.main([
        *_write(tmp_path, a, b), "--calibration-operator", CALIB_OP, "--out", str(out_csv),
    ])
    table = pd.read_csv(out_csv)
    assert table["solver_id"].nunique() == 2
    assert table["operator_id"].nunique() >= 2, "never pooled across operators"
