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

import json
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
    identity_consistent: bool = False,
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
                    # overwritten below when identity_consistent
                    "free_content": 1.0 + rng.normal(0, 0.05),
                    "ann_coverage": 1.0,
                    "s_free": signal + rng.normal(0, 0.6),
                    "s_disagree": (rng.normal(0, 1.0) if score_beats_baseline
                                   else signal + rng.normal(0, 0.6)),
                    "s_residual": rng.normal(0, 0.1),
                })
    df = pd.DataFrame(rows)
    if identity_consistent:
        # D74's label inverts err^2 = s^2 + c^2 - 2sc cos(theta). Unless the three columns
        # actually satisfy it, cos(theta) falls outside [-1, 1], comes back NaN, and every
        # region is labelled negative -- so an alignment-label test on an inconsistent
        # fixture would pass by measuring nothing.
        rng2 = np.random.default_rng(seed + 500)
        s = df["s_free"].to_numpy(dtype=float)
        s = np.abs(s) + 0.5
        c = np.abs(df["free_content"].to_numpy(dtype=float)) + 0.5
        # hallucinated regions are anti-aligned, clean ones aligned
        hall = df["err_free"].to_numpy() > np.median(df["err_free"].to_numpy())
        cos = np.where(hall, rng2.uniform(-0.9, -0.1, len(df)),
                       rng2.uniform(0.1, 0.9, len(df)))
        df["s_free"] = s
        df["free_content"] = c
        df["err_free"] = np.sqrt(np.maximum(s**2 + c**2 - 2 * s * c * cos, 1e-12))
        df["err_total"] = df["err_free"] / np.sqrt(null_share)
    return df

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


def test_a_positive_verdict_is_available_on_the_direction_label(tmp_path, capsys):
    """The positive path, which only the direction label can reach (D81).

    On a magnitude label a positive is withheld, because D38's identity makes that label
    approximately a function of ``s_free`` and the score contains ``s_free``. So the
    verdict that *can* be positive has to be exercised under ``--label alignment``, on a
    bank whose three null-space columns actually satisfy the identity.
    """
    a = _bank("ddnm", 0.99, score_beats_baseline=False, seed=5, identity_consistent=True)
    b = _bank("dps", 0.82, score_beats_baseline=True, seed=6, identity_consistent=True)
    code = solver_ablation.main([
        *_write(tmp_path, a, b), "--calibration-operator", CALIB_OP,
        "--label", "alignment",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "NOT ANSWERED on this label" not in out, (
        "the direction label is the one a positive may be stated on"
    )
    assert "D68" in out


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


# -- the premise comes from Stage 2, not from the bank -----------------------------


def _stage2(tmp_path: Path, name: str, solver_id: str, shares: list[float],
            defect: float = 1e-6) -> str:
    """A `stage2_decomposition.py` artefact, in its real shape: key `operators`."""
    payload = {
        "config": "configs/experiments/fake.yaml",
        "solver": {"solver_id": solver_id},
        "operators": [
            {"operator_id": f"op{i}", "null_error_share": s,
             "orthogonality_defect": defect}
            for i, s in enumerate(shares)
        ],
    }
    p = tmp_path / f"stage2-{name}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_the_stage2_share_is_read_keyed_on_solver_id():
    """Stage 2's definition is the authoritative one: a whole-image projector split whose
    denominator is the sum of the two components, so it survives an only-approximately
    orthogonal projector (the CT case, D-012). This script must read it rather than
    recompute a per-region approximation."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        got = solver_ablation.read_stage2([
            _stage2(tmp, "ddnm", "diffusion.ddnm.x", [0.99, 0.991, 0.989]),
            _stage2(tmp, "dps", "diffusion.dps.x", [0.82, 0.85, 0.84]),
        ])
    assert set(got) == {"diffusion.ddnm.x", "diffusion.dps.x"}
    assert got["diffusion.ddnm.x"]["share"] == pytest.approx(0.99)
    assert got["diffusion.dps.x"]["share"] == pytest.approx(0.84)


def test_the_share_range_is_kept_so_a_solver_varying_by_operator_is_visible():
    """A median alone would hide a solver whose share swings across the grid, and that
    swing is the thing that would undermine the premise."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        got = solver_ablation.read_stage2(
            [_stage2(Path(d), "dps", "dps", [0.40, 0.84, 0.99])]
        )
    assert got["dps"]["lo"] == pytest.approx(0.40)
    assert got["dps"]["hi"] == pytest.approx(0.99)
    assert got["dps"]["share"] == pytest.approx(0.84)


def test_a_stage2_artefact_with_no_finite_share_is_refused():
    """`null_error_share` only exists when x_true was supplied, so a run without it must
    be refused rather than read as zero."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        p.write_text(json.dumps({
            "solver": {"solver_id": "x"},
            "operators": [{"operator_id": "op0", "null_error_share": None}],
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="no finite null_error_share"):
            solver_ablation.read_stage2([str(p)])


def test_an_artefact_without_per_operator_rows_is_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        p.write_text(json.dumps({"solver": {"solver_id": "x"}}), encoding="utf-8")
        with pytest.raises(ValueError, match="no per-operator rows"):
            solver_ablation.read_stage2([str(p)])


def test_the_bank_approximation_is_labelled_as_such_when_stage2_is_absent(tmp_path, capsys):
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=20)
    b = _bank("dps", null_share=0.82, score_beats_baseline=True, seed=21)
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert "APPROXIMATED FROM THE BANK" in out
    assert "--stage2" in out


def test_stage2_shares_are_preferred_over_the_bank_when_both_are_available(tmp_path, capsys):
    """The bank fixture plants 0.99/0.82; Stage 2 says 0.60/0.55. The printed premise must
    be Stage 2's, and the gap must be judged on Stage 2's numbers."""
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=22)
    b = _bank("dps", null_share=0.82, score_beats_baseline=True, seed=23)
    s2 = [_stage2(tmp_path, "ddnm", "ddnm", [0.60]), _stage2(tmp_path, "dps", "dps", [0.55])]
    solver_ablation.main([
        *_write(tmp_path, a, b), "--calibration-operator", CALIB_OP, "--stage2", *s2,
    ])
    captured = capsys.readouterr()
    assert "from Stage 2" in captured.out
    assert "0.6000" in captured.out and "0.5500" in captured.out
    assert "PREMISE ABSENT" in captured.err, (
        "a 0.05 gap is not a premise, whatever the bank says"
    )


# -- the two defects Track 2 found (D81) -------------------------------------------


def test_each_bank_gets_its_label_from_its_own_design_split(tmp_path, capsys):
    """**Defect 1.** The first version valued the label once on the *pooled* design rows
    of every bank, so each arm was scored at a positive rate that was not its own --
    measured by Track 2 as a pooled 0.856 against 0.906 (DDNM) and 0.726 (DPS). A label
    gate is a magnitude, and D-020 is this project's own finding that magnitudes do not
    transfer across substrates; two solvers are two substrates here.
    """
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=30)
    b = _bank("dps", null_share=0.82, score_beats_baseline=True, seed=31)
    # Shift one bank's error scale so a pooled quantile would differ from either bank's.
    b["err_free"] = b["err_free"] * 3.0
    b["err_total"] = b["err_total"] * 3.0
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert out.count("[ablation] label (") == 2, "one label line per bank"
    assert "the two labels differ" in out, (
        "banks on different error scales must get different thresholds"
    )


def test_a_positive_verdict_is_withheld_on_a_magnitude_label(tmp_path, capsys):
    """**Defect 2.** The script printed 'D68 ANSWERED, POSITIVE' off the magnitude label,
    which D38's identity makes approximately a function of `s_free` -- and the score
    contains `s_free` while `s_disagree` does not. Track 2 had to overturn that by hand."""
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=32)
    b = _bank("dps", null_share=0.82, score_beats_baseline=True, seed=33)
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert "D68 ANSWERED, POSITIVE" not in out
    assert "NOT ANSWERED on this label" in out
    assert "--label alignment" in out, "it must say what would answer it"


def test_a_negative_verdict_still_stands_on_a_magnitude_label(tmp_path, capsys):
    """The asymmetry is deliberate: if the score cannot win even where the label favours
    it by construction, that is decisive, and withholding it would lose a real finding."""
    a = _bank("ddnm", null_share=0.99, score_beats_baseline=False, seed=34)
    b = _bank("dps", null_share=0.82, score_beats_baseline=False, seed=35)
    solver_ablation.main([*_write(tmp_path, a, b), "--calibration-operator", CALIB_OP])
    out = capsys.readouterr().out
    assert "D68 ANSWERED, NEGATIVE" in out or "D68 PARTIAL" in out
    assert "NOT ANSWERED on this label" not in out
