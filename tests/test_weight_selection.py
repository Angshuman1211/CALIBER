"""Gates for D28's weight selection rule (T1.1).

The rule is pre-registered in `scoring/weights.py` and these tests pin it, so that a later
reading of a real design split cannot quietly become a different rule. Four behaviours
matter:

* a flat objective is resolved by the **tie-break**, not by its argmin;
* a genuine minimum is taken as-is, so the tie-break is not doing all the work;
* an endpoint is *reported* rather than avoided, because dropping a component is a finding;
* selecting anywhere but the design split is refused, since both alternatives fail silently.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.scorebank import read_bank
from caliber.scoring.weights import (
    DEFAULT_GRID,
    lipschitz_ceiling,
    select_weights,
    weight_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import select_weights as select_weights_script  # noqa: E402

COMMITTED_BANK = REPO_ROOT / "banks" / "w02_solver_smoke"

CAL = "mri.random1d.R4.c0.08.s0"
DEP = "mri.random1d.R8.c0.08.s0"


def _frame(
    disagree_noise: float = 0.60,
    free_noise: float = 0.10,
    n_clusters: int = 12,
    per_cluster: int = 64,
    split: str = "design",
    seed: int = 0,
) -> pd.DataFrame:
    """A design split where a shared latent signal drives the label.

    ``free_noise`` and ``disagree_noise`` set how faithfully each component reports that
    signal, which is what the sweep is supposed to discover.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_clusters):
        for op in (CAL, DEP):
            signal = rng.gamma(2.0, 0.4, per_cluster)
            parts.append(
                pd.DataFrame(
                    {
                        "sample_id": f"p{i}",
                        "operator_id": op,
                        "calib_group": op.split(".c0")[0],
                        "split": split,
                        "region_id": np.arange(per_cluster),
                        "s_free": signal + free_noise * rng.normal(size=per_cluster),
                        "s_disagree": 0.6 * signal
                        + disagree_noise * rng.normal(size=per_cluster),
                        "s_residual": 1e-8 * rng.normal(size=per_cluster),
                        "label": signal > np.quantile(signal, 0.72),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


# -- the rule ------------------------------------------------------------------


def test_a_clear_minimum_is_taken_without_a_tie_break():
    """When one component is plainly better, the sweep says so on its own."""
    selection = select_weights(_frame(disagree_noise=1.5, seed=1), CAL)
    assert selection.w_free >= 0.8
    assert selection.flag_area == selection.curve["flag_area"].min()


def test_a_flat_objective_is_resolved_toward_the_tighter_bound():
    """D28 step 4, which is the part that needed a ruling rather than a measurement.

    With both components reporting the signal equally well the curve is flat, several
    points sit within one SE of the minimum, and the raw argmin would be noise. The rule
    takes the largest `w_free` among them, because `L` falls as `w_free` rises (D41).
    """
    selection = select_weights(_frame(disagree_noise=0.11, free_noise=0.10, seed=3), CAL)
    assert len(selection.tied_with) > 1, "this control needs a flat region to resolve"
    assert selection.w_free == max(selection.tied_with)
    # The chosen point is not necessarily the argmin, and that is the intended behaviour.
    assert selection.flag_area <= selection.curve["flag_area"].min() + selection.se + 1e-12
    assert selection.lipschitz <= lipschitz_ceiling(min(selection.tied_with)) + 1e-12


def test_dropping_a_component_is_reported_not_avoided():
    """An endpoint is a substantive finding about the score's composition."""
    selection = select_weights(_frame(disagree_noise=3.0, seed=4), CAL)
    assert selection.w_free == 1.0
    assert selection.at_endpoint
    assert "ENDPOINT" in selection.note()
    assert selection.weights.disagree == 0.0


def test_the_residual_component_never_receives_weight():
    """D42, enforced by the shape of the sweep rather than by remembering to pass zero."""
    selection = select_weights(_frame(seed=5), CAL)
    assert selection.weights.residual == 0.0
    assert set(selection.curve["w_free"]) == set(DEFAULT_GRID)
    assert np.allclose(
        selection.curve["w_free"] + selection.curve["w_disagree"], 1.0
    ), "D25: the weights must stay on the simplex at every point"


# -- the premise the comparison rests on ---------------------------------------


def test_the_miss_rate_is_pinned_across_the_sweep_so_area_is_the_only_axis():
    """D39's argument, as a guard rather than a claim.

    If the miss rate ever varied materially along the sweep, the calibrator would not be
    doing its job and comparing flagged areas would no longer be like-for-like.
    """
    curve = weight_curve(_frame(seed=6), CAL, alpha=0.10)
    assert curve["miss_rate"].max() - curve["miss_rate"].min() < 0.02
    assert abs(curve["miss_rate"].mean() - 0.10) < 0.05


def test_the_lipschitz_ceiling_is_the_convex_combination_and_falls_with_w_free():
    assert lipschitz_ceiling(0.0) == pytest.approx(0.5716)
    assert lipschitz_ceiling(1.0) == pytest.approx(0.4916)
    values = [lipschitz_ceiling(w) for w in DEFAULT_GRID]
    assert values == sorted(values, reverse=True)


# -- refusals ------------------------------------------------------------------


def test_selecting_without_a_design_split_is_refused():
    """Both alternatives bias something silently (D-051), so this must not fall back."""
    calib_only = _frame(split="calib", seed=7)
    with pytest.raises(ValueError, match="design"):
        select_weights(calib_only, CAL)


def test_a_bank_with_no_split_column_at_all_is_refused():
    frame = _frame(seed=8).drop(columns=["split"])
    with pytest.raises(ValueError):
        select_weights(frame, CAL)


def test_a_design_split_missing_the_calibration_operator_is_refused():
    """The scales must be frozen on the calibration operator (D27)."""
    frame = _frame(seed=9)
    frame = frame[frame["operator_id"] != CAL]
    with pytest.raises(KeyError):
        select_weights(frame, CAL)


# -- the entry point (P-19: executing the script is not the same as the library) ----


def test_the_script_refuses_a_bank_with_no_design_split_and_says_what_to_run(capsys):
    """The message has to name the real knob. `design_fraction` is a *dataset* kwarg, not
    a `run.` one, and since D-048 a mistyped override is refused outright — so a message
    pointing at the wrong key would send someone into a loud failure."""
    assert select_weights_script.main([str(COMMITTED_BANK)]) == 1
    err = capsys.readouterr().err
    assert "design" in err
    assert "data.dataset.design_fraction" in err


def test_the_script_selects_end_to_end_on_a_bank_that_has_a_design_side(tmp_path, capsys):
    """Built by relabelling a real bank's patients, since no config writes one yet."""
    bank = tmp_path / "with-design"
    shutil.copytree(COMMITTED_BANK, bank)
    df = read_bank(COMMITTED_BANK)
    chosen = sorted(df["sample_id"].unique())[0]
    df["split"] = np.where(df["sample_id"] == chosen, "design", df["split"])
    df.to_parquet(bank / "scores.parquet", index=False)

    assert select_weights_script.main([str(bank), "--points", "5"]) == 0
    out = capsys.readouterr().out
    assert "SELECTED" in out
    assert "design split" in out
