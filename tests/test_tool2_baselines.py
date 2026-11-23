"""T1-C: the three baselines Tool 2 is scored against, and the direction of max-quantile.

The load-bearing test here is
:func:`test_the_conservative_reading_of_max_quantile_is_the_valid_one`. The frozen name
``max_quantile`` does not by itself say whether to take the largest or the smallest
per-group threshold, and with flags at ``score >= tau`` those are opposite choices. Picking
the one that makes Tool 2 look better would be this project's recurring failure mode exactly
(``prereg.AMENDMENTS``, eleven instances). So the reading is settled by the property the
frozen name appeals to -- that the arm is trivially *safe* -- and the test runs both
candidate readings on a bank where they disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.calibration import (
    apply_flags,
    calibrate_group_conditional_oracle,
    calibrate_max_quantile,
    calibrate_naive,
    calibrate_pooled,
)
from caliber.calibration.conformal import split_threshold
from caliber.evaluation import prereg

ALPHA = 0.1
NOMINAL = 1.0 - ALPHA
#: Calibration operators, and the deployment operator sitting *between* them -- the
#: interior fold the harness is about. Values are the positive-score centres.
CALIB_CENTRES = {"g_a": 1.0, "g_b": 2.0, "g_c": 3.0, "g_d": 4.0}
DEPLOY_CENTRE = 2.5
SPREAD = 0.5


def _bank(n_pos: int = 400, n_neg: int = 400, seed: int = 11) -> pd.DataFrame:
    """A score bank with four calibration operators and one held-out deployment operator."""
    rng = np.random.default_rng(seed)
    frames = []
    centres = {**CALIB_CENTRES, "g_deploy": DEPLOY_CENTRE}
    for group, centre in centres.items():
        pos = rng.normal(centre, SPREAD, n_pos)
        neg = rng.normal(centre - 2.0, SPREAD, n_neg)
        frames.append(pd.DataFrame({
            "score": np.concatenate([pos, neg]),
            "label": np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)]),
            "calib_group": group,
            "split": "calib",
        }))
    return pd.concat(frames, ignore_index=True)


def _pool(bank: pd.DataFrame) -> pd.DataFrame:
    """The calibration side: every operator except the held-out one."""
    return bank[bank["calib_group"] != "g_deploy"].copy()


def _deploy(bank: pd.DataFrame) -> pd.DataFrame:
    return bank[bank["calib_group"] == "g_deploy"].copy()


def _coverage(deploy: pd.DataFrame, tau: float) -> float:
    pos = deploy[deploy["label"]]
    return float((pos["score"].to_numpy() >= tau).mean())


def _area(deploy: pd.DataFrame, report) -> float:
    return float(apply_flags(deploy, report, out_col="flag")["flag"].mean())


def _per_group_taus(pool: pd.DataFrame) -> dict[str, float]:
    out = {}
    for group, rows in pool.groupby(pool["calib_group"].astype(str)):
        out[str(group)] = split_threshold(
            rows["score"].to_numpy(), rows["label"].to_numpy(), ALPHA).threshold
    return out


# -- the direction, settled by property ---------------------------------------


def test_the_conservative_reading_of_max_quantile_is_the_valid_one():
    """Both readings of the frozen name, on a bank where they disagree."""
    bank = _bank()
    pool, deploy = _pool(bank), _deploy(bank)
    taus = _per_group_taus(pool)

    smallest_tau, largest_tau = min(taus.values()), max(taus.values())
    assert smallest_tau < largest_tau, "the fixture must make the two readings differ"

    cov_small = _coverage(deploy, smallest_tau)
    cov_large = _coverage(deploy, largest_tau)

    assert cov_small >= NOMINAL, (
        f"the smallest per-group tau must be valid at the unseen operator: {cov_small:.3f}")
    assert cov_large < NOMINAL, (
        "the largest per-group tau is NOT trivially safe, which is what rules that reading "
        f"out: coverage {cov_large:.3f} against nominal {NOMINAL}")

    # And the implementation takes the reading that survives.
    got = calibrate_max_quantile(pool, alpha=ALPHA)
    assert got.thresholds["__all__"] == pytest.approx(smallest_tau)


def test_max_quantile_is_valid_wherever_any_single_group_arm_is():
    """Its defining property: safe in every calibration group at once."""
    bank = _bank()
    pool, deploy = _pool(bank), _deploy(bank)
    mq = calibrate_max_quantile(pool, alpha=ALPHA)
    tau_mq = mq.thresholds["__all__"]

    for group, tau in _per_group_taus(pool).items():
        assert tau_mq <= tau, f"must be at or below {group}'s own threshold"
        assert _coverage(deploy, tau_mq) >= _coverage(deploy, tau), (
            f"coverage must not fall below the {group} arm's")


def test_max_quantile_flags_at_least_as_much_as_any_borrowed_group():
    """It buys its validity with area -- that is the cost Tool 2 has to undercut."""
    bank = _bank()
    pool, deploy = _pool(bank), _deploy(bank)
    mq_area = _area(deploy, calibrate_max_quantile(pool, alpha=ALPHA))
    for group in CALIB_CENTRES:
        borrowed = calibrate_naive(pool, calibration_group=group, alpha=ALPHA)
        assert mq_area >= _area(deploy, borrowed) - 1e-12, (
            f"the trivially-safe rule cannot flag less area than borrowing {group}")


# -- the thin-group handling that keeps the baseline non-degenerate -----------


def test_a_thin_group_does_not_collapse_max_quantile_to_flagging_everything():
    """The degenerate case: one unusable group must not swallow the whole baseline."""
    bank = _bank()
    pool = _pool(bank)
    # Leave this group two positives: floor(0.1 * 3) = 0, so it cannot certify.
    thin = pool[(pool["calib_group"] == "g_d") & pool["label"]].index[2:]
    pool = pool.drop(index=thin)

    got = calibrate_max_quantile(pool, alpha=ALPHA)
    assert "g_d" in got.uncertifiable_groups, "the thin group must be visible as such"
    assert got.reportable is False, "and must make the report non-reportable"
    assert np.isfinite(got.thresholds["__all__"]), (
        "but it must not drag the threshold to -inf, which would flag every region and "
        "make the baseline compare against nothing")
    assert _area(_deploy(bank), got) < 1.0


def test_max_quantile_says_so_when_no_group_can_certify():
    bank = _bank()
    pool = _pool(bank)
    keep = pool[~pool["label"]].index.tolist()
    keep += pool[pool["label"]].groupby(pool["calib_group"]).head(2).index.tolist()
    got = calibrate_max_quantile(pool.loc[keep], alpha=ALPHA)
    assert got.reportable is False
    assert got.thresholds["__all__"] == -np.inf, (
        "with nothing certifiable the conservative threshold is the honest answer")


def test_max_quantile_refuses_an_empty_calibration_side():
    with pytest.raises(KeyError):
        calibrate_max_quantile(_bank().iloc[0:0], alpha=ALPHA)


# -- pooled, and why it is legitimate here ------------------------------------


def test_pooled_uses_every_calibration_operator_and_none_of_the_deployment_one():
    bank = _bank()
    pool, deploy = _pool(bank), _deploy(bank)
    pooled = calibrate_pooled(pool, alpha=ALPHA)
    tau_pooled = pooled.thresholds["__all__"]

    expect = split_threshold(pool["score"].to_numpy(), pool["label"].to_numpy(),
                             ALPHA).threshold
    assert tau_pooled == pytest.approx(expect)
    assert pooled.method == "pooled"

    # The objection calibrate_naive's docstring raises is that pooling drags the threshold
    # down by including the deployment regime. Here it cannot: that regime is held out.
    contaminated = calibrate_pooled(bank, alpha=ALPHA).thresholds["__all__"]
    assert contaminated != pytest.approx(tau_pooled), (
        "pooling WITH the deployment operator must give a different answer, which is why "
        "this arm is only ever called on the held-out pool")
    # Pooling is a compromise, not a guarantee: its threshold sits strictly above the
    # trivially-safe one, so whether it is valid is a property of the bank. On this
    # fixture it happens to be, which is exactly why it must not be asserted in general.
    tau_safe = calibrate_max_quantile(pool, alpha=ALPHA).thresholds["__all__"]
    assert tau_pooled > tau_safe
    assert _coverage(deploy, tau_pooled) <= _coverage(deploy, tau_safe)


def test_pooled_sits_between_the_extreme_single_group_arms():
    """A mixture threshold is not conservative; it is a weighted compromise."""
    bank = _bank()
    pool = _pool(bank)
    taus = _per_group_taus(pool)
    tau_pooled = calibrate_pooled(pool, alpha=ALPHA).thresholds["__all__"]
    assert min(taus.values()) <= tau_pooled <= max(taus.values())


# -- the oracle ---------------------------------------------------------------


def test_the_oracle_matches_a_naive_fit_on_the_held_out_operators_own_rows():
    bank = _bank()
    deploy = _deploy(bank)
    oracle = calibrate_group_conditional_oracle(deploy, "g_deploy", alpha=ALPHA)
    expect = calibrate_naive(deploy, calibration_group="g_deploy", alpha=ALPHA)
    assert oracle.thresholds == expect.thresholds


def test_the_oracle_is_valid_and_flags_less_area_than_the_trivially_safe_rule():
    """The headroom the deployable arms are competing for."""
    bank = _bank()
    pool, deploy = _pool(bank), _deploy(bank)
    oracle = calibrate_group_conditional_oracle(deploy, "g_deploy", alpha=ALPHA)
    assert _coverage(deploy, oracle.thresholds["__all__"]) >= NOMINAL - 0.02
    assert _area(deploy, oracle) < _area(deploy, calibrate_max_quantile(pool, alpha=ALPHA))


# -- the freeze ---------------------------------------------------------------


def test_the_three_arms_are_exactly_the_frozen_baseline_set():
    from caliber.calibration import tool2_baselines as baselines

    names = {n.removeprefix("calibrate_") for n in baselines.__all__}
    assert names == set(prereg.TOOL2_BASELINES), (
        f"implemented {sorted(names)} against frozen {sorted(prereg.TOOL2_BASELINES)}")
    assert "nearest" not in prereg.TOOL2_BASELINES, (
        "nearest was retired in D-070 Sec. 3 for winning on area while invalid")


def test_the_comparison_rule_is_frozen_and_not_a_flag():
    assert prereg.TOOL2_REQUIRE_BOTH_VALID is True
    assert prereg.TOOL2_REPORT_ALL_WEIGHTINGS is True
    assert prereg.TOOL2_SELECTION_SCOPE == "calibration_operators_only"


def test_an_invalid_arm_posts_the_smallest_area_which_is_why_both_must_be_valid():
    """``prereg.TOOL2_REQUIRE_BOTH_VALID`` exists because of this, and D-070 Sec. 3
    retired ``nearest`` for it: an arm that has stopped catching anything flags almost
    nothing, and its area reads as efficiency unless validity is checked first."""
    bank = _bank()
    pool, deploy = _pool(bank), _deploy(bank)

    rows = []
    for group in CALIB_CENTRES:
        report = calibrate_naive(pool, calibration_group=group, alpha=ALPHA)
        tau = report.thresholds["__all__"]
        rows.append((group, _coverage(deploy, tau) >= NOMINAL, _area(deploy, report)))

    valid_areas = [a for _, ok, a in rows if ok]
    invalid_areas = [a for _, ok, a in rows if not ok]
    assert valid_areas and invalid_areas, "the fixture must contain both kinds of arm"
    assert min(invalid_areas) < min(valid_areas), (
        "the broken arms win on raw area, so ranking on area alone would select them: "
        f"invalid {sorted(invalid_areas)} against valid {sorted(valid_areas)}")


# -- the T1-C verdict: the fold pairing TOOL2_REQUIRE_BOTH_VALID asks for --------
#
# The arm-average shortcut is what let `nearest` post the smallest area in D-070 Sec. 3
# while under-covering in three folds of seven: averaging each arm over the folds it is
# valid on compares two different fold sets. These tests pin the pairing instead.

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tool2_holdout as T2H  # noqa: E402


class _Args:
    def __init__(self, alpha: float = ALPHA, out: str | None = None, t1c: bool = True):
        self.alpha, self.out, self.t1c = alpha, out, t1c
        self.exclude_nearest = 0


def _table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"interior": True, **r} for r in rows])


def test_the_verdict_compares_area_only_on_folds_where_both_arms_are_valid(capsys):
    """Tool 2 wins on its own two folds and loses on the two it shares. The shared
    folds are the comparison, so the saving must come out negative."""
    rows = []
    for fold, (t2_valid, t2_area, safe_area) in {
        "f1": (True, 0.30, 0.80),   # tool2 valid, and cheap -- but safe is invalid here
        "f2": (True, 0.70, 0.60),   # both valid, tool2 WORSE
        "f3": (True, 0.75, 0.60),   # both valid, tool2 WORSE
        "f4": (True, 0.20, 0.90),   # tool2 valid, safe invalid again
    }.items():
        safe_valid = fold in ("f2", "f3")
        rows.append({"held_out": fold, "arm": "weighted_h1", "valid": t2_valid,
                     "flag_area": t2_area})
        rows.append({"held_out": fold, "arm": "max_quantile", "valid": safe_valid,
                     "flag_area": safe_area})
    T2H.t1c_verdict(_table(rows), _Args(), [])
    out = capsys.readouterr().out

    assert "folds_both_valid" in out
    # Averaging over all four folds would have shown tool2 saving ~34%; the paired
    # restriction to f2 and f3 must show it LOSING.
    assert "NO LESS area" in out, out
    naive = (1 - (0.30 + 0.70 + 0.75 + 0.20) / (0.80 + 0.60 + 0.60 + 0.90)) * 100
    assert naive > 20, "the fixture must make the unpaired shortcut look like a win"


def test_the_verdict_withholds_any_area_claim_when_tool2_is_not_valid_everywhere(capsys):
    rows = []
    for fold, t2_valid in (("f1", True), ("f2", False), ("f3", True)):
        rows.append({"held_out": fold, "arm": "weighted_h1", "valid": t2_valid,
                     "flag_area": 0.2})
        rows.append({"held_out": fold, "arm": "max_quantile", "valid": True,
                     "flag_area": 0.8})
    T2H.t1c_verdict(_table(rows), _Args(), [])
    out = capsys.readouterr().out
    assert "NOT valid at every unseen operator" in out
    assert "no efficiency claim" in out or "no efficiency claim to make" in out
    assert "less area than the trivially-safe" not in out, (
        "an area headline beside a validity failure is the nearest-arm error again")


def test_the_verdict_reports_the_whole_bandwidth_span_not_the_best_point(capsys):
    rows = []
    for fold in ("f1", "f2"):
        rows.append({"held_out": fold, "arm": "max_quantile", "valid": True,
                     "flag_area": 0.80})
        for mult, area in ((0.25, 0.75), (1.0, 0.40), (4.0, 0.78)):
            rows.append({"held_out": fold, "arm": f"weighted_h{mult:g}", "valid": True,
                         "flag_area": area})
    T2H.t1c_verdict(_table(rows), _Args(), [])
    out = capsys.readouterr().out
    assert "50.0% less area" in out, out
    assert "spans" in out and "Report the span, not the best point." in out


def test_the_verdict_flags_a_trivially_safe_rule_that_is_not_safe(capsys):
    """max_quantile failing validity means the bank or the label is wrong, not the method."""
    rows = [{"held_out": "f1", "arm": "max_quantile", "valid": False, "flag_area": 0.8},
            {"held_out": "f1", "arm": "weighted_h1", "valid": True, "flag_area": 0.4}]
    T2H.t1c_verdict(_table(rows), _Args(), [])
    out = capsys.readouterr().out
    assert "valid in only 0 of 1 folds" in out
    assert "points at the bank or the label" in out


def test_the_verdict_is_undecided_with_no_interior_fold(capsys):
    rows = pd.DataFrame([{"held_out": "f1", "arm": "weighted_h1", "valid": True,
                          "flag_area": 0.4, "interior": False}])
    T2H.t1c_verdict(rows, _Args(), [])
    assert "UNDECIDED" in capsys.readouterr().out


def test_the_bandwidth_grid_is_frozen_and_every_point_is_reported(capsys):
    assert prereg.TOOL2_BANDWIDTH_GRID == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert 1.0 in prereg.TOOL2_BANDWIDTH_GRID, (
        "D33's default must be on the grid, or the grid moves the method's own choice")
    rows = [{"held_out": "f1", "arm": "max_quantile", "valid": True, "flag_area": 0.8}]
    for mult in prereg.TOOL2_BANDWIDTH_GRID:
        rows.append({"held_out": "f1", "arm": f"weighted_h{mult:g}", "valid": True,
                     "flag_area": 0.5})
    T2H.t1c_verdict(_table(rows), _Args(), [])
    out = capsys.readouterr().out
    for mult in prereg.TOOL2_BANDWIDTH_GRID:
        assert f"weighted_h{mult:g}" in out, f"bandwidth {mult} must appear in the table"


def test_amendment_fourteen_froze_the_grid_before_t1c_was_run():
    entry = [t for _, t in prereg.AMENDMENTS if "TOOL2_BANDWIDTH_GRID" in t]
    assert len(entry) == 1, "amendment 14 must appear exactly once"
    assert "before T1-C has been run on either bank" in entry[0], (
        "a grid frozen after the numbers exist is not a freeze")
    assert "NARROWS" in entry[0]
    dates = [d for d, _ in prereg.AMENDMENTS]
    assert dates == sorted(dates)
    assert sum("LOWERS THE PRIMARY ENDPOINT" in t for _, t in prereg.AMENDMENTS) == 1
