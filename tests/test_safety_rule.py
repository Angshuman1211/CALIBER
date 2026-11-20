"""T1-A's sign rule: does it report a real error rate, and does it stay quiet on noise?

The tests that matter here are the negative ones. The project's recurring failure mode is a
statistic that returns the hoped-for number -- D73's bootstrap held the threshold fixed and
reported a null |delta| of 0.128 against a 0.02 floor, and only a planted-null fixture
caught it. So the central test below is :func:`test_null_fixture_is_not_a_result`, and the
one after it checks the permutation test's type-I error empirically rather than trusting
that shuffling labels must be valid because it sounds valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation import prereg, safety


# -- fixtures ------------------------------------------------------------------


def _grid(k: int = 9) -> list[str]:
    return [f"op{i:02d}" for i in range(k)]


def _ordered_pairs(ops: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in ops for b in ops if a != b]


def _frame(ops, signed_of, coverage_of, half_width: float = 0.0) -> pd.DataFrame:
    """An ordered-pair frame. ``half_width`` 0 means an exact (zero-width) interval.

    The planted fixtures below want exact coverage, so they use a zero-width interval and
    the label reduces to the point-estimate form. Stating it as a width makes the choice
    visible rather than letting the fixture quietly exercise a different label from the
    one the real analysis uses.
    """
    rows = []
    for a, b in _ordered_pairs(ops):
        cov = coverage_of(a, b)
        rows.append(
            {
                "calibration": a,
                "deployment": b,
                "signed_distance": signed_of(a, b),
                "coverage": cov,
                "coverage_hi": cov + half_width,
            }
        )
    return pd.DataFrame(rows)


def _severity(ops):
    """A monotone ill-posedness index, standing in for view count."""
    return {op: i for i, op in enumerate(ops)}


def perfect_frame(k: int = 9) -> pd.DataFrame:
    """Signed distance determines coverage exactly: deploying harder under-covers.

    This is the world the rule is designed for, and the rule should be perfect in it.
    """
    ops = _grid(k)
    sev = _severity(ops)
    signed = lambda a, b: float(sev[b] - sev[a])
    cover = lambda a, b: 0.80 if sev[b] > sev[a] else 0.95
    return _frame(ops, signed, cover)


def null_frame(k: int = 9, seed: int = 0) -> pd.DataFrame:
    """Coverage drawn independently of the operators. There is nothing here to find."""
    rng = np.random.default_rng(seed)
    ops = _grid(k)
    sev = _severity(ops)
    draws = {p: float(rng.uniform(0.80, 0.98)) for p in _ordered_pairs(ops)}
    return _frame(ops, lambda a, b: float(sev[b] - sev[a]), lambda a, b: draws[(a, b)])


def reversed_frame(k: int = 9) -> pd.DataFrame:
    """Safety is anti-predicted by the sign: the rule is reliably wrong."""
    ops = _grid(k)
    sev = _severity(ops)
    return _frame(
        ops,
        lambda a, b: float(sev[b] - sev[a]),
        lambda a, b: 0.95 if sev[b] > sev[a] else 0.80,
    )


@pytest.fixture(autouse=True)
def _cheap_resampling(monkeypatch):
    """The real counts are 10 000 each; the fixtures do not need that to show a bug."""
    monkeypatch.setattr(prereg, "N_PERMUTATIONS", 300)
    monkeypatch.setattr(prereg, "N_BOOT", 300)


# -- the rule in the world it is for -------------------------------------------


def test_perfect_fixture_is_a_clean_positive():
    r = safety.safety_rule(perfect_frame())
    assert r.false_safe == 0
    assert r.false_safe_rate == 0.0
    assert r.youden_j == pytest.approx(1.0)
    assert r.green_light_fraction == pytest.approx(0.5)
    assert r.beats_trivial
    assert r.verdict == "positive"
    assert r.p_value < 0.05


def test_perfect_fixture_bootstrap_brackets_the_point_estimate():
    r = safety.safety_rule(perfect_frame())
    lo, hi = r.false_safe_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= r.false_safe_rate <= hi


# -- the tests that exist to stop a non-result being reported as one -----------


def test_null_fixture_is_not_a_result():
    """Coverage independent of distance must not clear the trivial bar or the null."""
    r = safety.safety_rule(null_frame())
    assert r.n_unsafe > 0, "fixture must contain some under-covering cells to be a test"
    assert not r.beats_trivial
    assert r.verdict != "positive"
    assert r.p_value > 0.05
    # The whole point of the trivial comparator: the rate alone looks respectable.
    assert r.false_safe_rate == pytest.approx(r.trivial_false_safe_rate, abs=0.25)


def test_permutation_null_has_calibrated_type_one_error():
    """Rejection rate on pure noise, over independent fixtures.

    A valid two-sided permutation test rejects at most 5% of the time. With 40 trials the
    99% Monte-Carlo upper bound on a true 0.05 is about 0.14; the discreteness of a
    9-label permutation can only make the test more conservative, so 0.20 is a generous
    ceiling and anything above it is a defect in the null, not noise.
    """
    rejects = 0
    trials = 40
    for seed in range(trials):
        r = safety.safety_rule(null_frame(seed=seed + 1), seed=seed, with_null=True)
        if np.isfinite(r.p_value) and r.p_value < 0.05:
            rejects += 1
    assert rejects / trials <= 0.20, f"{rejects}/{trials} false positives on noise"


def test_reversed_fixture_is_significant_but_not_positive():
    """Significance is not the verdict. A reliably wrong rule must not read as a result."""
    r = safety.safety_rule(reversed_frame())
    assert r.youden_j < 0
    assert r.p_value < 0.05, "two-sided test should detect a reversal"
    assert not r.beats_trivial
    assert r.verdict == "no_better_than_trivial"


# -- refusals ------------------------------------------------------------------


def test_green_lighting_nothing_is_a_refusal_not_a_perfect_score():
    ops = _grid(5)
    sev = _severity(ops)
    # Every signed distance strictly positive, so the rule blocks every pair.
    frame = _frame(ops, lambda a, b: 1.0 + abs(sev[b] - sev[a]), lambda a, b: 0.80)
    r = safety.safety_rule(frame, with_null=False)
    assert r.n_green_lit == 0
    assert np.isnan(r.false_safe_rate)
    assert not r.beats_trivial
    assert r.verdict == "green_lights_nothing"


def test_green_lighting_too_little_is_refused():
    ops = _grid(6)
    sev = _severity(ops)
    # Exactly one ordered pair gets a non-positive distance.
    def signed(a, b):
        return -1.0 if (a, b) == (ops[0], ops[1]) else 1.0

    frame = _frame(ops, signed, lambda a, b: 0.95 if (a, b) == (ops[0], ops[1]) else 0.80)
    r = safety.safety_rule(frame, with_null=False)
    assert r.false_safe_rate == 0.0, "the one green-lit pair is genuinely safe"
    assert r.green_light_fraction < 0.10
    assert r.verdict == "green_lights_too_little"
    assert not r.beats_trivial


def test_no_unsafe_pairs_is_refused():
    ops = _grid(5)
    sev = _severity(ops)
    frame = _frame(ops, lambda a, b: float(sev[b] - sev[a]), lambda a, b: 0.97)
    r = safety.safety_rule(frame, with_null=False)
    assert r.n_unsafe == 0
    assert r.verdict == "no_unsafe_pairs"


def test_sparse_grid_refuses_a_p_value_rather_than_inventing_one():
    """An incomplete pair set breaks the permutation's exchangeability argument."""
    frame = perfect_frame()
    sparse = frame.iloc[::4].reset_index(drop=True)
    r = safety.safety_rule(sparse)
    assert r.pair_completeness < safety.MIN_PAIR_COMPLETENESS
    assert np.isnan(r.p_value)
    assert safety.safety_rule(frame).pair_completeness == pytest.approx(1.0)


# -- bookkeeping that has gone wrong before ------------------------------------


def test_uncertifiable_cells_are_excluded_and_counted():
    frame = perfect_frame()
    frame.loc[:5, ["coverage", "coverage_hi"]] = np.nan
    r = safety.safety_rule(frame, with_null=False)
    assert r.n_excluded == 6
    assert r.n_pairs == len(frame) - 6
    assert r.true_safe + r.false_safe + r.false_alarm + r.true_unsafe == r.n_pairs


def test_false_safe_rate_and_unsafe_miss_rate_are_different_quantities():
    """Same numerator, different denominators. Conflating them has happened before."""
    ops = _grid(6)
    sev = _severity(ops)
    rng = np.random.default_rng(7)
    draws = {p: float(rng.uniform(0.82, 0.96)) for p in _ordered_pairs(ops)}
    frame = _frame(ops, lambda a, b: float(sev[b] - sev[a]), lambda a, b: draws[(a, b)])
    r = safety.safety_rule(frame, with_null=False)
    assert r.false_safe > 0
    assert r.false_safe_rate != pytest.approx(r.unsafe_miss_rate)
    assert r.false_safe_rate == pytest.approx(r.false_safe / r.n_green_lit)
    assert r.unsafe_miss_rate == pytest.approx(r.false_safe / r.n_unsafe)


def test_distance_lookup_fills_the_reverse_direction_antisymmetrically():
    frame = pd.DataFrame(
        [{"calibration": "a", "deployment": "b", "signed_distance": 0.4,
          "coverage": 0.9, "coverage_hi": 0.9}]
    )
    lookup = safety._distance_lookup(frame)
    assert lookup[("a", "b")] == pytest.approx(0.4)
    assert lookup[("b", "a")] == pytest.approx(-0.4)


def test_zero_distance_is_called_safe():
    """The rule is ``<= 0``; a pair at exactly zero must not fall through to blocked."""
    assert safety._predict_safe(np.array([0.0]))[0]


def test_threshold_is_the_frozen_one():
    assert prereg.SIGN_RULE_THRESHOLD == 0.0
    assert prereg.UNSAFE_TOLERANCE == 0.0
    assert prereg.PERMUTATION_STATISTIC == "youden_j"


# -- guards --------------------------------------------------------------------


def test_self_pairs_are_rejected():
    frame = pd.DataFrame(
        [{"calibration": "a", "deployment": "a", "signed_distance": 0.0,
          "coverage": 0.9, "coverage_hi": 0.9}]
    )
    with pytest.raises(ValueError, match="itself"):
        safety.safety_rule(frame)


def test_duplicate_pairs_are_rejected():
    row = {"calibration": "a", "deployment": "b", "signed_distance": 0.1,
           "coverage": 0.9, "coverage_hi": 0.9}
    with pytest.raises(ValueError, match="more than once"):
        safety.safety_rule(pd.DataFrame([row, dict(row)]))


def test_missing_columns_are_named():
    with pytest.raises(KeyError, match="signed_distance"):
        safety.safety_rule(pd.DataFrame([{"calibration": "a", "deployment": "b"}]))


def test_coverage_hi_is_required_not_optional():
    """D85: a silent fall-back to the point estimate would use the superseded label."""
    frame = perfect_frame().drop(columns=["coverage_hi"])
    with pytest.raises(KeyError, match="coverage_hi"):
        safety.safety_rule(frame)


def test_an_upper_end_below_its_point_estimate_is_rejected():
    frame = perfect_frame(5)
    frame.loc[0, "coverage_hi"] = frame.loc[0, "coverage"] - 0.1
    with pytest.raises(ValueError, match="coverage_hi below coverage"):
        safety.safety_rule(frame)


def test_a_wide_interval_forgives_cells_that_merely_scattered_low():
    """The point of the amended label, shown rather than asserted.

    A cell at 0.86 with a wide interval is not evidence of under-coverage; the same cell
    with an exact measurement is. The label must distinguish them, and in the direction
    that makes "unsafe" harder to earn.
    """
    ops = _grid(6)
    sev = _severity(ops)
    signed = lambda a, b: float(sev[b] - sev[a])
    cover = lambda a, b: 0.86 if sev[b] > sev[a] else 0.95

    exact = safety.safety_rule(_frame(ops, signed, cover, half_width=0.0), with_null=False)
    wide = safety.safety_rule(_frame(ops, signed, cover, half_width=0.08), with_null=False)
    assert exact.n_unsafe > 0, "0.86 measured exactly is under-coverage"
    assert wide.n_unsafe == 0, "0.86 +/- 0.08 does not establish under-coverage"
    assert wide.verdict == "no_unsafe_pairs"


def test_the_frozen_label_is_the_interval_form():
    assert prereg.UNSAFE_LABEL == "interval_upper_below_nominal"
    assert prereg.as_dict()["unsafe_label"] == prereg.UNSAFE_LABEL


# -- family breakdown and sensitivity ------------------------------------------


def test_pooled_and_family_cells_are_both_returned():
    frame = perfect_frame(6)
    ops = _grid(6)
    fam = {op: ("sparse" if i < 3 else "limited") for i, op in enumerate(ops)}
    frame["family_calibration"] = frame["calibration"].map(fam)
    frame["family_deployment"] = frame["deployment"].map(fam)
    out = safety.safety_by_family(frame, with_null=False)
    assert out[0].label == "pooled"
    assert len(out) >= 3, "pooled plus the ordered family cells"
    table = safety.safety_table(out)
    assert set(table["cells"]) >= {"pooled"}
    assert (table["prereg"] == prereg.fingerprint()).all()


def test_tolerance_sweep_is_monotone_and_marks_the_frozen_row():
    sweep = safety.tolerance_sensitivity(null_frame())
    assert sweep["frozen_primary"].sum() == 1
    assert sweep.loc[sweep["frozen_primary"], "tolerance"].iloc[0] == prereg.UNSAFE_TOLERANCE
    assert (sweep["n_unsafe"].diff().dropna() <= 0).all(), "a wider band cannot add unsafe cells"
