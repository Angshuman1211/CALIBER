"""T3-C: the exact split of the hidden-content error, and the threshold-direction mechanism.

The split ``s^2 = a^2 + o^2``, ``err^2 = (a - c)^2 + o^2`` is algebra, so it is checked against
vectors built with a known angle rather than against a bank. The model gates pin the two
corners of the (kappa, beta) map that the re-check's claim rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.theory.alignment import (
    decompose,
    elasticities,
    energy_departure,
    log_slope,
    operator_alignment,
    threshold_direction_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def vectors(n: int = 400, dim: int = 32, seed: int = 0):
    """Hidden-content vectors with a chosen alignment: x_hat = kappa * x + w, w random."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    w = rng.normal(size=(n, dim)) * rng.uniform(0.1, 2.0, size=(n, 1))
    kappa = rng.uniform(0.0, 1.2, size=(n, 1))
    xh = kappa * x + w
    s, c = np.linalg.norm(xh, axis=1), np.linalg.norm(x, axis=1)
    e = np.linalg.norm(xh - x, axis=1)
    cos = (xh * x).sum(1) / (s * c)
    return s, c, e, cos


def test_the_split_is_exact_and_recovers_the_angle():
    s, c, e, cos = vectors()
    dec = decompose(s, c, e)
    assert np.allclose(dec["cos"], cos, atol=1e-9)
    assert np.allclose(dec["aligned"] ** 2 + dec["invented"] ** 2, s**2, rtol=1e-9)
    assert np.allclose(dec["recovery_error"] ** 2 + dec["invented"] ** 2, e**2, rtol=1e-9)


def test_the_orthogonal_approximation_is_exact_only_at_zero_alignment():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 16))
    w = rng.normal(size=(200, 16))
    w -= (w * x).sum(1, keepdims=True) / (x * x).sum(1, keepdims=True) * x  # w orthogonal to x
    s, c = np.linalg.norm(w, axis=1), np.linalg.norm(x, axis=1)
    e = np.linalg.norm(w - x, axis=1)
    assert np.abs(decompose(s, c, e)["orth_rel_err"]).max() < 1e-9
    s2, c2, e2, _ = vectors()
    assert np.nanmedian(np.abs(decompose(s2, c2, e2)["orth_rel_err"])) > 0.05


def test_a_region_that_contradicts_the_identity_is_undefined_not_clipped():
    dec = decompose([1.0], [1.0], [3.0])  # |cos| = 3.5
    assert dec["cos"].isna().all() and dec["invented"].isna().all()


def test_log_slope_recovers_a_power_law():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert log_slope(x, 3 * x**0.6) == pytest.approx(0.6)
    assert np.isnan(log_slope(x[:2], x[:2]))


def test_the_model_corners_the_claim_rests_on():
    """Orthogonal fill with invention slower than the hidden truth: tau falls. Recovery: rises."""
    falls = threshold_direction_model(0.0, 0.0, n=60_000)
    rises = threshold_direction_model(1.0, 1.0, invention_scale=0.1, n=60_000)
    assert falls["tau_direction"] == -1 and abs(falls["cos_median"]) < 0.05
    assert rises["tau_direction"] == +1 and rises["cos_median"] > 0.95
    # with invention as fast as the hidden truth, even orthogonal fill does not reverse it
    assert threshold_direction_model(0.0, 1.0, n=60_000)["tau_direction"] == +1


def test_operator_alignment_and_elasticities_on_a_constructed_bank():
    rows = []
    for k, lam in enumerate((1.0, 2.0, 4.0)):
        s, c, e, _ = vectors(seed=k)
        rows.append(pd.DataFrame({
            "operator_id": f"op{k}", "split": np.where(np.arange(s.size) < 40, "design",
                                                        "calib"),
            "s_free": lam * s, "free_content": lam * c, "err_free": lam * e,
            "score": lam * s, "label": (lam * e) > 3.0,
        }))
    table = operator_alignment(pd.concat(rows, ignore_index=True))
    assert list(table["operator_id"]) == ["op0", "op1", "op2"]
    assert table["undefined"].max() == 0.0
    el = elasticities(table).set_index("column")
    assert el.loc["s_median", "elasticity"] == pytest.approx(1.0, abs=0.1)
    assert {"tau", "marginal_cos_median", "identity_spearman"} <= set(table.columns)


def test_the_script_prints_the_model_map(capsys):
    import t3c_alignment_check as script

    assert script.main(["--model"]) == 0
    out = capsys.readouterr().out
    assert "threshold direction model" in out and "k=1.0" in out


# -- T3-D: the departure from the sqrt(null_frac) law, split exactly -------------------------


def _bank(kappa: float, invent_amp, truth_amp, n: int = 300, dim: int = 24, seed: int = 0):
    """Operators with chosen hidden-truth and invention amplitudes; x_hat = kappa x + w."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, dim))
    w0 = rng.normal(size=(n, dim))
    w0 -= (w0 * x0).sum(1, keepdims=True) / (x0 * x0).sum(1, keepdims=True) * x0
    frames = []
    for op in invent_amp:
        x, w = truth_amp[op] * x0, invent_amp[op] * w0
        xh = kappa * x + w
        frames.append(pd.DataFrame({
            "operator_id": op, "operator_family": "f", "split": "calib",
            "sample_id": [f"i{i}" for i in range(n)], "slice_idx": 0, "region_id": 0,
            "s_free": np.linalg.norm(xh, axis=1), "free_content": np.linalg.norm(x, axis=1),
            "err_free": np.linalg.norm(xh - x, axis=1),
        }))
    return pd.concat(frames, ignore_index=True)


def test_white_fill_follows_the_law_and_carries_no_aligned_part():
    nf = {"a": 0.25, "b": 0.5, "c": 0.75}
    amp = {k: np.sqrt(v) for k, v in nf.items()}  # invention amplitude ~ sqrt(null_frac)
    t = energy_departure(_bank(0.0, amp, {k: 1.0 for k in nf}), "a", nf).set_index(
        "operator_id")
    assert np.allclose(t["quotient_s"], 1.0, atol=1e-9)
    assert np.allclose(t["aligned_contribution"], 0.0, atol=1e-9)
    assert np.abs(t["identity_residual"]).max() < 1e-12


def test_pure_recovery_departs_exactly_as_the_truths_hidden_energy_does():
    nf = {"a": 0.25, "b": 0.5}
    truth = {"a": 1.0, "b": 2.0}  # the truth's hidden energy quadruples, the law says doubles
    t = energy_departure(_bank(1.0, {"a": 1e-9, "b": 1e-9}, truth), "a", nf).set_index(
        "operator_id")
    assert t.loc["b", "R_c"] == pytest.approx(4.0)
    assert t.loc["b", "aligned_share_ref"] == pytest.approx(1.0, abs=1e-6)
    assert t.loc["b", "departure"] == pytest.approx(4.0 - 2.0, rel=1e-6)
    assert t.loc["b", "quotient_s"] == pytest.approx(t.loc["b", "quotient_truth"], rel=1e-6)


def test_energy_departure_refuses_an_unknown_reference():
    with pytest.raises(KeyError, match="reference"):
        energy_departure(_bank(0.5, {"a": 1.0}, {"a": 1.0}), "zz")


# -- T3-C: the exact positive branches ------------------------------------------------------


def test_every_positive_is_in_exactly_one_branch():
    from caliber.theory.alignment import _branches

    s, c, e, cos = vectors(n=2000, seed=7)
    eps = float(np.quantile(e, 0.6))
    lab = e > eps
    out = _branches(s, c, cos, eps, lab, lab)
    assert out["branch_inconsistent"] == 0.0
    total = out["marginal_score_free"] + out["marginal_lower_branch"] + out["marginal_upper_branch"]
    assert total == pytest.approx(1.0)


def test_the_two_corners_of_the_branch_split():
    from caliber.theory.alignment import _branches

    eps = 1.0
    # orthogonal, hidden truth below eps, score just over the edge: upper branch, falling
    edge = np.sqrt(eps**2 - 0.6**2)
    up = _branches(np.array([edge + 0.01]), np.array([0.6]), np.array([0.0]), eps,
                   np.array([True]), np.array([True]))
    assert up["marginal_upper_branch"] == 1.0 and up["marginal_boundary_slope"] < 0
    # aligned, large hidden truth, little recovered: lower branch, rising
    low = _branches(np.array([0.5]), np.array([2.0]), np.array([0.95]), eps,
                    np.array([True]), np.array([True]))
    assert low["marginal_lower_branch"] == 1.0 and low["marginal_boundary_slope"] > 0
    # orthogonal with c > eps: positive whatever the score
    free = _branches(np.array([0.01]), np.array([1.5]), np.array([0.0]), eps,
                     np.array([True]), np.array([True]))
    assert free["marginal_score_free"] == 1.0
