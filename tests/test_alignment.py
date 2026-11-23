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
