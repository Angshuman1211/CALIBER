"""The weight-free Lipschitz bracket T3.3 depends on, and the script that computes it.

D41 hands Track 3 a bound that does not wait on the score weights: because D25
constrains the weights to the simplex, ``L = Σ wᵢ Lᵢ`` is a convex combination and
therefore lies between the smallest and largest per-component constant, whatever the
weights turn out to be.

That is a claim about `combine_components`, not about arithmetic in the abstract, and it
holds only while the combination stays affine with non-negative weights summing to one
(D25) and the scales stay frozen (D27). Both are properties of the code that someone
could change while making something else better. The first test below fails if either
does, which is the point of it.

The script test follows P-19's lesson: exercising the library a script calls is not the
same as executing the script.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from caliber.config import load_config
from caliber.runner import run
from caliber.scorebank import read_bank
from caliber.scoring.attribution import identity_residual
from caliber.scoring.score import ScoreComponents, ScoreScales, ScoreWeights, combine_components

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import t1_component_lipschitz  # noqa: E402

COMMITTED_BANK = REPO_ROOT / "banks" / "w02_solver_smoke"


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    """A real two-operator bank on disk, from the phantom smoke config."""
    out = tmp_path_factory.mktemp("banks")
    cfg = load_config(
        str(REPO_ROOT / "configs/experiments/w01_smoke_phantom.yaml"),
        [
            "run.limit=4",
            "experiment.n_posterior=4",
            "run.cache_reconstructions=0",
            "run.skip_memory_check=true",
            f"run.output_root={out.as_posix()}",
        ],
    )
    return run(cfg)


def _components(rng: np.random.Generator, n: int = 64) -> ScoreComponents:
    return ScoreComponents(
        s_free=torch.tensor(rng.uniform(0.1, 1.0, n)),
        s_disagree=torch.tensor(rng.uniform(0.1, 1.0, n)),
        s_residual=torch.tensor(rng.uniform(0.1, 1.0, n)),
        ref_intensity=1.0,
    )


def _secants(left: ScoreComponents, right: ScoreComponents, scales: ScoreScales, d_op: float):
    """Signed standardised mean shift per component, and its ratio — as the script does."""
    out = {}
    for name, scale in (("s_free", scales.free), ("s_disagree", scales.disagree),
                        ("s_residual", scales.residual)):
        shift = float((getattr(right, name) - getattr(left, name)).mean()) / scale
        out[name] = (shift, abs(shift) / d_op)
    return out


def test_no_weighting_can_push_the_combination_above_the_largest_component():
    """D41's bound, tested against the real combination rule over the whole simplex.

    This is the half that is true, and the half T3.3 needs: a coverage-gap bound wants an
    *upper* bound on `L`. It holds because D25 keeps the weights non-negative and summing
    to one, so the combination is convex and the triangle inequality applies.
    """
    rng = np.random.default_rng(0)
    left, right = _components(rng), _components(rng)
    scales = ScoreScales.from_components(left)
    d_op = 0.35
    hi = max(ratio for _, ratio in _secants(left, right, scales, d_op).values())

    for _ in range(200):
        w = rng.dirichlet([1.0, 1.0, 1.0])
        weights = ScoreWeights(free=float(w[0]), disagree=float(w[1]), residual=float(w[2]))
        combined = abs(
            float(
                (combine_components(right, weights, scales)
                 - combine_components(left, weights, scales)).mean()
            )
        ) / d_op
        assert combined <= hi + 1e-9, (
            f"weights {w} put the combination at {combined:.4f}, above the largest "
            f"component's {hi:.4f} — the combination is no longer convex (D25)"
        )


def test_opposing_components_cancel_so_there_is_no_lower_bracket():
    """The half that is false, pinned so nobody re-derives it.

    `min_i L_i ≤ L` looks like the natural companion to the bound above and does not
    hold: components that move in opposite directions under the same operator change
    cancel, and the combination is then *less* operator-sensitive than any part of it.
    This is T3.4 §3's cross-component cancellation, and on the real-prior bank it is
    nearly total — `s_residual` falls where the other two rise.
    """
    rng = np.random.default_rng(1)
    left = _components(rng)
    right = ScoreComponents(
        s_free=left.s_free + 0.20,        # rises
        s_disagree=left.s_disagree + 0.20,  # rises
        s_residual=left.s_residual - 0.20,  # falls, as measured on the real bank
        ref_intensity=1.0,
    )
    scales = ScoreScales.from_components(left)
    d_op = 0.35
    secants = _secants(left, right, scales, d_op)
    signs = {shift > 0 for shift, _ in secants.values()}
    assert len(signs) == 2, "this control needs components moving in both directions"

    weights = ScoreWeights(free=1 / 3, disagree=1 / 3, residual=1 / 3)
    combined = abs(
        float((combine_components(right, weights, scales)
               - combine_components(left, weights, scales)).mean())
    ) / d_op
    lo = min(ratio for _, ratio in secants.values())
    assert combined < lo, (
        "cancellation did not occur, so this test is no longer demonstrating the thing "
        "it exists to demonstrate"
    )


def test_excluding_a_component_can_only_tighten_the_bound():
    """D42's claim: dropping a component lowers `max_i L_i`, never raises it.

    Numbers are the measured ones from the real-prior bank, so this also pins the fact
    that `s_residual` is the largest of the three.
    """
    ratios = {"s_free": 0.4916, "s_disagree": 0.5716, "s_residual": 1.0970}
    two = {k: v for k, v in ratios.items() if k != "s_residual"}
    assert max(two.values()) < max(ratios.values())


def test_the_script_runs_end_to_end_on_a_bank(bank, capsys):
    """P-19's lesson: execute the script, not only the library it calls."""
    assert t1_component_lipschitz.main([str(bank)]) == 0
    out = capsys.readouterr().out
    assert "d_op" in out
    assert "ell_s_free" in out
    assert "shift_s_free" in out, "signs must be printed, not only magnitudes"
    assert "L  <=  max_i L_i" in out


def test_the_script_refuses_a_single_operator_bank(bank, tmp_path_factory, capsys):
    """One operator is no secant at all, and must be said rather than divided by zero.

    Built by filtering an existing bank rather than by configuring a one-operator run:
    `experiment.operators[0].acceleration=[4]` is accepted and silently ignored by the
    config loader, so a test written that way would quietly exercise the wrong thing.
    """
    out = tmp_path_factory.mktemp("one-op") / "single"
    shutil.copytree(bank, out)
    df = read_bank(bank)
    keep = sorted(df["operator_id"].unique())[0]
    df[df["operator_id"] == keep].to_parquet(out / "scores.parquet", index=False)

    assert t1_component_lipschitz.main([str(out)]) == 1
    assert "at least two operators" in capsys.readouterr().out


@pytest.mark.skipif(not COMMITTED_BANK.exists(), reason="committed bank not present")
def test_the_committed_real_prior_bank_satisfies_the_identity():
    """D-045's bank is a shared interface now; D-037's identity is its integrity check.

    A non-zero residual here means the score components and the truth columns were
    normalised differently, which would invalidate every ratio Q6 and D41 read off it.
    """
    df = read_bank(COMMITTED_BANK)
    res = identity_residual(df["s_free"], df["free_content"], df["err_free"])
    assert res["n"] > 0
    assert res["max_relative"] < 1e-3
