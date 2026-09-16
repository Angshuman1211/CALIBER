"""Gates for the Q6 attribution diagnostic (T1.1).

The real bank is not on this machine, so the diagnostic is validated the only way
that is honest: against banks whose answer is known by construction. Three
controls, and the diagnostic has to get all three right before its reading of a
real bank means anything.

* **anatomy-confounded** — ``s_free`` is an exact multiple of the truth's own
  null-space content and nothing else. It must score a high plain AUC (because
  regions with more anatomy happen to be labelled more often) and must collapse
  to chance once anatomical content is held fixed. This is Q6's feared outcome
  and the diagnostic exists to catch it.
* **invention-driven** — ``s_free`` tracks the invention magnitude, with
  anatomical content independent. Separation must survive the control.
* **uninformative** — ``s_free`` is noise. Neither AUC may look informative.

Plus the identity the whole module rests on, verified end to end through the
repository's own scoring and labelling code rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from caliber.operators.mri import MRIUndersamplingOperator
from caliber.scoring.attribution import (
    ANATOMY_RHO_HIGH,
    AUC_INFORMATIVE,
    attribute_components,
    component_ablation,
    content_alignment,
    fill_ratio,
    identity_residual,
    partial_spearman,
    q6_verdict,
    rank_auc,
    spearman,
    stratified_auc,
    within_cell_rank,
)
from caliber.scoring.labels import compute_region_truth
from caliber.scoring.regions import RegionGrid
from caliber.scoring.score import compute_score_components

TAU = 1.75


# -- rank statistics -----------------------------------------------------------


def test_rank_auc_is_one_when_every_positive_outranks_every_negative():
    assert rank_auc([1.0, 2.0, 3.0, 4.0], [False, False, True, True]) == 1.0
    assert rank_auc([1.0, 2.0, 3.0, 4.0], [True, True, False, False]) == 0.0


def test_rank_auc_returns_chance_for_a_constant_score_not_an_ordering_artefact():
    """A structurally dead component (D35) must read as chance, not as 0 or 1.

    Without tie correction the answer depends on the sort's stability, which
    would make a degenerate component look like a perfect detector on one
    platform and a perfect anti-detector on another.
    """
    assert rank_auc([2.0] * 6, [True, True, True, False, False, False]) == 0.5


def test_rank_auc_is_nan_when_a_class_is_empty_rather_than_silently_chance():
    assert np.isnan(rank_auc([1.0, 2.0, 3.0], [False, False, False]))
    assert np.isnan(rank_auc([1.0, 2.0, 3.0], [True, True, True]))


def test_partial_spearman_removes_a_shared_driver():
    rng = np.random.default_rng(0)
    control = rng.normal(size=400)
    a = control + 0.05 * rng.normal(size=400)
    b = control + 0.05 * rng.normal(size=400)
    assert spearman(a, b) > 0.9
    assert abs(partial_spearman(a, b, control)) < 0.3


def test_within_cell_rank_is_invariant_to_a_per_cell_rescaling():
    """Scores are deliberately *not* comparable across operators (D27).

    The scales are frozen on the calibration operator so the distributions may
    differ, and that difference is the headline result. Any statistic pooled
    across operators therefore has to be rank-normalised within operator first,
    or operator-level shifts drive it.
    """
    df = pd.DataFrame(
        {
            "operator_id": ["a"] * 5 + ["b"] * 5,
            "s_free": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    ranks = within_cell_rank(df, "s_free", ("operator_id",)).to_numpy()
    assert np.allclose(ranks[:5], ranks[5:])


# -- the identity, through the repository's own code ---------------------------


def _end_to_end_columns(acceleration: float = 4.0, n_draws: int = 4):
    """One item through the real scoring and labelling path, as the runner wires it."""
    shape = (64, 64)
    op = MRIUndersamplingOperator(
        shape, family="random1d", acceleration=acceleration, center_fraction=0.08, mask_seed=1
    )
    grid = RegionGrid(shape, region_size=16)
    gen = torch.Generator().manual_seed(7)

    yy = torch.linspace(-1, 1, shape[0]).view(-1, 1)
    xx = torch.linspace(-1, 1, shape[1]).view(1, -1)
    x_true = (0.4 + 0.6 * torch.exp(-(yy**2 + xx**2) * 3.0)) * (
        (yy**2 + xx**2) < 0.81
    ).float()
    x_true = x_true + 0.15 * torch.rand(shape, generator=gen)

    y = op.forward(x_true)
    base = op.pinv(y)
    draws = []
    for _ in range(n_draws):
        noise = 0.3 * torch.randn(shape, generator=gen)
        draws.append(base + op.project_null(noise.to(base.dtype)))
    samples = torch.stack(draws, dim=0)

    # Exactly the runner's wiring: one reference intensity, shared by both sides
    # (runner.py passes components.ref_intensity into compute_region_truth). The
    # identity holds only because of that, which is why the test asserts it here
    # rather than trusting the docstring.
    components = compute_score_components(samples, y, op, grid)
    truth = compute_region_truth(
        samples, x_true, op, grid, None, components.ref_intensity
    )
    return op, grid, samples, x_true, components, truth


def test_the_three_free_component_columns_satisfy_the_identity_exactly():
    _, _, _, _, components, truth = _end_to_end_columns()
    res = identity_residual(
        components.s_free.numpy(), truth.free_content.numpy(), truth.err_free.numpy()
    )
    assert res["n"] > 0
    # float32 columns through a difference of squares; anything above 1e-3 would
    # mean the two sides were normalised differently.
    assert res["max_relative"] < 1e-3


def test_content_alignment_recovers_the_true_angle_between_hidden_contents():
    """``cos θ`` from three scalars per region must equal the honest inner product."""
    op, grid, samples, x_true, components, truth = _end_to_end_columns()
    mean_recon = samples.mean(dim=0)
    a = op.project_null(mean_recon)
    b = op.project_null(x_true.to(a.dtype))

    inner = grid.reduce(torch.real(a * torch.conj(b)), how="mean")
    norm_a = grid.reduce(a.abs(), how="rms")
    norm_b = grid.reduce(b.abs(), how="rms")
    expected = (inner / (norm_a * norm_b)).numpy()

    got = content_alignment(
        components.s_free.numpy(), truth.free_content.numpy(), truth.err_free.numpy()
    )
    keep = np.isfinite(got)
    assert keep.sum() >= grid.n_regions // 2
    assert np.abs(got[keep] - expected[keep]).max() < 1e-2


def test_alignment_is_nan_rather_than_clipped_when_the_identity_cannot_hold():
    """Out-of-range ``cos θ`` is evidence the bank is inconsistent, not a rounding nuisance."""
    got = content_alignment([1.0], [1.0], [3.0])  # implies cos theta = -3.5
    assert np.isnan(got[0])


def test_over_filling_forces_the_relative_ratio_above_one_whatever_the_alignment():
    """D37's pass/fail check has an unstated precondition, and this is it.

    ``err_free/free_content > 1`` was read as evidence of a mismatched prior. The
    identity says it is *forced* whenever ``u = s_free/free_content > 2``, at any
    alignment — so with an over-filling prior the check can never fail and proves
    nothing about the prior's suitability.
    """
    rng = np.random.default_rng(1)
    u = rng.uniform(2.01, 6.0, size=500)
    cos = rng.uniform(-1.0, 1.0, size=500)
    ratio = np.sqrt(u**2 + 1.0 - 2.0 * u * cos)
    assert ratio.min() > 1.0

    # And below u = 2 it is genuinely diagnostic: a well-aligned prior can land under 1.
    u_small = np.full(500, 1.0)
    ratio_small = np.sqrt(u_small**2 + 1.0 - 2.0 * u_small * cos)
    assert ratio_small.min() < 1.0


def test_fill_ratio_refuses_regions_with_no_free_content_to_speak_of():
    u = fill_ratio([1.0, 1.0], [1.0, 1e-9])
    assert u[0] == 1.0
    assert np.isnan(u[1])


# -- the three controls --------------------------------------------------------


def _bank(kind: str, n: int = 800, seed: int = 0) -> pd.DataFrame:
    """A bank whose Q6 answer is known by construction.

    All three share the same label definition — the relative criterion at
    ``TAU`` — and differ only in what ``s_free`` is built from.
    """
    rng = np.random.default_rng(seed)
    free_content = rng.uniform(0.4, 2.0, size=n)

    if kind == "anatomy":
        # The label rate rises with anatomical content, and `s_free` measures
        # nothing but anatomical content. Any apparent detection is the confound.
        ratio = 1.0 + 1.6 * (free_content - 0.4) / 1.6 + 0.10 * rng.normal(size=n)
        s_free = 3.0 * free_content
    elif kind == "invention":
        # Invention drives the label; `s_free` measures invention; anatomy is
        # independent of both.
        invention = rng.gamma(2.0, 0.5, size=n)
        ratio = 1.0 + invention + 0.10 * rng.normal(size=n)
        s_free = invention + 0.15 * rng.normal(size=n)
    elif kind == "noise":
        ratio = 1.0 + rng.gamma(2.0, 0.5, size=n) + 0.10 * rng.normal(size=n)
        s_free = rng.normal(size=n)
    else:  # pragma: no cover - guard against a typo in a test
        raise ValueError(kind)

    err_free = np.clip(ratio, 0.0, None) * free_content
    return pd.DataFrame(
        {
            "operator_id": np.where(rng.random(n) < 0.5, "mri.random1d.R4", "mri.random1d.R8"),
            "solver_id": "diffusion.ddnm.test",
            "calib_group": np.where(rng.random(n) < 0.5, "g1", "g2"),
            "split": np.where(rng.random(n) < 0.5, "calib", "test"),
            "region_id": np.arange(n),
            "free_content": free_content,
            "err_free": err_free,
            "err_total": err_free,
            "ann_coverage": 0.0,
            "s_free": s_free,
            "s_disagree": 0.5 * s_free + 0.5 * rng.normal(size=n),
            "s_residual": 1e-8 * rng.normal(size=n),
            "label": (err_free / free_content) > TAU,
        }
    )


def test_anatomy_confounded_bank_is_caught_as_anatomy_confounded():
    """And caught by the paired reference, not by the stratified AUC alone.

    This control is the reason ``AUC_EXCESS_MIN`` exists. ``s_free`` here is an
    exact multiple of ``free_content``, so it is a pure anatomy meter and the
    right verdict is not in doubt — yet its detail-matched AUC stays near 0.87,
    because stratifying on a *continuous* confound with four bins only attenuates
    it. An absolute threshold on the detail-matched AUC would read that as a
    strong detector and Q6 would be answered wrongly. Scoring ``free_content``
    itself through the same path gives the same 0.87, and the difference is zero.
    """
    df = _bank("anatomy", seed=1)
    assert 0.1 < df["label"].mean() < 0.9, "control bank must have both classes well represented"
    table = attribute_components(df).set_index("component")
    row = table.loc["s_free"]

    assert row["auc"] >= AUC_INFORMATIVE
    # Stratification alone does NOT rescue this — the documented reason for the
    # paired instrument. If this assertion ever starts failing because the
    # detail-matched AUC collapsed on its own, the finding above has changed.
    assert row["auc_detail_matched"] > 0.7
    # The paired comparison is what settles it: no better than anatomy itself.
    assert abs(row["auc_over_anatomy"]) < 1e-9
    assert row["rho_anatomy"] >= ANATOMY_RHO_HIGH
    assert row["verdict"] == "anatomy-confounded"
    assert q6_verdict(table.reset_index()).verdict == "anatomy-confounded"


def test_strong_confounding_starves_the_strata_and_that_is_itself_reported():
    """When the control nearly determines the label, most strata lose a class.

    One surviving stratum out of four is not a weaker version of the same
    evidence — it is a symptom, and the verdict text has to say so rather than
    quoting a confident-looking number computed on a quarter of the bank.
    """
    row = attribute_components(_bank("anatomy", seed=1)).set_index("component").loc["s_free"]
    assert row["strata_used"] < 4
    assert "strata survived" in row["reason"] or row["strata_used"] >= 2


def test_invention_driven_bank_survives_the_detail_matched_control():
    df = _bank("invention", seed=2)
    table = attribute_components(df).set_index("component")
    row = table.loc["s_free"]
    assert row["auc"] >= AUC_INFORMATIVE
    assert row["auc_detail_matched"] >= AUC_INFORMATIVE
    assert row["rho_invention_given_anatomy"] > 0.2
    assert row["verdict"] == "invention-driven"


def test_pure_noise_component_reads_as_uninformative():
    df = _bank("noise", seed=3)
    row = attribute_components(df).set_index("component").loc["s_free"]
    assert row["verdict"] == "uninformative"


def _identity_bank(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """A bank that satisfies the identity, built from ``(free_content, u, cos θ)``.

    The three columns of a real bank are not free to vary independently — they
    are three reductions of two fields. A control that draws them independently
    can validate the rank statistics but says nothing about the ratio argument,
    because on such a bank ``s_free / free_content`` is not the quantity the
    label is monotone in. This generator picks the two physical degrees of
    freedom and derives the columns, so the identity holds by construction.
    """
    rng = np.random.default_rng(seed)
    free_content = rng.uniform(0.4, 2.0, size=n)  # how much anatomy is hidden here
    u = rng.gamma(4.0, 0.45, size=n)  # how much the prior put there, relative
    cos = rng.uniform(-0.3, 0.7, size=n)  # how much of it agrees with the truth
    ratio = np.sqrt(u**2 + 1.0 - 2.0 * u * cos)
    return pd.DataFrame(
        {
            "operator_id": "mri.random1d.R4",
            "solver_id": "diffusion.ddnm.test",
            "calib_group": np.where(rng.random(n) < 0.5, "g1", "g2"),
            "split": np.where(rng.random(n) < 0.5, "calib", "test"),
            "region_id": np.arange(n),
            "free_content": free_content,
            "s_free": u * free_content,
            "err_free": ratio * free_content,
            "err_total": ratio * free_content,
            "ann_coverage": 0.0,
            "s_disagree": 0.3 * u * free_content + 0.7 * rng.normal(size=n),
            "s_residual": 1e-8 * rng.normal(size=n),
            "fill_ratio_true": u,
            "label": ratio > TAU,
        }
    )


def test_identity_bank_round_trips_through_the_diagnostic():
    """The generator's own algebra must survive `content_alignment`'s inversion."""
    df = _identity_bank(seed=9)
    res = identity_residual(df["s_free"], df["free_content"], df["err_free"])
    assert res["max_relative"] < 1e-9
    u = fill_ratio(df["s_free"], df["free_content"])
    assert np.abs(u - df["fill_ratio_true"].to_numpy()).max() < 1e-9


def test_the_oracle_ratio_beats_the_raw_component_on_an_identity_consistent_bank():
    """The mechanism, not just the symptom.

    The relative criterion is monotone in ``u = s_free / free_content`` at fixed
    alignment, so ``u`` is the statistic the label is built on and ``s_free`` is
    that statistic multiplied by an anatomical factor the label divides back out.
    Dividing by local content must therefore recover detection that the raw
    component loses — which is the argument for normalising the component by a
    *deployment-available* measure of local content (the A2 ask) rather than
    reporting ``s_free`` raw.
    """
    df = _identity_bank(seed=4)
    assert 0.1 < df["label"].mean() < 0.9
    df["ORACLE_s_free_over_content"] = df["s_free"] / df["free_content"]
    table = attribute_components(
        df, component_cols=("s_free", "ORACLE_s_free_over_content")
    ).set_index("component")
    assert table.loc["ORACLE_s_free_over_content", "auc"] > table.loc["s_free", "auc"] + 0.1
    assert table.loc["ORACLE_s_free_over_content", "verdict"] == "invention-driven"


# -- refusals ------------------------------------------------------------------


def test_a_bank_without_the_a1_column_is_refused_not_answered_weakly():
    df = _bank("invention").drop(columns=["free_content"])
    with pytest.raises(KeyError):
        attribute_components(df)


def test_a_bank_whose_a1_column_is_all_null_is_refused():
    df = _bank("invention")
    df["free_content"] = np.nan
    with pytest.raises(ValueError):
        attribute_components(df)


def test_thin_classes_give_no_usable_stratum_rather_than_a_confident_number():
    rng = np.random.default_rng(5)
    scores = rng.normal(size=200)
    labels = np.zeros(200, dtype=bool)
    labels[:4] = True  # four positives over four strata
    control = rng.normal(size=200)
    strat = stratified_auc(scores, labels, control, n_strata=4)
    assert strat.strata_used == 0
    assert not strat.usable
    assert np.isnan(strat.value)


def test_verdict_is_undecidable_when_the_component_is_absent():
    table = attribute_components(_bank("invention"), component_cols=("s_disagree",))
    assert q6_verdict(table).verdict == "undecidable"


# -- the ablation --------------------------------------------------------------


def test_ablation_compares_flag_area_and_reports_calibration_held():
    df = _bank("invention", n=1200, seed=6)
    df["score"] = 0.5 * df["s_free"] + 0.5 * df["s_disagree"]
    out = component_ablation(df, candidates={"score": "score", "s_disagree": "s_disagree"})
    assert set(out["candidate"]) == {"score", "s_disagree"}
    assert out["flag_area"].between(0.0, 1.0).all()
    # Calibration is what pins the miss rate, so this is the check that it held
    # — not the comparison. The comparison is flag_area.
    assert out["miss_rate"].max() < 0.35


def test_d2_tie_is_reported_when_disagreement_alone_matches_the_combination():
    """The honest-negative outcome Q6 flagged, made detectable.

    If the combination is a monotone function of ``s_disagree`` then every
    conformal threshold puts the flags in the same places, the flagged areas are
    identical, and the free-component decomposition has bought nothing. The
    verdict must say so rather than quietly reporting two similar numbers.
    """
    df = _bank("invention", n=1200, seed=7)
    df["score"] = 2.0 * df["s_disagree"] + 1.0  # same ordering, different units
    attribution = attribute_components(df, component_cols=("s_free", "s_disagree", "score"))
    ablation = component_ablation(
        df, candidates={"score": "score", "s_disagree": "s_disagree"}
    )
    verdict = q6_verdict(attribution, ablation)
    assert verdict.decomposition_earns_its_place is False
    assert "flagged area" in verdict.summary() or verdict.notes


def test_d2_check_is_skipped_rather_than_read_off_an_uncertifiable_group():
    """A flagged area from a group too thin to certify is not a guarantee (D32)."""
    df = _bank("invention", n=120, seed=8)
    df["calib_group"] = [f"g{i % 30}" for i in range(len(df))]  # ~4 rows per group
    df["score"] = df["s_disagree"]
    attribution = attribute_components(df, component_cols=("s_free", "s_disagree", "score"))
    ablation = component_ablation(
        df, candidates={"score": "score", "s_disagree": "s_disagree"}
    )
    assert not ablation["reportable"].all()
    verdict = q6_verdict(attribution, ablation)
    assert verdict.decomposition_earns_its_place is None
    assert any("thin to certify" in n for n in verdict.notes)
