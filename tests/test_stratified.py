"""Gates for the stratified detection report and the design-split threshold (T1.4, D62).

D62 moved the anatomy correction out of the label and into the analysis, after measuring
that no function of `err_free` and `free_content` alone is anatomy-neutral within an
operator. These tests pin the two halves of that:

* the threshold comes from the design split and nowhere else;
* the report is computed **within** operator, because the retraction in D62 was exactly a
  pooled statistic hiding two cancelling per-operator biases.

The last test is that retraction, written as a regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation.stratified import (
    detection_by_content,
    stratification_verdict,
)
from caliber.scoring.labels import absolute_threshold_from_design


def _frame(
    n: int = 300,
    operator: str = "op_a",
    content_shift: float = 0.0,
    detector: str = "real",
    seed: int = 0,
) -> pd.DataFrame:
    """Regions with an anatomy density and a label driven by error magnitude.

    ``detector='real'`` gives a score that tracks the invention; ``'anatomy'`` gives one
    that only ranks anatomy density, which is the Q6 failure mode the report must expose.
    """
    rng = np.random.default_rng(seed)
    content = rng.uniform(0.1, 1.0, n) + content_shift
    invention = rng.gamma(2.0, 0.2, n)
    err_free = 0.6 * content + invention
    label = err_free > np.quantile(err_free, 0.70)
    # The label rises with content here, so an anatomy ranker scores it with +content.
    score = invention + 0.05 * rng.normal(size=n) if detector == "real" else content
    return pd.DataFrame(
        {
            "operator_id": operator,
            "free_content": content,
            "err_free": err_free,
            "label": label,
            "score": score,
            "split": "test",
        }
    )


# -- the design-split threshold ------------------------------------------------


def test_the_threshold_is_the_design_quantile_and_ignores_the_other_sides():
    design = _frame(n=200, seed=1).assign(split="design", err_free=np.linspace(0.0, 1.0, 200))
    other = _frame(n=200, seed=2).assign(split="test", err_free=np.full(200, 99.0))
    df = pd.concat([design, other], ignore_index=True)
    eps = absolute_threshold_from_design(df, quantile=0.75)
    assert 0.70 < eps < 0.80, f"should be the design side's q75, got {eps}"


def test_no_design_side_is_refused_rather_than_falling_back():
    """Both fallbacks bias something silently (D-051), so this must stop."""
    with pytest.raises(ValueError, match="design"):
        absolute_threshold_from_design(_frame(seed=3))
    with pytest.raises(ValueError, match="split"):
        absolute_threshold_from_design(_frame(seed=3).drop(columns=["split"]))


def test_a_quantile_cannot_flag_every_region():
    """P-02's defect was the fixed constant, not the absolute form.

    `err_free > 0.10` flagged 100% of regions on real data. A design quantile puts a known
    fraction on each side however the substrate is scaled, which is the property that makes
    the absolute criterion portable across MRI and CT (D-020).
    """
    design = _frame(n=400, seed=4).assign(split="design")
    for scale in (0.01, 1.0, 100.0):
        scaled = design.assign(err_free=design["err_free"] * scale)
        eps = absolute_threshold_from_design(scaled, quantile=0.75)
        rate = float((scaled["err_free"] > eps).mean())
        assert 0.20 < rate < 0.30, f"scale {scale} gave positive rate {rate}"


# -- the stratified report -----------------------------------------------------


def test_an_anatomy_ranker_is_exposed_by_the_strata_not_by_the_pooled_auc():
    """The Q6 failure mode, and the reason the report exists.

    A score that only ranks anatomy density scores well pooled, because the label itself
    rises with anatomy. Holding anatomy fixed, it has nothing left.
    """
    table = detection_by_content(_frame(detector="anatomy", seed=5))
    pooled = table[table["stratum"] == "all"].iloc[0]
    strata = table[(table["stratum"] != "all") & ~table["thin"]]
    assert pooled["auc"] > 0.6, "the pooled number should look respectable"
    assert not strata["beats_anatomy"].any(), "no stratum should survive"
    assert "FAILS" in stratification_verdict(table)


def test_a_real_detector_survives_every_stratum():
    table = detection_by_content(_frame(detector="real", seed=6))
    strata = table[(table["stratum"] != "all") & ~table["thin"]]
    assert len(strata) >= 2
    assert strata["beats_anatomy"].all()
    assert "beats the anatomy artefact" in stratification_verdict(table)


def test_thin_cells_are_marked_and_left_out_of_the_verdict():
    df = _frame(n=40, seed=7)
    table = detection_by_content(df, min_class=12)
    assert table["thin"].any()
    assert table.loc[table["thin"], "auc"].isna().all()


def test_strata_are_cut_within_each_operator_not_globally():
    """The content distribution moves with the operator -- a larger null space hides more of
    the same image -- so global bins would put one operator entirely inside one stratum."""
    df = pd.concat(
        [_frame(operator="op_a", content_shift=0.0, seed=8),
         _frame(operator="op_b", content_shift=5.0, seed=9)],
        ignore_index=True,
    )
    table = detection_by_content(df)
    for operator in ("op_a", "op_b"):
        strata = table[(table["operator_id"] == operator) & (table["stratum"] != "all")]
        assert len(strata) == 3
        assert (strata["n"] > 0).all(), f"{operator} lost a stratum to global binning"


def test_two_cancelling_operator_biases_are_not_reported_as_neutral():
    """D62's retraction, as a regression.

    The candidacy gate looked anatomy-neutral at 0.499 pooled and was 0.662 at R8: two
    opposite per-operator biases averaging to chance. Any report that pools across operators
    can do this, so the function must not offer a cross-operator row at all.
    """
    df = pd.concat(
        [_frame(operator="low_bias", detector="anatomy", seed=10),
         _frame(operator="high_bias", detector="real", seed=11)],
        ignore_index=True,
    )
    table = detection_by_content(df)
    assert set(table["operator_id"]) == {"low_bias", "high_bias"}
    assert "all" not in set(table["operator_id"]), "there must be no pooled-operator row"
    # And the verdict must name the failing operator rather than averaging it away.
    verdict = stratification_verdict(table)
    assert "FAILS" in verdict and "low_bias" in verdict
