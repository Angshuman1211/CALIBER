"""Selecting the score's component weights on the design split (T1.1, D28).

**Track 1 owns this module.** It closes the last unset number in the score definition:
§18 asks T1.1 for the score's *definition*, and an unweighted score is a family of scores
rather than one, so the weights are part of the deliverable and not validation of it.

## What is already settled before this runs

* **D25** — weights are non-negative and sum to one, so the combination is convex.
* **D26/D27** — components are standardised by scales frozen on the calibration operator.
* **D42** — `s_residual` carries no weight. The simplex is therefore one-dimensional:
  `w_free + w_disagree = 1`, and selecting the score means selecting one scalar.
* **D39** — the comparison is **flagged area at fixed α**, never power. A conformal
  threshold pins the miss rate, so power is ≈ 1 − α for every candidate and would be
  comparing calibrators rather than scores.
* **D48** — selection happens on the `design` split and nowhere else.

## The rule, pre-registered

Written before any bank with a design side exists, so the rule cannot be chosen to suit
what the data turns out to say — the same discipline D37 applied to τ and Q6 to its
verdict thresholds.

1. Sweep `w_free` over a fixed grid on the **design** rows only.
2. At each point, calibrate group-conditionally at α and record flagged area, pooled over
   operators.
3. Take the minimum, then **the set of points within one bootstrap SE of it** — because
   the objective is flat. Measured on the real-prior bank, flag area moves by 0.015 across
   `w_free ∈ [0.3, 0.6]`, which is inside the noise of a design split of eight patients.
   Picking the raw argmin of a flat curve selects noise.
4. **Break that tie toward the larger `w_free`.** Within a statistically indistinguishable
   set, prefer the weight that gives the tighter coverage-gap bound: D41 measured
   `L = w·L_free + (1−w)·L_disagree` with `L_free = 0.4916 < L_disagree = 0.5716`, so `L`
   falls monotonically as `w_free` rises. This buys Track 3 a tighter constant at no
   measurable detection cost, and it is a *principled* tie-break rather than a second
   pass over the same data.

**Endpoints are admissible and are reported, not avoided.** If the selection lands at
`w_free = 1`, cross-sample disagreement contributes nothing and that is a substantive
finding about the score's composition — the mirror of D2's honest-negative — not a
degenerate case to be excluded by fiat. :attr:`WeightSelection.at_endpoint` says so.

## What this does not claim

The design rows are both calibrated and evaluated on here, so the *level* of the flagged
area is optimistically biased. The bias is common to every point on the sweep, so the
*comparison* — which is all the rule uses — stays valid. Nothing about the level should be
quoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..calibration.calibrators import apply_flags, calibrate_mondrian
from ..evaluation.metrics import flag_area, miss_rate
from .score import ScoreComponents, ScoreScales, ScoreWeights, combine_components

#: The sweep. Eleven points is enough to locate a flat region without implying the
#: objective is resolved more finely than a design split can resolve it.
DEFAULT_GRID = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 11))

#: Per-component local Lipschitz ratios measured on the real-prior bank (D41). Used only
#: for the tie-break, and overridable, because they are a property of a bank rather than
#: of the score.
L_FREE = 0.4916
L_DISAGREE = 0.5716

#: Bootstrap replicates for the SE that defines "indistinguishable".
N_BOOT = 500

#: Largest share of the sweep that may be tied with the minimum before the selection is
#: declared uninformative.
#:
#: Without this, D49's tie-break substitutes for data. Track 2 ran the selection on 8 design
#: patients and it returned `w_free = 1.00` -- an endpoint, dropping `s_disagree` from the
#: score entirely -- because **8 of 11** points sat within one SE of the minimum and the
#: tie-break walks toward the tighter Lipschitz bound. That is not a finding about the
#: score, it is the rule reporting the boundary of its own search.
#:
#: The size of the tied set is the right instrument, and more direct than comparing the
#: sweep's range against its noise: if most of the sweep is statistically indistinguishable
#: from the best point, the sweep has not chosen anything. Half is the natural line.
TIED_FRACTION_LIMIT = 0.5


def lipschitz_ceiling(
    w_free: float, l_free: float = L_FREE, l_disagree: float = L_DISAGREE
) -> float:
    """``w·L_free + (1−w)·L_disagree`` — the convex combination from D41."""
    return float(w_free * l_free + (1.0 - w_free) * l_disagree)


@dataclass(frozen=True)
class WeightSelection:
    """The selected weight, the curve it came from, and why that point."""

    w_free: float
    w_disagree: float
    flag_area: float
    se: float
    lipschitz: float
    n_design_clusters: int
    tied_with: tuple[float, ...]
    discriminated: bool = True
    curve: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def at_endpoint(self) -> bool:
        """True if one component was dropped entirely — a finding, not a failure."""
        return self.w_free in (0.0, 1.0)

    @property
    def weights(self) -> ScoreWeights:
        """The ratified object, with `s_residual` at zero per D42."""
        return ScoreWeights(free=self.w_free, disagree=self.w_disagree, residual=0.0)

    def note(self) -> str:
        """One line stating what was selected and on what, so a table is never orphaned."""
        if not self.discriminated:
            return (
                f"w_free={self.w_free:.2f}, w_disagree={self.w_disagree:.2f} -- NOT "
                f"SELECTED. The design split ({self.n_design_clusters} clusters) cannot "
                f"separate any weight from any other: {len(self.tied_with)} of the sweep's "
                f"points are tied with the minimum. Defaulted to equal weights over the "
                f"live components (D25), which is a stated default and not a measurement."
            )
        tie = (
            f"tied with {len(self.tied_with)} point(s) within 1 SE; "
            f"broken toward the tighter bound (D28 step 4)"
            if len(self.tied_with) > 1
            else "a clear minimum, no tie-break needed"
        )
        end = " ENDPOINT -- one component carries no weight" if self.at_endpoint else ""
        return (
            f"w_free={self.w_free:.2f}, w_disagree={self.w_disagree:.2f} "
            # ASCII only: this string is printed, and a Windows console at cp1252
            # raises UnicodeEncodeError on the typographic forms (Track 2, Week 5).
            f"(flag area {self.flag_area:.4f} +/- {self.se:.4f}, L <= {self.lipschitz:.4f}, "
            f"{self.n_design_clusters} design clusters) -- {tie}{end}"
        )


def _components(frame: pd.DataFrame) -> ScoreComponents:
    return ScoreComponents(
        s_free=torch.tensor(frame["s_free"].to_numpy()),
        s_disagree=torch.tensor(frame["s_disagree"].to_numpy()),
        s_residual=torch.tensor(frame["s_residual"].to_numpy()),
        ref_intensity=1.0,
    )


def _cluster_se(flags: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> float:
    """Bootstrap SE of flagged area, resampling clusters — the exchangeable unit (P-23)."""
    keys = np.unique(clusters)
    if keys.size < 2:
        return float("nan")
    flagged = np.array([float(flags[clusters == k].sum()) for k in keys])
    counts = np.array([float((clusters == k).sum()) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, keys.size, size=(n_boot, keys.size))
    boot = flagged[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1.0)
    return float(boot.std(ddof=1))


def weight_curve(
    df: pd.DataFrame,
    calibration_operator: str,
    grid: tuple[float, ...] = DEFAULT_GRID,
    alpha: float = 0.10,
    label_col: str = "label",
    group_col: str = "calib_group",
    cluster_col: str = "sample_id",
    split_col: str = "split",
    design_split: str = "design",
    n_boot: int = N_BOOT,
    seed: int = 0,
    l_free: float = L_FREE,
    l_disagree: float = L_DISAGREE,
) -> pd.DataFrame:
    """Flagged area against ``w_free`` on the design rows, with a cluster SE at each point.

    Raises if the frame carries no design rows. That is deliberate: selecting on
    calibration rows biases the conformal threshold and selecting on test rows biases the
    coverage number, and both failures are silent (D-051). A missing design side is a
    reason to stop, not to fall back.
    """
    if split_col in df.columns:
        design = df[df[split_col].astype(str) == design_split]
    else:
        design = df.iloc[0:0]
    if design.empty:
        raise ValueError(
            f"no rows with {split_col}={design_split!r}; weight selection requires a design "
            "split (D48). Regenerate the bank with the dataset's design_fraction set -- it "
            "is a dataset kwarg, so `--set data.dataset.design_fraction=0.10`, and it is "
            "wired for fastmri only (the phantom dataset takes one fixed split)."
        )

    calib_rows = design[design["operator_id"].astype(str) == str(calibration_operator)]
    if calib_rows.empty:
        raise KeyError(
            f"the design split has no rows at {calibration_operator!r}, so the scales "
            "cannot be frozen on the calibration operator (D27)"
        )
    scales = ScoreScales.from_components(_components(calib_rows))

    rows = []
    for w in grid:
        weights = ScoreWeights(free=float(w), disagree=float(1.0 - w), residual=0.0)
        work = design.copy()
        work["score"] = combine_components(_components(work), weights, scales).numpy()
        report = calibrate_mondrian(
            work, score_col="score", label_col=label_col, group_col=group_col,
            alpha=alpha, split_col="__none__",
        )
        flagged = apply_flags(work, report, score_col="score", group_col=group_col)
        flags = flagged["flag"].to_numpy().astype(bool)
        labels = flagged[label_col].to_numpy().astype(bool)
        rows.append(
            {
                "w_free": float(w),
                "w_disagree": float(1.0 - w),
                "flag_area": flag_area(flags),
                "se": _cluster_se(flags, flagged[cluster_col].to_numpy(), n_boot, seed),
                "miss_rate": miss_rate(flags, labels),
                "lipschitz": lipschitz_ceiling(float(w), l_free, l_disagree),
                "reportable": report.reportable,
            }
        )
    return pd.DataFrame(rows)


def select_weights(
    df: pd.DataFrame,
    calibration_operator: str,
    grid: tuple[float, ...] = DEFAULT_GRID,
    alpha: float = 0.10,
    cluster_col: str = "sample_id",
    **kwargs,
) -> WeightSelection:
    """Apply D28's pre-registered rule to a bank's design split.

    Minimise flagged area; among points within one SE of the minimum, take the largest
    ``w_free``, because `L` falls as `w_free` rises (D41) and the detection difference is
    inside the noise.
    """
    curve = weight_curve(
        df, calibration_operator, grid=grid, alpha=alpha, cluster_col=cluster_col, **kwargs
    )
    usable = curve[np.isfinite(curve["flag_area"])]
    if usable.empty:
        raise ValueError("no point on the sweep produced a finite flagged area")

    design_clusters = df[df.get("split", pd.Series(dtype=str)).astype(str) == "design"]
    n_clusters = int(design_clusters[cluster_col].nunique()) if len(design_clusters) else 0

    best = usable.loc[usable["flag_area"].idxmin()]
    tolerance = float(best["se"]) if np.isfinite(best["se"]) else 0.0
    tied = usable[usable["flag_area"] <= float(best["flag_area"]) + tolerance]

    # Does the sweep carry information at all? Checked before the tie-break, because the
    # tie-break is for choosing among separated-but-tied points, not for standing in when
    # nothing is separated.
    if len(tied) > TIED_FRACTION_LIMIT * len(usable):
        row = usable.iloc[(usable["w_free"] - 0.5).abs().argmin()]
        return WeightSelection(
            w_free=float(row["w_free"]),
            w_disagree=float(row["w_disagree"]),
            flag_area=float(row["flag_area"]),
            se=float(row["se"]),
            lipschitz=float(row["lipschitz"]),
            n_design_clusters=n_clusters,
            tied_with=tuple(float(v) for v in sorted(tied["w_free"])),
            discriminated=False,
            curve=curve,
        )

    chosen = tied.loc[tied["w_free"].idxmax()]

    return WeightSelection(
        w_free=float(chosen["w_free"]),
        w_disagree=float(chosen["w_disagree"]),
        flag_area=float(chosen["flag_area"]),
        se=float(chosen["se"]),
        lipschitz=float(chosen["lipschitz"]),
        n_design_clusters=n_clusters,
        tied_with=tuple(float(v) for v in sorted(tied["w_free"])),
        discriminated=True,
        curve=curve,
    )
