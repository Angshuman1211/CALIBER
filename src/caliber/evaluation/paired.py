"""The coverage gap as a paired within-patient difference (T1.4).

**Track 1 owns this module** — it fixes the *estimand*, which is the protocol's call
(T1.4, statistical-validation methodology), not the harness's. It lives beside Track 2's
T2.4 metrics because it is the same kind of object and reads the same bank.

## Why this exists

P-23 measured per-operator coverage intervals on the Week 3 fastMRI bank and found them
useless: ±0.125 at best, against the ±0.10 that would separate "coverage restored" from
"coverage failed". Its Week 4 update concluded that fastMRI ``singlecoil_val`` "cannot
support a per-operator coverage claim at any grid size", and handed Track 1 four options.

The conclusion is right about **per-operator coverage** and wrong about the **project's
claim**, for two reasons that are properties of the experimental design rather than of the
data.

**1. The design is fully crossed, so the patient budget is not divided.** This is a
retrospective undersampling study (Sec. 13): one k-space volume, every mask applied to it.
Verified on both banks in the repository — every operator sees every item. P-23's grid
table divides ~40 test patients by ``k`` operators and concludes a six-operator grid gives
±0.305; in the crossed design each operator sees all ~40, so the per-operator half-width
stays at ~0.125 no matter how many operators the grid holds. :func:`is_crossed` checks
this on a bank rather than assuming it, because a future design that samples different
patients per operator would make the division correct again.

**2. The headline estimand is a difference, and the pairing cancels the patient.** The
central figure is coverage *gap* against operator distance, not k coverages side by side.
Because the same patient is reconstructed at both operators, that gap is a **within-patient
paired difference**:

    Var(cov_A − cov_B) = Var(cov_A) + Var(cov_B) − 2·Cov(cov_A, cov_B)

and the dominant term in each variance is how hard that patient's anatomy is — which is
*shared* between the two operators and cancels in the difference. Estimating the gap by
subtracting two independently-bootstrapped coverages throws that cancellation away and
reports an interval far wider than the design supports.

:func:`paired_coverage_gap` resamples clusters **once** per bootstrap replicate and
evaluates both operators on the same resampled clusters, so the covariance is retained. It
reports the unpaired standard error alongside, and the ratio between them, so the gain is
visible and never assumed — on a bank where patient effects do not in fact correlate, the
ratio comes back at 1 and the honest conclusion is that pairing bought nothing.

## What this does not fix

It does not make a *single operator's* coverage better determined; that really is capped by
96 patients, and P-23's first row stands. So per-operator coverage is reported with its
interval, as a secondary descriptive quantity, and the claim rests on the gap. Nor does it
change flag area, which was already well determined because it is a fraction of every
region rather than of the few positive ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Bootstrap replicates. 2000 matches Track 2's `cluster_bootstrap_coverage`, so the two
#: intervals in one table are computed at the same resolution.
DEFAULT_N_BOOT = 2000


def is_crossed(
    df: pd.DataFrame, cluster_col: str = "sample_id", operator_col: str = "operator_id"
) -> dict[str, object]:
    """Does every operator see every cluster? The premise the pairing rests on.

    A crossed design is what retrospective undersampling produces and what makes the
    patient budget *shared* across operators rather than divided among them. Checked
    rather than assumed: a design that reconstructed different patients at different
    operators would be un-crossed, the pairing would be unavailable, and P-23's division
    would be the correct arithmetic after all.
    """
    per_operator = {
        str(op): set(frame[cluster_col].unique())
        for op, frame in df.groupby(operator_col, observed=True)
    }
    if not per_operator:
        return {"crossed": False, "n_operators": 0, "n_shared": 0, "reason": "no operators"}
    shared = set.intersection(*per_operator.values())
    sizes = {op: len(s) for op, s in per_operator.items()}
    crossed = all(n == len(shared) for n in sizes.values()) and len(shared) > 0
    return {
        "crossed": bool(crossed),
        "n_operators": len(per_operator),
        "n_shared": len(shared),
        "clusters_per_operator": sizes,
        "reason": (
            "every operator sees every cluster"
            if crossed
            else "operators see different cluster sets — pairing is unavailable"
        ),
    }


def _per_cluster_counts(
    frame: pd.DataFrame, cluster_col: str, flag_col: str, label_col: str, keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Caught positives and total positives per cluster, aligned to ``keys``."""
    labels = frame[label_col].to_numpy().astype(bool)
    flags = frame[flag_col].to_numpy().astype(bool)
    clusters = frame[cluster_col].to_numpy()
    caught = np.zeros(len(keys), dtype=float)
    positives = np.zeros(len(keys), dtype=float)
    index = {k: i for i, k in enumerate(keys)}
    for cluster, lab, flag in zip(clusters, labels, flags, strict=True):
        i = index.get(cluster)
        if i is None:
            continue
        if lab:
            positives[i] += 1.0
            if flag:
                caught[i] += 1.0
    return caught, positives


@dataclass(frozen=True)
class PairedGap:
    """A coverage gap with the interval the paired design actually supports."""

    calibration_operator: str
    deployment_operator: str
    coverage_calibration: float
    coverage_deployment: float
    gap: float
    lo: float
    hi: float
    se_paired: float
    se_unpaired: float
    n_clusters: int
    n_positive_calibration: int
    n_positive_deployment: int

    @property
    def variance_reduction(self) -> float:
        """How many times narrower the paired interval is. 1.0 means pairing bought nothing."""
        if not np.isfinite(self.se_paired) or self.se_paired <= 0:
            return float("nan")
        return float(self.se_unpaired / self.se_paired)

    @property
    def excludes_zero(self) -> bool:
        """Is the gap distinguishable from 'no coverage loss' at this interval?"""
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi) and (self.lo > 0 or self.hi < 0))

    def as_row(self) -> dict[str, object]:
        return {
            "deployment_operator": self.deployment_operator,
            "coverage_calib": self.coverage_calibration,
            "coverage_deploy": self.coverage_deployment,
            "gap": self.gap,
            "lo": self.lo,
            "hi": self.hi,
            "se_paired": self.se_paired,
            "se_unpaired": self.se_unpaired,
            "narrower_by": self.variance_reduction,
            "excludes_zero": self.excludes_zero,
            "n_clusters": self.n_clusters,
        }


def paired_coverage_gap(
    df: pd.DataFrame,
    calibration_operator: str,
    deployment_operator: str,
    cluster_col: str = "sample_id",
    operator_col: str = "operator_id",
    flag_col: str = "flag",
    label_col: str = "label",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedGap:
    """Coverage gap between two operators, bootstrapped over shared clusters.

    The one line that matters: each bootstrap replicate draws **one** set of clusters and
    scores *both* operators on it, so a patient that is hard at the calibration operator is
    the same hard patient at the deployment operator and its difficulty cancels out of the
    difference. ``se_unpaired`` is the same computation with the two operators resampled
    independently, reported so the gain is measured rather than claimed.

    Coverage is the ratio estimator `caught / positives` pooled over the resampled
    clusters, matching Track 2's `cluster_bootstrap_coverage` so the two agree on the
    marginal quantities.
    """
    left = df[df[operator_col].astype(str) == str(calibration_operator)]
    right = df[df[operator_col].astype(str) == str(deployment_operator)]
    if left.empty or right.empty:
        raise KeyError(
            f"bank has no rows for {calibration_operator!r} or {deployment_operator!r}"
        )

    keys = np.array(
        sorted(set(left[cluster_col].unique()) & set(right[cluster_col].unique())),
        dtype=object,
    )
    if keys.size == 0:
        raise ValueError(
            "the two operators share no cluster, so there is nothing to pair — check "
            "is_crossed() before calling this"
        )

    caught_l, pos_l = _per_cluster_counts(left, cluster_col, flag_col, label_col, keys)
    caught_r, pos_r = _per_cluster_counts(right, cluster_col, flag_col, label_col, keys)
    if pos_l.sum() == 0 or pos_r.sum() == 0:
        return PairedGap(
            str(calibration_operator), str(deployment_operator),
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), int(keys.size),
            int(pos_l.sum()), int(pos_r.sum()),
        )

    def ratio(caught: np.ndarray, positives: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return caught[idx].sum(axis=1) / np.maximum(positives[idx].sum(axis=1), 1.0)

    rng = np.random.default_rng(seed)
    shared_idx = rng.integers(0, keys.size, size=(n_boot, keys.size))
    paired = ratio(caught_l, pos_l, shared_idx) - ratio(caught_r, pos_r, shared_idx)

    # The same estimator with the pairing deliberately broken, as the comparison.
    rng_u = np.random.default_rng(seed + 1)
    idx_a = rng_u.integers(0, keys.size, size=(n_boot, keys.size))
    idx_b = rng_u.integers(0, keys.size, size=(n_boot, keys.size))
    unpaired = ratio(caught_l, pos_l, idx_a) - ratio(caught_r, pos_r, idx_b)

    lo, hi = np.quantile(paired, [alpha / 2, 1 - alpha / 2])
    cov_l = float(caught_l.sum() / pos_l.sum())
    cov_r = float(caught_r.sum() / pos_r.sum())
    return PairedGap(
        calibration_operator=str(calibration_operator),
        deployment_operator=str(deployment_operator),
        coverage_calibration=cov_l,
        coverage_deployment=cov_r,
        gap=cov_l - cov_r,
        lo=float(lo),
        hi=float(hi),
        se_paired=float(paired.std(ddof=1)),
        se_unpaired=float(unpaired.std(ddof=1)),
        n_clusters=int(keys.size),
        n_positive_calibration=int(pos_l.sum()),
        n_positive_deployment=int(pos_r.sum()),
    )


def paired_gap_curve(
    df: pd.DataFrame,
    calibration_operator: str,
    cluster_col: str = "sample_id",
    operator_col: str = "operator_id",
    flag_col: str = "flag",
    label_col: str = "label",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """One paired gap per deployment operator — the headline figure's y-axis.

    The calibration operator appears as a row with gap 0 by construction, kept so the
    figure has its origin and so a reader can see it was not omitted.
    """
    rows = []
    for op in sorted(df[operator_col].astype(str).unique()):
        gap = paired_coverage_gap(
            df, calibration_operator, op, cluster_col=cluster_col, operator_col=operator_col,
            flag_col=flag_col, label_col=label_col, n_boot=n_boot, alpha=alpha, seed=seed,
        )
        rows.append(gap.as_row())
    return pd.DataFrame(rows)


def paired_method_gap(
    df: pd.DataFrame,
    flag_col_a: str,
    flag_col_b: str,
    cluster_col: str = "sample_id",
    label_col: str = "label",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedGap:
    """Coverage difference between two *methods* on the same rows — the tightest contrast.

    "Group-conditional calibration restores the coverage naive transfer loses" is a
    statement about two thresholds applied to **one** set of reconstructions. Nothing
    varies between the two arms except the threshold: same patients, same operator, same
    scores, same labels. Both the patient's difficulty *and* the operator's difficulty
    cancel, where :func:`paired_coverage_gap` cancels only the patient.

    So this is the best-determined form of the project's central claim, and it is the one
    that should carry it where the question is whether the remedy works. Use
    :func:`paired_coverage_gap` where the question is how coverage moves with the operator
    — that one needs two operators and cannot be collapsed to this.

    ``flag_col_a`` and ``flag_col_b`` are two boolean columns on the same frame, e.g. the
    output of :func:`caliber.calibration.apply_flags` under two calibrations.
    """
    for col in (flag_col_a, flag_col_b, label_col, cluster_col):
        if col not in df.columns:
            raise KeyError(f"frame has no {col!r} column")

    keys = np.array(sorted(df[cluster_col].unique()), dtype=object)
    caught_a, positives = _per_cluster_counts(df, cluster_col, flag_col_a, label_col, keys)
    caught_b, _ = _per_cluster_counts(df, cluster_col, flag_col_b, label_col, keys)
    if positives.sum() == 0:
        return PairedGap(
            flag_col_a, flag_col_b, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"), int(keys.size), 0, 0,
        )

    def ratio(caught: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return caught[idx].sum(axis=1) / np.maximum(positives[idx].sum(axis=1), 1.0)

    rng = np.random.default_rng(seed)
    shared = rng.integers(0, keys.size, size=(n_boot, keys.size))
    paired = ratio(caught_a, shared) - ratio(caught_b, shared)

    rng_u = np.random.default_rng(seed + 1)
    unpaired = ratio(caught_a, rng_u.integers(0, keys.size, size=(n_boot, keys.size))) - ratio(
        caught_b, rng_u.integers(0, keys.size, size=(n_boot, keys.size))
    )

    lo, hi = np.quantile(paired, [alpha / 2, 1 - alpha / 2])
    total = positives.sum()
    return PairedGap(
        calibration_operator=flag_col_a,
        deployment_operator=flag_col_b,
        coverage_calibration=float(caught_a.sum() / total),
        coverage_deployment=float(caught_b.sum() / total),
        gap=float((caught_a.sum() - caught_b.sum()) / total),
        lo=float(lo),
        hi=float(hi),
        se_paired=float(paired.std(ddof=1)),
        se_unpaired=float(unpaired.std(ddof=1)),
        n_clusters=int(keys.size),
        n_positive_calibration=int(total),
        n_positive_deployment=int(total),
    )


def clusters_for_paired_precision(
    se_paired: float, n_observed: int, target_half_width: float, z: float = 1.96
) -> int:
    """Clusters needed for a ``±target_half_width`` interval **on the gap**.

    Same ``1/sqrt(n)`` extrapolation as Track 2's `clusters_for_precision`, applied to the
    paired standard error instead of a single operator's. The distinction is the whole
    point: sizing the study against the per-operator SE asks for a precision the headline
    claim never needed.
    """
    if not np.isfinite(se_paired) or se_paired <= 0 or target_half_width <= 0:
        raise ValueError("need a positive paired SE and target half-width")
    return int(np.ceil(n_observed * (se_paired / (target_half_width / z)) ** 2))
