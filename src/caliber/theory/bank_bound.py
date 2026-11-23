"""T3.6 / T3-A: the measured coverage gap against the T3.3 bound, per ordered pair, on a bank.

For one ordered operator pair ``(src, dst)`` of a crossed bank this computes the six rows the
Phase 4 plan asks of T3-A (``PHASE4_WORK_PLAN.md`` Sec. 5), side by side:

1. **The measured gap**, in both of Track 1's frozen definitions (``prereg.GAP_DEFINITION``
   and its sensitivity arm): ``matched_minus_transfer = cov(dst->dst) - cov(src->dst)`` and
   ``nominal_minus_transfer = (1 - alpha) - cov(src->dst)``. The interval is a cluster
   bootstrap that resamples patients on **both** sides and **refits both thresholds inside
   every replicate** (``prereg.REFIT_THRESHOLD_PER_REPLICATE``). The first version of this
   module held the threshold fixed and resampled the test side only, which is D73's defect;
   that interval is still computed (``gap_fixed_lo/hi``) so the size of the defect is on the
   record rather than asserted.
2. **d_K**, the Kolmogorov distance between the score laws given ``label = 1`` (the Step 1
   bound, T3.3 sec. 2b), with an interval from the same patient resample, the check
   ``gap_matched <= d_K + 1/(n+1)`` with ``n`` the calibration positives at ``dst`` (the
   ``1/(n+1)`` is how far a matched split-conformal cell can sit *above* nominal), and the
   **tightness ratio** ``gap / bound``. ``cdf_gap_at_tau`` is the signed CDF difference at
   the transferred threshold, which is what the gap estimates (D89's identity); its ratio to
   ``d_K`` says how much of the bound's slack is *where* the two laws separate most rather
   than sampling noise.
3. **W1** between the score laws, label-conditional and unconditional, with intervals.
4. **Coupled upper bounds on W1.** The bank scores the same (item, region) under both
   operators, so ``mean |s(x; dst) - s(x; src)|`` bounds the unconditional ``W1`` for that
   coupling (T3.3 sec. 3(i)). The label-conditional laws have no such direct coupling,
   because the label itself changes with the operator; splitting off the flips gives an exact
   bound for them too (``coupled_w1_cond_bound``). Both inequalities are theorems, so a
   violation is a defect in this code, not a finding.
5. **The operator distance** (``d_op`` and the signed growth, D19/D47) is attached to each
   row by the caller; the Spearman correlations across pairs live in
   :func:`distance_correlations`, pooled and per family cell, never pooled alone.
6. *Optional*: score-level total variation by histogram at several bin counts. By data
   processing it is at most the data-level TV in Barber et al.'s bound.

Plus the sec. 3b decomposition and the constants of the ``sqrt(d_op)`` rate, unchanged.

Conventions are the project's, not this module's: the label is whatever ``label_spec``
returns (D-061's absolute label by default); thresholds are ``split_threshold``'s
``floor(alpha (n + 1))``-th smallest calibration positive, fitted on ``calib`` and read on
``test`` (D-022); the cluster is the patient where the bank has one, else the item (D-027).
The score's component scales are frozen on one operator (D26/D27): per pair on ``src`` by
default, or on a fixed ``scale_operator`` for the whole table, which is how Track 2's T2-A3
cell tables are built and what makes ``cov_transfer`` here reproduce theirs to the digit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..calibration.conformal import split_threshold
from ..evaluation.bank import LabelSpec, derive_score_and_label
from ..scoring.score import ScoreWeights
from .stats import crossing_probability, joint_cdf_gap, ks_distance, w1_distance

KEYS = ["sample_id", "slice_idx", "region_id"]

#: Bin counts for row 6's histogram TV. Several, because a histogram TV is a function of the
#: binning: too coarse and it falls toward (or below) ``d_K``, too fine and sampling noise
#: inflates it. Bins are equal-mass (``histogram_tv``'s quantile edges).
HIST_BINS = (10, 20, 50, 100)

#: Default bootstrap replicates and interval level. 95% matches Track 2's cell tables
#: (``reliability.clustered_coverage``), so a T3-A interval and a T2-A3 interval can be read
#: side by side.
N_BOOT = 2000
LEVEL = 0.95


@dataclass(frozen=True)
class PairReport:
    src: str
    dst: str
    d_op: float
    signed_growth: float
    # row 1 -- the measured gap, registry split, refit-in-every-replicate intervals
    gap: float  # nominal_minus_transfer: (1 - alpha) - cov(src -> dst)
    gap_lo: float
    gap_hi: float
    gap_matched: float  # matched_minus_transfer: cov(dst -> dst) - cov(src -> dst)
    gap_matched_lo: float
    gap_matched_hi: float
    gap_fixed_lo: float  # threshold held fixed, test side only: D73's defect, for the record
    gap_fixed_hi: float
    cov_dst: float  # cov(src -> dst), the transfer cell
    cov_matched: float  # cov(dst -> dst)
    cov_src_heldout: float  # cov(src -> src)
    tau_src: float
    tau_dst: float
    # row 2 -- Kolmogorov distance and the bound check
    ks_cond: float
    ks_cond_lo: float
    ks_cond_hi: float
    ks_uncond: float
    cdf_gap_at_tau: float
    slack: float
    tightness: float
    tightness_lo: float
    tightness_hi: float
    # row 3 -- Wasserstein-1
    w1_cond: float
    w1_cond_lo: float
    w1_cond_hi: float
    w1_uncond: float
    w1_uncond_lo: float
    w1_uncond_hi: float
    # row 4 -- coupled upper bounds on W1
    w1_coupled: float
    w1_coupled_cond: float
    # row 6 -- histogram TV of the label-conditional laws, one value per entry of HIST_BINS
    tv_hist: tuple[float, ...]
    # T3.3 sec. 3b
    pi_src: float
    pi_dst: float
    p_flip: float
    crossing: float
    sup_joint_gap: float
    algebra_bound: float
    l_s: float
    l_eps: float
    m_s: float
    m_eps: float
    rate_bound: float
    # counts
    n_pos_src: int
    n_pos_dst: int
    n_pos_calib_src: int
    n_pos_calib_dst: int
    n_clusters: int  # test clusters
    n_clusters_calib: int
    n_boot: int
    level: float
    certifiable: bool  # src can certify alpha: its threshold is a real order statistic
    certifiable_dst: bool  # so can dst; otherwise the matched cell flags everything

    @property
    def evaluable(self) -> bool:
        """Both thresholds certifiable and ``dst`` has held-out positives to cover.

        A source below ``min_positives`` flags everything and a matched cell below it does
        too, so a gap there is ``0`` or ``-0.10`` by construction, not by measurement.
        """
        return bool(self.certifiable and self.certifiable_dst and self.n_pos_dst > 0
                    and np.isfinite(self.gap_matched))

    @property
    def holds_ks(self) -> bool:
        """``gap <= d_K^+``: T3.3 sec. 2b(a), on the nominal definition."""
        return bool(self.gap <= self.ks_cond + 1e-9)

    @property
    def holds_slack(self) -> bool:
        """``gap_matched <= d_K^+ + 1/(n+1)``: the plan's check, on Track 1's definition."""
        return bool(self.gap_matched <= self.ks_cond + self.slack + 1e-9)

    @property
    def gap_minus_ks(self) -> float:
        """Positive means the measured gap exceeds the Kolmogorov bound (T3.3 sec. 2b)."""
        return float(self.gap - self.ks_cond)

    @property
    def significant_violation(self) -> bool:
        """The whole nominal-gap interval sits above the whole ``d_K^+`` interval.

        A point violation inside the bootstrap spread is what the image-level penalty of
        sec. 2b(c) predicts at a small number of patients; only a separated pair of intervals
        would count against the chain.
        """
        return bool(np.isfinite(self.gap_lo) and np.isfinite(self.ks_cond_hi)
                    and self.gap_lo > self.ks_cond_hi)

    @property
    def significant_violation_matched(self) -> bool:
        """The same, for ``gap_matched`` against ``d_K^+ + 1/(n+1)``."""
        return bool(np.isfinite(self.gap_matched_lo) and np.isfinite(self.ks_cond_hi)
                    and self.gap_matched_lo > self.ks_cond_hi + self.slack)

    @property
    def location_ratio(self) -> float:
        """``|F_dst(tau) - F_src(tau)| / d_K``: the share of the sup reached at the threshold."""
        if not (np.isfinite(self.ks_cond) and self.ks_cond > 0):
            return float("nan")
        return float(abs(self.cdf_gap_at_tau) / self.ks_cond)

    @property
    def holds_coupled(self) -> bool:
        """``W1 <= E|s_dst - s_src|``. A theorem: False is a bug, not a result."""
        return bool(self.w1_uncond <= self.w1_coupled * (1 + 1e-9) + 1e-12)

    @property
    def holds_coupled_cond(self) -> bool:
        """``W1^+ <= coupled_w1_cond_bound``. A theorem: False is a bug, not a result."""
        if not np.isfinite(self.w1_coupled_cond):
            return True
        return bool(self.w1_cond <= self.w1_coupled_cond * (1 + 1e-9) + 1e-12)

    @property
    def holds_rate(self) -> bool | None:
        """``None`` when the rate bound is undefined (relative label, or a zero distance)."""
        if not np.isfinite(self.rate_bound):
            return None
        return bool(self.gap <= self.rate_bound + 1e-9)

    @property
    def interval_width_ratio(self) -> float:
        """Refit interval width over the fixed-threshold one: how much D73's defect hid."""
        fixed = self.gap_fixed_hi - self.gap_fixed_lo
        refit = self.gap_hi - self.gap_lo
        if not (np.isfinite(fixed) and np.isfinite(refit)) or fixed <= 0:
            return float("nan")
        return float(refit / fixed)

    def as_row(self) -> dict:
        row = asdict(self)
        tv = row.pop("tv_hist")
        for bins, value in zip(HIST_BINS, tv, strict=True):
            row[f"tv_h{bins}"] = value
        row.update(
            evaluable=self.evaluable, holds_ks=self.holds_ks, holds_slack=self.holds_slack,
            holds_rate=self.holds_rate, holds_coupled=self.holds_coupled,
            holds_coupled_cond=self.holds_coupled_cond, gap_minus_ks=self.gap_minus_ks,
            significant_violation=self.significant_violation,
            significant_violation_matched=self.significant_violation_matched,
            location_ratio=self.location_ratio,
            interval_width_ratio=self.interval_width_ratio,
        )
        return row


# -- small exact pieces ------------------------------------------------------------------


def _kde_density(sample: np.ndarray, at: np.ndarray, bw: float | None = None) -> np.ndarray:
    sample = np.asarray(sample, float)
    sd = float(np.std(sample)) or 1.0
    iqr = float(np.subtract(*np.percentile(sample, [75, 25]))) / 1.349
    spread = min(sd, iqr) if iqr > 0 else sd
    h = bw or 0.9 * spread * sample.size ** (-0.2)
    # subsample the sample for speed on big banks; the density estimate barely changes
    if sample.size > 20_000:
        sample = np.random.default_rng(0).choice(sample, 20_000, replace=False)
        h = bw or 0.9 * spread * sample.size ** (-0.2)
    out = np.empty(at.size)
    for i in range(0, at.size, 512):
        g = at[i : i + 512, None]
        out[i : i + 512] = np.exp(-0.5 * ((g - sample[None, :]) / h) ** 2).sum(1)
    return out / (sample.size * h * np.sqrt(2 * np.pi))


def max_density(sample: np.ndarray, n_grid: int = 400) -> float:
    """Sup of a Gaussian-KDE density of ``sample`` on a grid (T3.3's ``M``)."""
    sample = np.asarray(sample, float)
    grid = np.linspace(sample.min(), sample.max(), n_grid)
    return float(_kde_density(sample, grid).max())


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """``split_threshold``'s rule on positives only, without the sort: the
    ``floor(alpha (n + 1))``-th smallest, or ``-inf`` when that index is below 1.

    The bootstrap calls this thousands of times per pair, so it uses a partial sort; a test
    pins it to ``split_threshold`` exactly.
    """
    scores = np.asarray(scores, dtype=float)
    k = int(math.floor(alpha * (scores.size + 1)))
    if k < 1:
        return float("-inf")
    return float(np.partition(scores, k - 1)[k - 1])


def coupled_w1_cond_bound(sa, la, sb, lb) -> float:
    """An upper bound on ``W1(F_a^+, F_b^+)`` from the paired (item, region) coupling.

    The label-conditional laws are laws of *different* subpopulations, because the label
    flips with the operator, so the identical-item coupling does not have them as marginals
    and sec. 3(i)'s one-line bound does not apply. Write ``G_x(t) = P(S_x <= t, L_x = 1)``
    and ``pi_x = P(L_x = 1)``. Then, exactly (T3.3 sec. 3b step (a)),

        F_a^+ - F_b^+ = (G_a - G_b) / pi_a - F_b^+ (pi_a - pi_b) / pi_a,

    and with ``H`` any upper end of the support, ``int_{-inf}^{H} F_b^+ = H - E[S_b | L_b]``,
    while pointwise ``int |1{S_a<=t} L_a - 1{S_b<=t} L_b| dt`` is ``|S_a - S_b|`` when both
    labels are 1, ``H - S_a`` (or ``H - S_b``) when only one is, and 0 otherwise. So

        W1^+ <= ( E[L_a L_b |S_a - S_b|] + E[L_a (1-L_b) (H - S_a)] + E[(1-L_a) L_b (H - S_b)]
                  + |pi_a - pi_b| (H - E[S_b | L_b]) ) / pi_a,

    and the same with ``a`` and ``b`` exchanged; the smaller of the two is returned. Every term
    is a per-region movement, so it is the bridge sec. 3b needs from operators to the law the
    calibrator actually uses. A flipped region is charged up to the whole score range, which
    is why this bound can be loose where flips are common.
    """
    sa, sb = np.asarray(sa, float).ravel(), np.asarray(sb, float).ravel()
    la, lb = np.asarray(la, bool).ravel(), np.asarray(lb, bool).ravel()
    if sa.size == 0 or sa.size != sb.size or not la.any() or not lb.any():
        return float("nan")
    top = float(max(sa.max(), sb.max()))

    def one_way(s1, l1, s2, l2) -> float:
        p1, p2 = float(l1.mean()), float(l2.mean())
        moved = (np.abs(s1 - s2) * (l1 & l2)).mean()
        flips = ((top - s1) * (l1 & ~l2)).mean() + ((top - s2) * (~l1 & l2)).mean()
        return float((moved + flips + abs(p1 - p2) * (top - s2[l2].mean())) / p1)

    return min(one_way(sa, la, sb, lb), one_way(sb, lb, sa, la))


def histogram_tv(a, b, bins: int, edges: str = "quantile") -> float:
    """``0.5 * sum |p_a - p_b|`` over ``bins`` bins of the pooled sample.

    ``edges="quantile"`` (the default) puts the bin edges at equally spaced quantiles of the
    pooled sample, so every bin holds about the same mass. ``"uniform"`` is equal-width over
    the pooled range, which the first version used: on the LIDC bank a few large scores
    stretch that range so far that almost all the mass falls in one or two bins, and the
    "TV" came out *below* ``d_K`` at 58-72 of 72 pairs, which no total variation can be.
    Either way the value is TV over a partition, so by data processing it is at most the
    score-level TV (and that at most the data-level TV in Barber et al.'s bound).
    """
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    if edges == "uniform":
        cuts = np.linspace(lo, hi, int(bins) + 1)
    elif edges == "quantile":
        cuts = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, int(bins) + 1)))
        if cuts.size < 2:
            return 0.0
    else:
        raise ValueError(f"edges must be 'quantile' or 'uniform', got {edges!r}")
    pa = np.histogram(a, cuts)[0] / a.size
    pb = np.histogram(b, cuts)[0] / b.size
    return float(0.5 * np.abs(pa - pb).sum())


def _per_cluster(clusters: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, list]:
    """Split each array by cluster: ``(keys, [[array_1 per key], [array_2 per key], ...])``."""
    codes, keys = pd.factorize(pd.Series(clusters), sort=True)
    order = np.argsort(codes, kind="stable")
    cuts = np.searchsorted(codes[order], np.arange(1, len(keys)))
    return np.asarray(keys, dtype=object), [np.split(np.asarray(a)[order], cuts) for a in arrays]


def _caught(per_cluster_sorted: list[np.ndarray], taus: np.ndarray) -> np.ndarray:
    """``(len(taus), n_clusters)``: positives scoring ``>= tau`` per cluster, per threshold."""
    out = np.zeros((taus.size, len(per_cluster_sorted)), dtype=float)
    for c, scores in enumerate(per_cluster_sorted):
        if scores.size:
            out[:, c] = scores.size - np.searchsorted(scores, taus, side="left")
    return out


def _quantiles(x, level: float, min_n: int = 10) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < max(10, min_n):
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    return float(lo), float(hi)


# -- the pair ----------------------------------------------------------------------------


def pair_frame(df: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """The coupled table: one row per (item, slice, region) seen at both operators."""
    a = df[df["operator_id"] == src].set_index(KEYS)
    b = df[df["operator_id"] == dst].set_index(KEYS)
    both = a.index.intersection(b.index)
    if len(both) == 0:
        raise ValueError(f"{src!r} and {dst!r} share no (item, region): the bank is not crossed")
    cols = ["score", "label", "err_free", "split"]
    cols += [c for c in ("patient_id",) if c in a.columns]
    out = a.loc[both, cols].join(b.loc[both, ["score", "label", "err_free"]],
                                 lsuffix="_a", rsuffix="_b")
    return out.reset_index()


def score_bank(
    df: pd.DataFrame,
    label: LabelSpec,
    scale_operator: str,
    weights: ScoreWeights | None = None,
) -> pd.DataFrame:
    """``score`` and ``label`` columns for the whole bank, scales frozen on one operator."""
    return derive_score_and_label(df, scale_operator, weights=weights, **label.kwargs)


def _sides(pf: pd.DataFrame, cluster: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(calib, test)`` rows, patient-disjoint. A bank with no split column is halved by
    cluster, never by row, so no patient lands on both sides."""
    if {"calib", "test"} <= set(pf["split"]):
        fit, read = pf[pf["split"] == "calib"], pf[pf["split"] == "test"]
    else:
        keys = np.array(sorted(pf.loc[pf["split"] != "design", cluster].unique()), dtype=object)
        keys = np.random.default_rng(seed).permutation(keys)
        half = set(keys[: len(keys) // 2])
        body = pf[pf["split"] != "design"]
        inside = body[cluster].isin(half)
        fit, read = body[inside], body[~inside]
    shared = set(fit[cluster]) & set(read[cluster])
    if shared:
        raise ValueError(
            f"{len(shared)} cluster(s) on both the calibration and test sides "
            f"(e.g. {sorted(map(str, shared))[:3]}): a threshold scored on its own patients "
            "is in-sample"
        )
    return fit, read


def pair_report(
    df: pd.DataFrame,
    src: str,
    dst: str,
    label: LabelSpec,
    d_op: float,
    signed_growth: float = float("nan"),
    alpha: float = 0.10,
    weights: ScoreWeights | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = LEVEL,
    scale_operator: str | None = None,
    scored: pd.DataFrame | None = None,
    cluster: str | None = None,
) -> PairReport:
    """Measure and bound one ordered pair.

    ``df`` is a raw bank frame; the score and label are derived from it with scales frozen on
    ``scale_operator`` (``src`` when ``None``). A caller looping over many pairs with one
    scale operator passes the already-derived frame as ``scored`` to skip that step.
    ``cluster`` is the resampling unit: ``patient_id`` when the bank has it, else
    ``sample_id`` (on the CT banks one subject is one slice, so the two coincide).
    """
    if scored is None:
        rows = df if scale_operator not in (None, src, dst) else df[
            df["operator_id"].isin([src, dst])]
        scored = score_bank(rows, label, scale_operator or src, weights)
    pf = pair_frame(scored[scored["operator_id"].isin([src, dst])], src, dst)
    if cluster is None:
        cluster = "patient_id" if "patient_id" in pf.columns else "sample_id"
    if cluster not in pf.columns:
        raise KeyError(f"no {cluster!r} column to cluster on; have {sorted(pf.columns)}")
    nominal = 1.0 - alpha

    laws = pf[pf["split"] != "design"]  # the design side set the label's threshold
    sa, sb = laws["score_a"].to_numpy(float), laws["score_b"].to_numpy(float)
    la, lb = laws["label_a"].to_numpy(bool), laws["label_b"].to_numpy(bool)
    ea, eb = laws["err_free_a"].to_numpy(float), laws["err_free_b"].to_numpy(float)
    pi_a, pi_b = float(la.mean()), float(lb.mean())

    # --- row 1, point: fit on calib positives, read on test positives --------------------
    fit, read = _sides(pf, cluster, seed)
    res = split_threshold(fit["score_a"].to_numpy(), fit["label_a"].to_numpy(bool), alpha)
    tau_a = float(res.threshold)
    res_b = split_threshold(fit["score_b"].to_numpy(), fit["label_b"].to_numpy(bool), alpha)
    tau_b = float(res_b.threshold)
    pos_b = read.loc[read["label_b"].astype(bool), "score_b"].to_numpy(float)
    pos_a = read.loc[read["label_a"].astype(bool), "score_a"].to_numpy(float)
    nan = float("nan")
    cov_ab = float((pos_b >= tau_a).mean()) if pos_b.size else nan
    cov_bb = float((pos_b >= tau_b).mean()) if pos_b.size else nan
    cov_aa = float((pos_a >= tau_a).mean()) if pos_a.size else nan
    gap, gap_matched = nominal - cov_ab, cov_bb - cov_ab
    n_cal_a = int(fit["label_a"].astype(bool).sum())
    n_cal_b = int(fit["label_b"].astype(bool).sum())

    # --- rows 2-4 and 6, point: the laws over every non-design row -----------------------
    pos_la, pos_lb = sa[la], sb[lb]
    ks_c = ks_distance(pos_la, pos_lb)
    ks_u = ks_distance(sa, sb)
    w1_c, w1_u = w1_distance(pos_la, pos_lb), w1_distance(sa, sb)
    w1_coupled = float(np.abs(sa - sb).mean()) if sa.size else nan
    w1_coupled_c = coupled_w1_cond_bound(sa, la, sb, lb)
    if pos_la.size and pos_lb.size and np.isfinite(tau_a):
        cdf_at_tau = float((pos_lb < tau_a).mean() - (pos_la < tau_a).mean())
    else:
        cdf_at_tau = nan
    slack = 1.0 / (n_cal_b + 1)
    bound = ks_c + slack
    tightness = float(gap_matched / bound) if np.isfinite(bound) and bound > 0 else nan
    tv = tuple(histogram_tv(pos_la, pos_lb, k) for k in HIST_BINS)

    # --- sec. 3b pieces ---------------------------------------------------------------
    p_flip = float((la != lb).mean()) if la.size else nan
    cross = crossing_probability(sa, sb)
    supg = joint_cdf_gap(sa, la, sb, lb)
    algebra = (supg + abs(pi_a - pi_b)) / pi_a if pi_a > 0 else nan
    eps = label.kwargs.get("threshold", nan)
    l_s = float(w1_coupled / d_op) if d_op > 1e-12 else nan
    l_eps = float(np.abs(ea - eb).mean() / d_op) if d_op > 1e-12 else nan
    m_s = max_density(sa)
    absolute = np.isfinite(eps) and not label.kwargs.get("relative")
    m_eps = float(_kde_density(ea, np.array([eps]))[0]) if absolute else nan
    if np.isfinite(l_s) and np.isfinite(l_eps) and np.isfinite(m_eps) and pi_a > 0:
        rate = (4 * np.sqrt(2 * m_eps * l_eps) + 2 * np.sqrt(2 * m_s * l_s)) / pi_a * np.sqrt(d_op)
    else:
        rate = nan

    # --- the bootstrap: one patient resample per replicate, both sides, both thresholds --
    boot = _bootstrap(fit, read, laws, cluster, alpha, nominal, n_boot, seed, tau_a)
    # as reliability.clustered_coverage: fewer than a tenth of the replicates usable is no interval
    q = {k: _quantiles(r, level, n_boot // 10) for k, r in boot.items()}

    return PairReport(
        src=src, dst=dst, d_op=float(d_op), signed_growth=float(signed_growth),
        gap=float(gap), gap_lo=q["gap"][0], gap_hi=q["gap"][1],
        gap_matched=float(gap_matched), gap_matched_lo=q["gap_matched"][0],
        gap_matched_hi=q["gap_matched"][1],
        gap_fixed_lo=q["gap_fixed"][0], gap_fixed_hi=q["gap_fixed"][1],
        cov_dst=cov_ab, cov_matched=cov_bb, cov_src_heldout=cov_aa,
        tau_src=tau_a, tau_dst=tau_b,
        ks_cond=ks_c, ks_cond_lo=q["ks_cond"][0], ks_cond_hi=q["ks_cond"][1],
        ks_uncond=ks_u, cdf_gap_at_tau=cdf_at_tau, slack=float(slack),
        tightness=tightness, tightness_lo=q["tightness"][0], tightness_hi=q["tightness"][1],
        w1_cond=w1_c, w1_cond_lo=q["w1_cond"][0], w1_cond_hi=q["w1_cond"][1],
        w1_uncond=w1_u, w1_uncond_lo=q["w1_uncond"][0], w1_uncond_hi=q["w1_uncond"][1],
        w1_coupled=w1_coupled, w1_coupled_cond=w1_coupled_c, tv_hist=tv,
        pi_src=pi_a, pi_dst=pi_b, p_flip=p_flip, crossing=cross,
        sup_joint_gap=supg, algebra_bound=float(algebra),
        l_s=l_s, l_eps=l_eps, m_s=m_s, m_eps=m_eps, rate_bound=float(rate),
        n_pos_src=int(la.sum()), n_pos_dst=int(lb.sum()),
        n_pos_calib_src=n_cal_a, n_pos_calib_dst=n_cal_b,
        n_clusters=int(read[cluster].nunique()), n_clusters_calib=int(fit[cluster].nunique()),
        n_boot=int(n_boot), level=float(level), certifiable=bool(res.certifiable),
        certifiable_dst=bool(res_b.certifiable),
    )


def _bootstrap(
    fit: pd.DataFrame,
    read: pd.DataFrame,
    laws: pd.DataFrame,
    cluster: str,
    alpha: float,
    nominal: float,
    n_boot: int,
    seed: int,
    tau_fixed: float,
) -> dict[str, np.ndarray]:
    """Replicates of every row-1..3 quantity from **one** patient resample per replicate.

    Calibration patients and test patients are drawn separately (they are disjoint), both
    thresholds are refitted on the drawn calibration patients, both coverages are read on the
    drawn test patients, and the laws behind ``d_K`` and ``W1`` are the union of the two
    draws. One resample for everything is what makes the tightness ratio's interval mean
    something: its numerator and denominator move with the same patients.
    """
    nan_rows = {k: np.full(max(n_boot, 0), np.nan) for k in (
        "gap", "gap_matched", "gap_fixed", "ks_cond", "tightness", "w1_cond", "w1_uncond")}
    if n_boot <= 0 or fit.empty or read.empty:
        return nan_rows

    def pos(frame, side):
        s = frame[f"score_{side}"].to_numpy(float)
        lab = frame[f"label_{side}"].to_numpy(bool)
        return np.where(lab, s, np.nan)

    _, (cal_a, cal_b) = _per_cluster(fit[cluster].to_numpy(), pos(fit, "a"), pos(fit, "b"))
    cal_a = [np.sort(x[np.isfinite(x)]) for x in cal_a]
    cal_b = [np.sort(x[np.isfinite(x)]) for x in cal_b]
    _, (tst_b,) = _per_cluster(read[cluster].to_numpy(), pos(read, "b"))
    tst_b = [np.sort(x[np.isfinite(x)]) for x in tst_b]
    tst_n = np.array([x.size for x in tst_b], dtype=float)

    law_keys_cal = np.array(sorted(fit[cluster].unique()), dtype=object)
    law_keys_tst = np.array(sorted(read[cluster].unique()), dtype=object)
    law_body = laws[laws[cluster].isin(set(law_keys_cal) | set(law_keys_tst))]
    keys, (la_pos, lb_pos, la_all, lb_all) = _per_cluster(
        law_body[cluster].to_numpy(), pos(law_body, "a"), pos(law_body, "b"),
        law_body["score_a"].to_numpy(float), law_body["score_b"].to_numpy(float))
    where = {k: i for i, k in enumerate(keys)}
    la_pos = [x[np.isfinite(x)] for x in la_pos]
    lb_pos = [x[np.isfinite(x)] for x in lb_pos]
    cal_map = np.array([where[k] for k in law_keys_cal])
    tst_map = np.array([where[k] for k in law_keys_tst])

    rng = np.random.default_rng(seed)
    ci = rng.integers(0, len(cal_a), size=(n_boot, len(cal_a)))
    ti = rng.integers(0, len(tst_b), size=(n_boot, len(tst_b)))
    tau_a = np.array([conformal_threshold(np.concatenate([cal_a[c] for c in r]), alpha)
                      for r in ci])
    tau_b = np.array([conformal_threshold(np.concatenate([cal_b[c] for c in r]), alpha)
                      for r in ci])
    rows = np.arange(n_boot)[:, None]
    den = np.maximum(tst_n[ti].sum(axis=1), 1.0)
    cov_ab = _caught(tst_b, tau_a)[rows, ti].sum(axis=1) / den
    cov_bb = _caught(tst_b, tau_b)[rows, ti].sum(axis=1) / den
    cov_fixed = _caught(tst_b, np.array([tau_fixed]))[0][ti].sum(axis=1) / den
    empty = tst_n[ti].sum(axis=1) == 0
    cov_ab[empty] = cov_bb[empty] = cov_fixed[empty] = np.nan
    # a replicate whose calibration draw cannot certify alpha flags everything; like
    # reliability.clustered_coverage, it is dropped rather than counted as full coverage
    cov_ab[~np.isfinite(tau_a)] = np.nan
    cov_bb[~np.isfinite(tau_b)] = np.nan

    ks_c = np.full(n_boot, np.nan)
    w1_c = np.full(n_boot, np.nan)
    w1_u = np.full(n_boot, np.nan)
    n_cal_b = np.array([sum(cal_b[c].size for c in r) for r in ci], dtype=float)
    for b in range(n_boot):
        pick = np.concatenate([cal_map[ci[b]], tst_map[ti[b]]])
        pa = np.concatenate([la_pos[i] for i in pick])
        pb = np.concatenate([lb_pos[i] for i in pick])
        if pa.size and pb.size:
            ks_c[b] = ks_distance(pa, pb)
            w1_c[b] = w1_distance(pa, pb)
        w1_u[b] = w1_distance(np.concatenate([la_all[i] for i in pick]),
                              np.concatenate([lb_all[i] for i in pick]))
    gap_m = cov_bb - cov_ab
    with np.errstate(divide="ignore", invalid="ignore"):
        tight = gap_m / (ks_c + 1.0 / (n_cal_b + 1.0))
    return {
        "gap": nominal - cov_ab,
        "gap_matched": gap_m,
        "gap_fixed": nominal - cov_fixed,
        "ks_cond": ks_c,
        "tightness": tight,
        "w1_cond": w1_c,
        "w1_uncond": w1_u,
    }


# -- row 5: operator distance against the score-law distances, across pairs -------------


def _family_cell(src_family: str, dst_family: str) -> str:
    return f"{src_family}->{dst_family}"


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from ..evaluation.gap import rank_correlation

    return rank_correlation(x, y)


def distance_correlations(
    table: pd.DataFrame,
    family_of: Mapping[str, str],
    xs: Sequence[str] = ("d_op", "signed_growth"),
    ys: Sequence[str] = ("ks_cond", "ks_uncond", "w1_cond", "w1_uncond", "w1_coupled",
                         "gap_matched", "gap"),
    n_boot: int = 2000,
    level: float = LEVEL,
    seed: int = 0,
    correlate: Callable[[np.ndarray, np.ndarray], float] = _spearman,
) -> pd.DataFrame:
    """Row 5: Spearman of each operator distance against each law distance, by scope.

    Scopes are ``pooled`` and every family cell (``sparse_view->limited_angle`` and so on),
    always both: the project's rule is never to report a pooled statistic across operators
    alone, and on the LIDC bank the pooled and within-cell signs disagree (D93). The
    interval resamples **operators** (``prereg.GAP_CLUSTER_UNIT``): the ordered pairs are
    generated by a handful of operators, so resampling pairs would claim a sample size the
    design does not have. A drawn operator twice contributes its pairs twice; a pair of an
    operator with itself does not exist and is skipped.
    """
    t = table.copy()
    pairs = list(zip(t["src"], t["dst"], strict=True))
    t["cell"] = [_family_cell(family_of[a], family_of[b]) for a, b in pairs]
    ops = np.array(sorted(set(t["src"]) | set(t["dst"])), dtype=object)
    by_pair = {pair: i for i, pair in enumerate(pairs)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, ops.size, size=(n_boot, ops.size))
    replicate_rows = []
    for d in draws:
        picked = ops[d]
        idx = [by_pair[(a, b)] for i, a in enumerate(picked) for j, b in enumerate(picked)
               if i != j and a != b and (a, b) in by_pair]
        replicate_rows.append(np.asarray(idx, dtype=int))

    out = []
    scopes = ["pooled"] + sorted(t["cell"].unique())
    for x in xs:
        for y in ys:
            if x not in t.columns or y not in t.columns:
                continue
            xv, yv = t[x].to_numpy(float), t[y].to_numpy(float)
            for scope in scopes:
                keep = np.ones(len(t), bool) if scope == "pooled" else (t["cell"] == scope)
                keep = np.asarray(keep)
                rho = correlate(xv[keep], yv[keep])
                reps = []
                for idx in replicate_rows:
                    sel = idx[keep[idx]] if idx.size else idx
                    if sel.size >= 3:
                        reps.append(correlate(xv[sel], yv[sel]))
                lo, hi = _quantiles(reps, level)
                out.append({"x": x, "y": y, "scope": scope, "rho": rho,
                            "n_pairs": int(np.isfinite(xv[keep] + yv[keep]).sum()),
                            "lo": lo, "hi": hi})
    res = pd.DataFrame(out)
    cells = res[res["scope"] != "pooled"]

    def same_sign(r: pd.Series):
        """True/False across the cells with a finite rho; None when fewer than two have one,
        because agreement between one cell and itself is not evidence of anything."""
        r = r.dropna()
        if len(r) < 2:
            return None
        return bool(np.all(r > 0) or np.all(r < 0))

    agree = cells.groupby(["x", "y"])["rho"].agg(same_sign).rename("cells_agree_in_sign")
    return res.merge(agree.reset_index(), on=["x", "y"], how="left")
