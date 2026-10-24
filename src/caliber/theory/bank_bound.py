"""T3.6: the corrected Step 1 chain and the label-conditional link, measured on a score bank.

For one ordered operator pair ``(src, dst)`` of a crossed bank this reports, side by side:

**the measurement** -- calibrate on ``src``'s hallucinated regions, deploy on ``dst``'s, and read
off the under-coverage gap ``(1 - alpha) - coverage_dst`` with a patient-level bootstrap;

**the bound** (T3.3 sec. 2b, 3b) -- the Kolmogorov distance ``d_K^+`` between the score laws
given ``label = 1`` (``gap <= d_K^+``), its decomposition into label flips and crossings (the
bank is crossed, so the coupling §3b needs is *observed*, not assumed), and the constants of the
``sqrt(d_op)`` rate: ``L_S``, ``L_eps``, ``M_S``, ``M_eps`` and ``pi``.

Everything follows the project's conventions rather than inventing its own: the score is T1.1's
standardised affine combination with scales frozen on the **source** operator (D27); the label is
whatever ``label_spec`` returns (D-061's absolute label by default); thresholds are fitted on the
``calib`` side and read on the ``test`` side, patient-disjoint (D-022); the cluster is the patient
where the bank carries one, else the item (D-027).

The Kolmogorov and decomposition quantities use every non-design row of both operators (more
data, and they describe laws, not a fitted threshold); the measured gap uses the held-out split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..calibration.conformal import split_threshold
from ..evaluation.bank import LabelSpec, derive_score_and_label
from ..scoring.score import ScoreWeights
from .stats import crossing_probability, joint_cdf_gap, ks_distance, w1_distance

KEYS = ["sample_id", "slice_idx", "region_id"]


@dataclass(frozen=True)
class PairReport:
    src: str
    dst: str
    d_op: float
    signed_growth: float
    gap: float
    gap_lo: float
    gap_hi: float
    cov_dst: float
    cov_src_heldout: float
    ks_cond: float
    ks_cond_lo: float
    ks_cond_hi: float
    ks_uncond: float
    w1_cond: float
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
    n_pos_src: int
    n_pos_dst: int
    n_clusters: int
    certifiable: bool

    @property
    def holds_ks(self) -> bool:
        return bool(self.gap <= self.ks_cond + 1e-9)

    @property
    def holds_rate(self) -> bool:
        return bool(self.gap <= self.rate_bound + 1e-9)

    def as_row(self) -> dict:
        row = asdict(self)
        row.update(holds_ks=self.holds_ks, holds_rate=self.holds_rate)
        return row


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


def _coverage(flags: np.ndarray, labels: np.ndarray) -> float:
    n = int(labels.sum())
    return float((flags & labels).sum() / n) if n else float("nan")


def pair_report(
    df: pd.DataFrame,
    src: str,
    dst: str,
    label: LabelSpec,
    d_op: float,
    signed_growth: float = float("nan"),
    alpha: float = 0.10,
    weights: ScoreWeights | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> PairReport:
    """Measure and bound one ordered pair. ``df`` is a bank frame (see module docstring)."""
    pair_rows = df[df["operator_id"].isin([src, dst])]
    d = derive_score_and_label(pair_rows, src, weights=weights, **label.kwargs)
    pf = pair_frame(d, src, dst)
    cluster = "patient_id" if "patient_id" in pf.columns else "sample_id"

    keep = pf["split"] != "design"  # the design side set the label's threshold; keep it out
    laws = pf[keep]
    sa, sb = laws["score_a"].to_numpy(), laws["score_b"].to_numpy()
    la, lb = laws["label_a"].to_numpy(bool), laws["label_b"].to_numpy(bool)
    ea, eb = laws["err_free_a"].to_numpy(), laws["err_free_b"].to_numpy()
    pi_a, pi_b = float(la.mean()), float(lb.mean())

    # --- measurement: fit on src calib positives, read on dst test positives -------------
    has_splits = {"calib", "test"} <= set(pf["split"])
    fit = pf[pf["split"] == "calib"] if has_splits else pf.sample(frac=0.5, random_state=seed)
    read = pf[pf["split"] == "test"] if has_splits else pf.drop(fit.index)
    res = split_threshold(fit["score_a"].to_numpy(), fit["label_a"].to_numpy(bool), alpha)
    tau = res.threshold
    flags_b = read["score_b"].to_numpy() >= tau
    flags_a = read["score_a"].to_numpy() >= tau
    lab_b, lab_a = read["label_b"].to_numpy(bool), read["label_a"].to_numpy(bool)
    cov_dst, cov_src = _coverage(flags_b, lab_b), _coverage(flags_a, lab_a)
    gap = (1 - alpha) - cov_dst

    # --- bound pieces -------------------------------------------------------------------
    ks_c = ks_distance(sa[la], sb[lb])
    ks_u = ks_distance(sa, sb)
    w1_c = w1_distance(sa[la], sb[lb])
    p_flip = float((la != lb).mean())
    cross = crossing_probability(sa, sb)
    supg = joint_cdf_gap(sa, la, sb, lb)
    algebra = (supg + abs(pi_a - pi_b)) / pi_a if pi_a > 0 else float("nan")

    eps = label.kwargs.get("threshold", float("nan"))
    l_s = float(np.abs(sa - sb).mean() / d_op) if d_op > 1e-12 else float("nan")
    l_eps = float(np.abs(ea - eb).mean() / d_op) if d_op > 1e-12 else float("nan")
    m_s = max_density(sa)
    absolute = np.isfinite(eps) and not label.kwargs.get("relative")
    m_eps = float(_kde_density(ea, np.array([eps]))[0]) if absolute else float("nan")
    if np.isfinite(l_s) and np.isfinite(l_eps) and np.isfinite(m_eps) and pi_a > 0:
        rate = (4 * np.sqrt(2 * m_eps * l_eps) + 2 * np.sqrt(2 * m_s * l_s)) / pi_a * np.sqrt(d_op)
    else:
        rate = float("nan")

    # --- patient-level bootstrap of the gap and of d_K^+ --------------------------------
    rng = np.random.default_rng(seed)
    clusters = read[cluster].to_numpy()
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    lc = laws[cluster].to_numpy()
    uniq_l = np.unique(lc)
    idx_l = {c: np.flatnonzero(lc == c) for c in uniq_l}
    gaps, kss = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx_by[c] for c in pick])
        gaps.append((1 - alpha) - _coverage(flags_b[ii], lab_b[ii]))
        pick = rng.choice(uniq_l, len(uniq_l))
        jj = np.concatenate([idx_l[c] for c in pick])
        if la[jj].any() and lb[jj].any():
            kss.append(ks_distance(sa[jj][la[jj]], sb[jj][lb[jj]]))
    g_lo, g_hi = np.nanquantile(gaps, [0.05, 0.95])
    k_lo, k_hi = np.quantile(kss, [0.05, 0.95]) if kss else (float("nan"),) * 2

    return PairReport(
        src=src, dst=dst, d_op=d_op, signed_growth=signed_growth,
        gap=float(gap), gap_lo=float(g_lo), gap_hi=float(g_hi),
        cov_dst=cov_dst, cov_src_heldout=cov_src,
        ks_cond=ks_c, ks_cond_lo=float(k_lo), ks_cond_hi=float(k_hi), ks_uncond=ks_u,
        w1_cond=w1_c, pi_src=pi_a, pi_dst=pi_b, p_flip=p_flip, crossing=cross,
        sup_joint_gap=supg, algebra_bound=float(algebra),
        l_s=l_s, l_eps=l_eps, m_s=m_s, m_eps=m_eps, rate_bound=float(rate),
        n_pos_src=int(la.sum()), n_pos_dst=int(lb.sum()), n_clusters=int(len(uniq)),
        certifiable=bool(res.certifiable),
    )
