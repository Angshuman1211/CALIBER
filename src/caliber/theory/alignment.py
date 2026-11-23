"""T3-C: the orthogonal-error approximation, re-checked where the prior recovers hidden content.

Every bank carries three null-space norms per region: ``s = s_free = ||P_N x_hat||`` (the
reconstruction's hidden content), ``c = free_content = ||P_N x||`` (the truth's) and
``err = err_free = ||P_N (x_hat - x)||``. They obey an exact identity (``PROJECT_STATUS.md``
Sec. 3d)::

    err^2 = s^2 + c^2 - 2 s c cos(theta).

Under the face prior ``cos theta ~ 0`` and ``err ~ hypot(s, c)``: that is the orthogonal-error
approximation, and it is what made every magnitude label a function of the score (D67). On
the medical priors ``cos theta`` is +0.60 (MRI, BraTS) and +0.88 (CT, LIDC), so any step that
used it has to be re-checked (T2.10 Sec. 11, ``PHASE4_WORK_PLAN.md`` T3-C).

**The exact split this module computes.** Write ``a = s cos(theta)`` for the part of the
reconstruction's hidden content that lies *along* the truth's, and ``o = s |sin(theta)|`` for
the part orthogonal to it. Then, with no approximation,

    s^2   = a^2 + o^2,
    err^2 = (a - c)^2 + o^2.

``o`` is invention in the project's sense (content in the null space that the truth does not
have); ``a - c`` is recovery error (hidden truth the prior under- or over-shot). The
orthogonal-error approximation is the statement ``a ~ 0``, i.e. that every hidden-content
error is either unrecovered truth (``|a - c| ~ c``) or invention. Its relative error is
``(hypot(s, c) - err) / err``, returned per region.

``cos theta`` is inverted from the identity (``scoring.attribution.content_alignment``), so the
split inherits that function's guard: a region whose three columns contradict the identity
(``|cos theta| > 1``, which on the CT banks is the solver's non-additivity, T2-A5/T2-A6) is
NaN and counted, not clipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..evaluation.gap import rank_correlation
from ..scoring.attribution import content_alignment


def decompose(s_free, free_content, err_free, min_content: float = 1e-3) -> pd.DataFrame:
    """Per region: ``cos``, ``aligned`` (``a``), ``invented`` (``o``), ``recovery_error``
    (``a - c``), ``invented_share`` (``o^2 / err^2``) and ``orth_rel_err``."""
    s = np.asarray(s_free, dtype=float)
    c = np.asarray(free_content, dtype=float)
    e = np.asarray(err_free, dtype=float)
    cos = content_alignment(s, c, e, min_content=min_content)
    sin = np.sqrt(np.clip(1.0 - cos**2, 0.0, 1.0))
    aligned, invented = s * cos, s * sin
    recovery = aligned - c
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(e > 0, invented**2 / e**2, np.nan)
        orth = np.where(e > 0, (np.hypot(s, c) - e) / e, np.nan)
    return pd.DataFrame({
        "cos": cos, "aligned": aligned, "invented": invented, "recovery_error": recovery,
        "invented_share": share, "orth_rel_err": orth,
    })


def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _threshold(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    k = int(np.floor(alpha * (scores.size + 1)))
    return float(scores[k - 1]) if k >= 1 else float("-inf")


def operator_alignment(
    df: pd.DataFrame,
    alpha: float = 0.10,
    marginal: float = 2.0,
    exclude_design: bool = True,
    eps: float | None = None,
) -> pd.DataFrame:
    """One row per operator: the alignment regime, and where the threshold's positives sit.

    ``df`` needs the three norms and, for the threshold columns, ``score`` and ``label`` (as
    ``derive_score_and_label`` returns them). ``tau`` is the split-conformal threshold on the
    ``calib`` rows (every non-design row if the bank has no split). The **marginal positives**
    are the positives scoring at or below the ``marginal * alpha`` quantile of positive
    scores: the ones the threshold is fitted on.

    With the label's ``eps`` given, two columns locate them against the **orthogonal edge**
    ``sqrt(max(eps^2 - c^2, 0))``. Since ``err^2 = s^2 + c^2 - 2sc cos(theta)``, a region with
    ``cos(theta) >= 0`` can be positive only if ``s`` clears that edge, and at
    ``cos(theta) = 0`` clearing it is also sufficient: the positives are then *exactly* the
    score law truncated at an edge set by the hidden truth alone. ``marginal_edge_gap`` is
    the median of ``(s - edge) / s`` over the marginal positives (near 0: the threshold sits on
    the edge); ``below_edge`` is the share of all positives under it, which only
    ``cos(theta) < 0`` allows; ``marginal_truth_driven`` is the share of marginal positives
    whose hidden truth exceeds ``eps`` (``c >= eps``). **That is "positive whatever the score"
    only at ``cos(theta) = 0``**; in general it means the region *has* a low-score positive
    branch (below).

    **The exact branches.** Holding the region's ``c`` and ``theta`` fixed, the identity makes
    it positive iff its ``s`` lies outside ``[s_lo, s_hi]``,
    ``s_lo/hi = c cos(theta) -/+ sqrt(eps^2 - c^2 sin^2(theta))``. So every positive is in one
    of three classes, reported as shares of the marginal positives:

    * ``marginal_score_free``: ``c |sin(theta)| >= eps``, positive at every ``s``;
    * ``marginal_lower_branch``: ``s <= s_lo``, which exists exactly when ``c > eps`` and
      ``cos(theta) > 0`` -- the prior recovered too little of a large hidden truth;
    * ``marginal_upper_branch``: ``s >= s_hi`` -- at ``cos(theta) = 0`` this is the orthogonal
      edge.

    ``marginal_boundary_slope`` is the median, over branch positives, of ``d s_lo / d c`` or
    ``d s_hi / d c``: how the boundary the marginal positive sits on moves as the operator hides
    more truth. ``d s_lo / d c = cos + c sin^2 / sqrt(eps^2 - c^2 sin^2) > 0`` always;
    ``d s_hi / d c = cos - c sin^2 / sqrt(...)``, negative at ``cos = 0``. H-T3C (T3.6 Sec. 11)
    reads the sign of the threshold's move off this. ``branch_inconsistent`` counts positives
    the identity says cannot be positive (float error only, if the columns are consistent).

    ``positive_score_truth_spearman`` is Spearman(score, ``c``) over the positives: how far the
    score of a positive moves with the hidden truth that made it positive. Under orthogonal
    fill the two are coupled only through shared region energy; under recovery the aligned
    part ``a = s cos(theta)`` couples them directly.
    """
    rows = []
    for op, g in df.groupby("operator_id", sort=True):
        body = g[g["split"] != "design"] if exclude_design and "split" in g else g
        dec = decompose(body["s_free"], body["free_content"], body["err_free"])
        s, c, e = (body[k].to_numpy(float) for k in ("s_free", "free_content", "err_free"))
        row = {
            "operator_id": op,
            "n": int(len(body)),
            "undefined": float(dec["cos"].isna().mean()),
            "cos_median": _median(dec["cos"]),
            "cos_q25": float(np.nanquantile(dec["cos"], 0.25)) if dec["cos"].notna().any()
            else float("nan"),
            "cos_q75": float(np.nanquantile(dec["cos"], 0.75)) if dec["cos"].notna().any()
            else float("nan"),
            "s_median": _median(s),
            "c_median": _median(c),
            "aligned_median": _median(dec["aligned"]),
            "invented_median": _median(dec["invented"]),
            "invented_energy_share": float(np.nansum(dec["invented"] ** 2)
                                           / np.nansum(dec["invented"] ** 2
                                                       + dec["recovery_error"] ** 2)),
            "orth_rel_err_median": _median(np.abs(dec["orth_rel_err"])),
            "identity_spearman": rank_correlation(e, np.hypot(s, c)),
        }
        if {"score", "label"} <= set(body.columns):
            lab = body["label"].to_numpy(bool)
            sc = body["score"].to_numpy(float)
            fit = (body["split"] == "calib").to_numpy() if "split" in body else np.ones(
                len(body), bool)
            if not fit.any():
                fit = np.ones(len(body), bool)
            pos_scores = sc[lab]
            row["pi"] = float(lab.mean())
            row["tau"] = _threshold(sc[fit & lab], alpha)
            if pos_scores.size:
                cut = float(np.quantile(pos_scores, min(1.0, marginal * alpha)))
                m = lab & (sc <= cut)
                row["marginal_cos_median"] = _median(dec["cos"].to_numpy()[m])
                row["marginal_c_over_s"] = _median(c[m] / np.maximum(s[m], 1e-12))
                row["marginal_invented_share"] = _median(dec["invented_share"].to_numpy()[m])
                if eps is not None and np.isfinite(eps):
                    edge = np.sqrt(np.maximum(eps**2 - c**2, 0.0))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel = np.where(s > 0, (s - edge) / s, np.nan)
                    row["marginal_edge_gap"] = _median(rel[m])
                    row["below_edge"] = float((s[lab] < edge[lab] - 1e-12).mean())
                    row["marginal_truth_driven"] = float((c[m] >= eps).mean())
                    row.update(_branches(s, c, dec["cos"].to_numpy(), eps, m, lab))
            if lab.sum() >= 3:
                row["positive_score_truth_spearman"] = rank_correlation(sc[lab], c[lab])
        rows.append(row)
    return pd.DataFrame(rows)


def _branches(s, c, cos, eps: float, marginal, positive) -> dict[str, float]:
    """Shares of the marginal positives in each exact branch, and the boundary slope."""
    sin2 = np.clip(1.0 - cos**2, 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(eps**2 - c**2 * sin2)  # NaN where c|sin| > eps: score-free
        s_lo, s_hi = c * cos - root, c * cos + root
        free = np.sqrt(sin2) * c >= eps
        lower = ~free & np.isfinite(root) & (s <= s_lo + 1e-12)
        upper = ~free & np.isfinite(root) & (s >= s_hi - 1e-12)
        slope = np.where(lower, cos + c * sin2 / root,
                         np.where(upper, cos - c * sin2 / root, np.nan))
    defined = np.isfinite(cos)
    m = marginal & defined
    pos = positive & defined
    inconsistent = pos & ~free & ~lower & ~upper
    n = max(int(m.sum()), 1)
    return {
        "marginal_score_free": float((free & m).sum() / n),
        "marginal_lower_branch": float((lower & m).sum() / n),
        "marginal_upper_branch": float((upper & m).sum() / n),
        "marginal_boundary_slope": _median(slope[m]),
        "branch_inconsistent": float(inconsistent.sum() / max(int(pos.sum()), 1)),
    }


def log_slope(x, y) -> float:
    """Least-squares slope of ``log y`` on ``log x`` over finite, positive pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if keep.sum() < 3 or np.ptp(np.log(x[keep])) == 0:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def elasticities(
    summary: pd.DataFrame,
    by: str = "c_median",
    columns: Sequence[str] = ("s_median", "aligned_median", "invented_median", "tau"),
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Elasticity of each column with respect to ``by`` across operators, pooled and by group.

    ``by = c_median`` uses the truth's own hidden content as the ill-posedness coordinate,
    which needs nothing from the operator and is defined across families (null-space
    *dimension* is not: four of the nine CT operators share 64,000). Then the hidden truth
    has elasticity 1 by construction and ``invented_median``'s elasticity is the model's
    ``beta``.
    """
    out = []
    scopes = [("pooled", np.ones(len(summary), bool))]
    if groups is not None:
        g = np.asarray(groups)
        scopes += [(str(k), g == k) for k in sorted(set(g))]
    for name, keep in scopes:
        part = summary[keep]
        for col in columns:
            if col not in part:
                continue
            out.append({"scope": name, "column": col, "n_operators": int(len(part)),
                        "elasticity": log_slope(part[by], part[col]),
                        "spearman": rank_correlation(part[by], part[col])})
    return pd.DataFrame(out)


def threshold_direction_model(
    kappa: float,
    beta: float,
    invention_scale: float = 1.0,
    lams: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
    alpha: float = 0.10,
    label_quantile: float = 0.75,
    n: int = 200_000,
    seed: int = 0,
) -> dict[str, object]:
    """How ``tau`` moves with ill-posedness in a one-line generative model of the split.

    Per region, a shared energy ``g`` (background regions are small in everything); the
    truth's hidden content ``c = lam * c0``; the reconstruction recovers a fraction ``kappa``
    of it (``a = kappa * c``, per-region noise sd 0.25, clipped to [0, 1.5]) and invents
    ``o = lam**beta * o0`` orthogonal to it. ``lam`` indexes ill-posedness; ``beta`` is how fast
    invention grows relative to the hidden truth. Score ``s = hypot(a, o)``, error
    ``err = hypot(a - c, o)``, one global ``eps`` (the ``label_quantile`` of err pooled over
    operators, D-061), and ``tau`` the split-conformal threshold among positives.

    **Scope: the spread-dominated regime.** The per-region spreads here are lognormal and
    wide, so ``tau`` is set by the bulk of the positive score law and moves with its scale.
    It omits the other regime, which the white-noise harness is in: when scores are nearly
    deterministic per region the orthogonal edge ``sqrt(eps^2 - c^2)`` binds and ``tau``
    falls as the hidden truth grows *whatever* ``beta`` is (T3.6 Part II). It is a model of
    one mechanism, not of any bank: its parameters are not fitted.
    """
    rng = np.random.default_rng(seed)
    g = rng.lognormal(0.0, 0.8, n)
    c0 = g * rng.lognormal(0.0, 0.3, n)
    o0 = invention_scale * g * rng.lognormal(0.0, 0.5, n)
    k = np.clip(kappa + rng.normal(0.0, 0.25, n), 0.0, 1.5) if kappa > 0 else np.zeros(n)
    cells, pooled = [], []
    for lam in lams:
        c, o = lam * c0, lam**beta * o0
        a = k * c
        s, err = np.hypot(a, o), np.hypot(a - c, o)
        cells.append((s, err, np.where(s > 0, a / s, 0.0)))
        pooled.append(err[: n // 10])
    eps = float(np.quantile(np.concatenate(pooled), label_quantile))
    taus = np.array([_threshold(s[err > eps], alpha) for s, err, _ in cells])
    cos = float(np.median(np.concatenate([cs for _, _, cs in cells])))
    return {
        "kappa": kappa, "beta": beta, "cos_median": cos, "lams": tuple(lams),
        "taus": tuple(float(t) for t in taus),
        "tau_elasticity": log_slope(np.asarray(lams), taus),
        "tau_direction": int(np.sign(rank_correlation(lams, taus))),
    }


# -- T3-D: the departure from T3.4's sqrt(null_frac) law, split exactly ---------------------

_KEYS = ["sample_id", "slice_idx", "region_id"]


def _quotient(r: float, r_nf: float) -> float:
    """``sqrt(r / r_nf)``: an energy ratio against the law's, as an amplitude quotient."""
    if np.isfinite(r) and np.isfinite(r_nf) and r_nf > 0 and r >= 0:
        return float(np.sqrt(r / r_nf))
    return float("nan")


def energy_departure(
    df: pd.DataFrame,
    reference: str,
    null_fraction: dict[str, float] | None = None,
    exclude_design: bool = True,
) -> pd.DataFrame:
    """How far each operator's hidden-content energy departs from the sqrt(null_frac) law.

    T3.4 Sec. 2 derived ``s_free ~ sqrt(null_frac)`` for white-noise fill, so the law predicts
    the energy ratio ``R_nf = nf(B) / nf(A)`` between an operator ``B`` and a ``reference``
    ``A``. Over the same regions at both operators (the bank is crossed), energies add exactly,
    ``E_s = E_a + E_o``, so the measured ratio splits with no approximation:

        R_s = C_A * R_a + (1 - C_A) * R_o,      C_A = E_a(A) / E_s(A),

    and the departure from the law splits with it:

        R_s - R_nf = C_A (R_a - R_nf) + (1 - C_A) (R_o - R_nf).

    The law can only ever have been about the invented part ``o`` (white fill is orthogonal
    to the truth by construction); the aligned part ``a`` is recovered truth and follows the
    truth's own hidden energy, whose ratio ``R_c`` is T3.4 Sec. 1's energy-weighted null
    fraction. So a real prior's departure is (i) its aligned share ``C_A`` times how far the
    truth's hidden energy is from the dimension count, plus (ii) its invented share times how
    far its invention is from white. ``quotient_*`` are the amplitude ratios
    ``sqrt(R_* / R_nf)``, the form T3.4 and T3.5 quote (1.0 = the law holds).
    """
    body = df[df["split"] != "design"] if exclude_design and "split" in df else df
    parts = {}
    for op, g in body.groupby("operator_id", sort=True):
        dec = decompose(g["s_free"], g["free_content"], g["err_free"])
        t = g[_KEYS].reset_index(drop=True).copy()
        t["a2"] = dec["aligned"].to_numpy() ** 2
        t["o2"] = dec["invented"].to_numpy() ** 2
        t["c2"] = g["free_content"].to_numpy(float) ** 2
        parts[op] = t.set_index(_KEYS)
    if reference not in parts:
        raise KeyError(f"reference {reference!r} is not in the bank; have {sorted(parts)}")
    fam = (df.drop_duplicates("operator_id").set_index("operator_id")["operator_family"]
           .astype(str).to_dict() if "operator_family" in df else {})
    ref = parts[reference]
    rows = []
    for op, cur in parts.items():
        both = ref.join(cur, lsuffix="_A", rsuffix="_B", how="inner")
        both = both[np.isfinite(both).all(axis=1)]
        ea, eo, ec = (float(both[f"{k}_A"].sum()) for k in ("a2", "o2", "c2"))
        ba, bo, bc = (float(both[f"{k}_B"].sum()) for k in ("a2", "o2", "c2"))
        es_a, es_b = ea + eo, ba + bo
        nan = float("nan")
        r_s = es_b / es_a if es_a > 0 else nan
        r_a = ba / ea if ea > 0 else nan
        r_o = bo / eo if eo > 0 else nan
        r_c = bc / ec if ec > 0 else nan
        c_ref = ea / es_a if es_a > 0 else nan
        nf_a = (null_fraction or {}).get(reference, nan)
        nf_b = (null_fraction or {}).get(op, nan)
        r_nf = nf_b / nf_a if nf_a and np.isfinite(nf_a) and nf_a > 0 else nan
        rows.append({
            "operator_id": op, "family": fam.get(op, "?"), "n_regions": int(len(both)),
            "null_fraction": nf_b, "R_nf": r_nf, "R_s": r_s, "R_a": r_a, "R_o": r_o,
            "R_c": r_c, "aligned_share_ref": c_ref,
            "aligned_share": ba / es_b if es_b > 0 else nan,
            "identity_residual": (r_s - (c_ref * r_a + (1 - c_ref) * r_o)
                                  if np.isfinite(r_a) and np.isfinite(r_o) else nan),
            "departure": r_s - r_nf,
            "aligned_contribution": c_ref * (r_a - r_nf),
            "invented_contribution": (1 - c_ref) * (r_o - r_nf),
            "quotient_s": _quotient(r_s, r_nf), "quotient_invented": _quotient(r_o, r_nf),
            "quotient_truth": _quotient(r_c, r_nf),
        })
    return pd.DataFrame(rows)
