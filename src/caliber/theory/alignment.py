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
) -> pd.DataFrame:
    """One row per operator: the alignment regime, and where the threshold's positives sit.

    ``df`` needs the three norms and, for the threshold columns, ``score`` and ``label`` (as
    ``derive_score_and_label`` returns them). ``tau`` is the split-conformal threshold on the
    ``calib`` rows (every non-design row if the bank has no split). The **marginal positives**
    are the positives scoring at or below the ``marginal * alpha`` quantile of positive
    scores: the ones the threshold is fitted on, whose score elasticity decides which way
    ``tau`` moves when the operator changes (``threshold_direction_model``).
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
        rows.append(row)
    return pd.DataFrame(rows)


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

    It is a model of the *mechanism*, not of any bank: its parameters are not fitted, and a
    bank is consistent with it only in the sense of falling in the same region of the
    ``(cos theta, beta)`` plane.
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
