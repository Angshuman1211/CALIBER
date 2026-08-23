"""Evaluation harness — the six metrics of Blueprint Sec. 15.2 (task T2.4).

| Blueprint metric                          | Function                    |
|-------------------------------------------|-----------------------------|
| Empirical coverage against nominal level  | ``empirical_coverage``      |
| Flag area at fixed reliability            | ``flag_area``               |
| Region-level false-discovery rate & power | ``region_fdr`` / ``region_power`` |
| Coverage gap against operator distance    | ``coverage_curve``          |
| Effective sample size under weighting     | ``effective_sample_size``   |
| Measured gap vs derived bound             | ``bound_agreement``         |

Vocabulary, fixed here so all three tracks report the same number: the
controlled risk is the **region-level false-negative (miss) rate** — the
fraction of genuinely hallucinated regions the flag fails to raise (O2, Week
2). "Coverage" is one minus that miss rate. A method at nominal level 0.9 is
promising to catch 90% of hallucinated regions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_bool(x) -> np.ndarray:
    return np.asarray(x).astype(bool).ravel()


# -- core rates ---------------------------------------------------------------


def miss_rate(flags, labels) -> float:
    """Region-level false-negative rate: hallucinated regions that were not flagged."""
    flags, labels = _as_bool(flags), _as_bool(labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    return float((~flags & labels).sum() / n_pos)


def empirical_coverage(flags, labels) -> float:
    """One minus the miss rate — the quantity the guarantee is about."""
    return 1.0 - miss_rate(flags, labels)


def flag_area(flags) -> float:
    """Fraction of regions flagged. A valid but over-conservative flag is useless."""
    flags = _as_bool(flags)
    return float(flags.mean()) if flags.size else float("nan")


def region_fdr(flags, labels) -> float:
    """Fraction of flagged regions that were not actually hallucinated."""
    flags, labels = _as_bool(flags), _as_bool(labels)
    n_flagged = int(flags.sum())
    if n_flagged == 0:
        return float("nan")
    return float((flags & ~labels).sum() / n_flagged)


def region_power(flags, labels) -> float:
    """Fraction of hallucinated regions correctly flagged (= empirical coverage)."""
    return empirical_coverage(flags, labels)


def effective_sample_size(weights) -> float:
    """``(sum w)^2 / sum w^2`` — the honest size of a reweighted calibration set.

    Blueprint Sec. 9, Tool 2: reweighting reduces the usable sample size, and
    the analysis tracks that cost explicitly rather than hiding it. Track 3's
    L3 turns this into a finite-sample deficit.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0 or not np.isfinite(w).all() or w.sum() <= 0:
        return float("nan")
    return float(w.sum() ** 2 / np.square(w).sum())


def coverage_gap(nominal: float, empirical: float) -> float:
    """Reliability promised minus reliability delivered. Positive means under-coverage."""
    return float(nominal - empirical)


# -- aggregation over a score bank --------------------------------------------


def evaluate_groups(
    df: pd.DataFrame,
    flag_col: str = "flag",
    label_col: str = "label",
    by: tuple[str, ...] = ("operator_id",),
    nominal: float | None = None,
) -> pd.DataFrame:
    """Per-group metric table over a flagged score bank.

    ``df`` must already carry a boolean ``flag_col`` (produced by calibration)
    and ``label_col`` (produced by the ground-truth criterion).
    """
    for col in (flag_col, label_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not present; run calibration/labelling first")

    rows: list[dict[str, Any]] = []
    for keys, part in df.groupby(list(by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        cov = empirical_coverage(part[flag_col], part[label_col])
        row = dict(zip(by, keys, strict=True))
        row.update(
            {
                "n_regions": int(len(part)),
                "n_hallucinated": int(_as_bool(part[label_col]).sum()),
                "empirical_coverage": cov,
                "miss_rate": 1.0 - cov,
                "flag_area": flag_area(part[flag_col]),
                "region_fdr": region_fdr(part[flag_col], part[label_col]),
            }
        )
        if nominal is not None:
            row["nominal_coverage"] = nominal
            row["coverage_gap"] = coverage_gap(nominal, cov)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_curve(
    per_operator: pd.DataFrame,
    distances: dict[str, float],
    distance_name: str,
) -> pd.DataFrame:
    """Join per-operator coverage to operator distance — the headline relationship.

    The claim (Sec. 11.2) is that this curve is monotone in distance, and that
    the MRI and CT curves collapse onto a shared trend once distance is
    normalised for ill-posedness.
    """
    out = per_operator.copy()
    out["operator_distance"] = out["operator_id"].map(distances)
    out["distance_measure"] = distance_name
    missing = out["operator_distance"].isna().sum()
    if missing:
        raise ValueError(f"{missing} operators have no distance to the calibration operator")
    return out.sort_values("operator_distance").reset_index(drop=True)


def monotonicity(curve: pd.DataFrame, y: str = "coverage_gap") -> float:
    """Spearman correlation of ``y`` with operator distance; 1.0 is perfectly monotone."""
    if len(curve) < 3:
        return float("nan")
    a = curve["operator_distance"].rank()
    b = curve[y].rank()
    return float(np.corrcoef(a, b)[0, 1])


def bound_agreement(measured: pd.Series, bound: pd.Series) -> dict[str, float]:
    """Compare measured coverage gaps with Track 3's derived bound (O6, Week 10).

    A valid bound is never violated; a useful bound is not vacuously loose.
    Both are reported, because a bound that is valid only because it is huge is
    not a result.
    """
    m = np.asarray(measured, dtype=float)
    b = np.asarray(bound, dtype=float)
    ok = np.isfinite(m) & np.isfinite(b)
    if not ok.any():
        return {"violation_rate": float("nan"), "median_slack": float("nan"), "n": 0}
    m, b = m[ok], b[ok]
    return {
        "violation_rate": float((m > b).mean()),
        "median_slack": float(np.median(b - m)),
        "max_violation": float(np.max(m - b)),
        "n": int(ok.sum()),
    }
