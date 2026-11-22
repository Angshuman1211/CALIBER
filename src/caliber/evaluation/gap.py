"""T1-B: does operator distance predict the coverage gap it costs you? (D88)

**Track 1 owns this module.** It answers the question T1-A leaves open. T1-A asks a yes/no
question -- is this transfer safe -- and answers it from the *sign* of the signed distance.
T1-B asks the quantitative one: given the distance, how much coverage do you lose? A sign
rule tells a practitioner whether to trust a deployment; a gap curve tells them what it
costs, which is what a pre-deployment estimate has to be.

## The three traps this module is built around

**1. The pairs are not the sample size.** There are 72 ordered pairs but only 9 operators,
and every operator appears in 16 pairs. An interval computed by resampling pairs would
describe a sample the design does not have. Every interval here resamples **operators**
(:data:`caliber.evaluation.prereg.GAP_CLUSTER_UNIT`) with the dyadic weighting that
implies, so a replicate that draws an operator twice weights its pairs accordingly.

**2. A baseline must be beaten on its own terms, not on a handicap.** Each competing
predictor is run through *the same isotonic machinery* on a different ``x``, so what is
compared is the information in the distance against the information in a view count --
not a flexible fit against a rigid one. The constant baseline is the L2-optimal constant
(the training mean), fitted the way isotonic is fitted and scored the way isotonic is
scored.

**3. Monotone interpolation is not a free lunch at the edges.** A fit on one family
predicting another can be asked for values outside its training range, where any monotone
predictor must clamp and so degenerates towards a constant. That would make the method
look exactly as good as the baseline it has to beat, for a reason having nothing to do
with the claim. :attr:`GapPrediction.extrapolated` reports the share of held-out points
that fell outside the training range, and it is printed every time rather than on request.

## What is frozen before any gap was read

The distance, the functional form and the baselines were frozen in D83/D84. The *response*
was not, which D88 closed: :data:`~caliber.evaluation.prereg.GAP_DEFINITION`,
``GAP_ERROR_METRIC``, ``GAP_PREDICTION_PROTOCOLS`` and ``GAP_CLUSTER_UNIT`` are named
there, together with ``GAP_DECLARATION_KNEW`` -- the summary figures Track 1 had already
been shown, including one known monotonicity violation. That list is the honest statement
of what this freeze is worth: it is "before the per-pair gaps", not "before any knowledge".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from caliber.evaluation import prereg

#: Where each predictor is defined, mirroring ``prereg.BASELINE_SCOPE`` in code.
#: ``None`` means "every pair"; a family name means "both operators in that family".
PREDICTOR_SCOPE = MappingProxyType(
    {
        "signed_distance": None,
        "constant": None,
        "acquisition_parameter": "within_family",
        "null_space_dimension": None,
    }
)


# -- isotonic regression, by pool-adjacent-violators ----------------------------------


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """L2 isotonic regression of ``y`` with weights ``w``, assuming ``x`` already sorted.

    Written out because the project has neither scikit-learn nor a working scipy. The
    algorithm is the standard one: walk left to right keeping a stack of blocks, and merge
    backwards whenever the new block would break monotonicity.
    """
    blocks: list[list[float]] = []   # [weighted mean, total weight, count]
    for value, weight in zip(y, w):
        cur = [float(value), float(weight), 1]
        while blocks and blocks[-1][0] > cur[0]:
            prev = blocks.pop()
            total = prev[1] + cur[1]
            cur = [
                (prev[0] * prev[1] + cur[0] * cur[1]) / total if total > 0 else prev[0],
                total,
                prev[2] + cur[2],
            ]
        blocks.append(cur)
    out: list[float] = []
    for value, _, count in blocks:
        out.extend([value] * int(count))
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class IsotonicFit:
    """A non-decreasing step-and-interpolate predictor: the frozen functional form.

    Prediction is linear between knots and clamped outside them. Clamping is what makes
    :attr:`GapPrediction.extrapolated` worth reporting -- outside the training range every
    monotone predictor returns the same number, which is a constant predictor in disguise.

    ``direction`` is ``increasing`` for the frozen form. A ``decreasing`` fit is available
    only so that a reversed relation is *visible*: a non-decreasing fit to a decreasing
    truth collapses to very nearly the training mean, so the frozen form reports a real
    reversal as "no information", which is the wrong answer for the wrong reason. Showing
    both separates "the distance says nothing" from "the distance says the opposite".
    """

    x: np.ndarray
    y: np.ndarray
    direction: str = "increasing"

    @property
    def lo(self) -> float:
        return float(self.x[0]) if self.x.size else float("nan")

    @property
    def hi(self) -> float:
        return float(self.x[-1]) if self.x.size else float("nan")

    def predict(self, x: Sequence[float]) -> np.ndarray:
        q = np.asarray(x, dtype=float)
        if self.x.size == 0:
            return np.full(q.shape, float("nan"))
        if self.x.size == 1:
            return np.where(np.isfinite(q), float(self.y[0]), float("nan"))
        out = np.interp(q, self.x, self.y)
        return np.where(np.isfinite(q), out, float("nan"))


def isotonic(
    x: Sequence[float], y: Sequence[float], direction: str | None = None
) -> IsotonicFit:
    """Fit the frozen form. Ties in ``x`` are averaged into one knot, then weighted.

    Averaging ties rather than ordering them arbitrarily matters here: the null-space
    baseline ties four of the nine operators, so its ``x`` has heavy ties by construction
    and an arbitrary tie-break would let the fit read structure out of input order.

    ``direction`` defaults to :data:`prereg.GAP_PREDICTION_DIRECTION`. The decreasing fit
    is the same algorithm on a negated response, which is exactly the L2-optimal
    non-increasing fit -- not a second method with its own tuning.
    """
    direction = prereg.GAP_PREDICTION_DIRECTION if direction is None else direction
    if direction not in prereg.GAP_PREDICTION_DIRECTIONS:
        raise ValueError(
            f"{direction!r} is not a frozen direction; "
            f"have {list(prereg.GAP_PREDICTION_DIRECTIONS)}"
        )
    sign = 1.0 if direction == "increasing" else -1.0
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return IsotonicFit(np.empty(0), np.empty(0), direction)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    uniq, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=ys) / counts
    fitted = sign * _pava(sign * means, counts.astype(float))
    return IsotonicFit(uniq, fitted, direction)


# -- the operator grid and the predictors built on it ---------------------------------


def operator_metadata(manifest: str | Path) -> dict[str, dict]:
    """``operator_id -> {family, views, arc_deg, null_space_dim}`` from a run manifest."""
    blob = json.loads(Path(manifest).read_text(encoding="utf-8"))
    out = {}
    for op in blob.get("operators", ()):
        params = op.get("params", {}) or {}
        out[str(op["operator_id"])] = {
            "family": op.get("family", "?"),
            "views": params.get("n_views"),
            "arc_deg": params.get("angular_range_deg"),
            "null_space_dim": op.get("null_space_dim"),
        }
    if not out:
        raise ValueError(f"{manifest} lists no operators")
    return out


def _meta_for(op: str, meta: Mapping[str, Mapping]) -> Mapping:
    """``meta[op]``, raising rather than defaulting to "undefined".

    An operator the grid does not know is a typo or a stale manifest, not a pair whose
    baseline is out of scope. Returning NaN for it would move that pair out of the
    denominator silently, and a denominator that moves on its own is the defect this
    project has now been bitten by twice.
    """
    try:
        return meta[op]
    except KeyError:
        raise KeyError(
            f"{op!r} is not in the operator grid; have {sorted(meta)}"
        ) from None


def _acquisition_knob(meta: Mapping[str, object]) -> float:
    """The one scalar that was turned, per family: views for sparse view, arc for limited.

    Undefined across families *by construction* and not by omission: every limited-angle
    operator on this grid has 120 views and every sparse-view one spans 180 degrees, so
    neither knob can order an operator of one family against the other (D84).
    """
    if meta.get("family") == "sparse_view":
        return float(meta["views"]) if meta.get("views") is not None else float("nan")
    if meta.get("family") == "limited_angle":
        return float(meta["arc_deg"]) if meta.get("arc_deg") is not None else float("nan")
    return float("nan")


def acquisition_parameter(a: str, b: str, meta: Mapping[str, Mapping]) -> float:
    """Signed knob difference, oriented like the distance: positive means *more* ill-posed.

    Fewer views, or a narrower arc, is more ill-posed -- so the orientation is ``a - b``.
    Returns NaN when the two operators are in different families, which is the scope
    ``prereg.BASELINE_SCOPE`` declares, rather than a number that pretends to compare them.
    """
    ma, mb = _meta_for(a, meta), _meta_for(b, meta)
    if ma.get("family") != mb.get("family"):
        return float("nan")
    ka, kb = _acquisition_knob(ma), _acquisition_knob(mb)
    return float(ka - kb)


def null_space_dimension(a: str, b: str, meta: Mapping[str, Mapping]) -> float:
    """Signed null-space dimension difference, ``dim(b) - dim(a)``: bigger null space at the
    deployment operator means more of the image is unconstrained, i.e. uphill."""
    da = _meta_for(a, meta).get("null_space_dim")
    db = _meta_for(b, meta).get("null_space_dim")
    if da is None or db is None:
        return float("nan")
    return float(db) - float(da)


PREDICTORS: Mapping[str, Callable[[str, str, Mapping], float]] = MappingProxyType(
    {
        "acquisition_parameter": acquisition_parameter,
        "null_space_dimension": null_space_dimension,
    }
)


# -- the response: what the gap is ----------------------------------------------------


def gap_column(
    frame: pd.DataFrame,
    definition: str | None = None,
    coverage_col: str = "coverage",
    keys: Sequence[str] = ("calibration", "deployment"),
) -> pd.Series:
    """The frozen response variable (:data:`prereg.GAP_DEFINITION`), as a column.

    ``matched_minus_transfer`` needs the **matched** cell at each deployment operator, so
    the frame must carry its diagonal. T1-A's ordered-pair frame deliberately excludes it;
    Track 2's 81-cell table carries it. A frame without it raises rather than falling back
    to the nominal-difference form, because the two definitions are not interchangeable and
    silently reading one as the other is how a response variable changes under a result.
    """
    which = prereg.GAP_DEFINITION if definition is None else str(definition)
    if which not in prereg.GAP_DEFINITION_SENSITIVITY:
        raise ValueError(
            f"{which!r} is not a gap definition; have "
            f"{list(prereg.GAP_DEFINITION_SENSITIVITY)}"
        )
    cov = frame[coverage_col].astype(float)
    if which == "nominal_minus_transfer":
        return (prereg.NOMINAL - cov).rename("gap")
    cal, dep = keys
    matched = frame[frame[cal].astype(str) == frame[dep].astype(str)]
    if matched.empty:
        raise KeyError(
            f"{which!r} needs the matched diagonal (rows where {cal} == {dep}) and this "
            "frame has none. Pass the full cell table, or name "
            "'nominal_minus_transfer' deliberately as the sensitivity arm."
        )
    level = matched.set_index(matched[dep].astype(str))[coverage_col].astype(float)
    base = frame[dep].astype(str).map(level)
    missing = sorted(set(frame.loc[base.isna(), dep].astype(str)))
    if missing:
        raise KeyError(
            f"no matched cell for deployment operator(s) {missing}; the gap would be "
            "undefined there and dropping them would move the denominator"
        )
    return (base - cov).rename("gap")


# -- held-out protocols ---------------------------------------------------------------


def _families(pairs: pd.DataFrame, meta: Mapping[str, Mapping], keys) -> tuple:
    cal, dep = keys
    fa = pairs[cal].astype(str).map(lambda o: _meta_for(o, meta)["family"])
    fb = pairs[dep].astype(str).map(lambda o: _meta_for(o, meta)["family"])
    return fa, fb


def pair_scope(
    pairs: pd.DataFrame,
    meta: Mapping[str, Mapping],
    keys: Sequence[str] = ("calibration", "deployment"),
) -> pd.Series:
    """``within_family`` or ``cross_family`` per pair -- the axis results are split on."""
    fa, fb = _families(pairs, meta, keys)
    return pd.Series(
        np.where(fa.to_numpy() == fb.to_numpy(), "within_family", "cross_family"),
        index=pairs.index, name="scope",
    )


def folds(
    pairs: pd.DataFrame,
    protocol: str,
    meta: Mapping[str, Mapping],
    keys: Sequence[str] = ("calibration", "deployment"),
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(fold name, train mask, test mask)`` for a frozen protocol.

    ``cross_family`` trains on the pairs **internal** to one family and tests on everything
    else -- the other family's internal pairs *and* the pairs that span the two. The span
    pairs are the point: no acquisition knob orders an operator of one family against the
    other, so they are where a probe-based measure can show information no parameter has.

    ``leave_one_operator_out`` holds out all 16 ordered pairs touching one operator. Each
    pair is therefore tested twice, once per endpoint. The duplication is kept rather than
    averaged: averaging two held-out predictions is a small ensemble, and it would help the
    flexible predictor more than the constant one.
    """
    cal, dep = keys
    a = pairs[cal].astype(str).to_numpy()
    b = pairs[dep].astype(str).to_numpy()
    if protocol == "cross_family":
        fa, fb = _families(pairs, meta, keys)
        fa, fb = fa.to_numpy(), fb.to_numpy()
        out = []
        for fam in sorted(set(fa) | set(fb)):
            train = (fa == fam) & (fb == fam)
            if not train.any():
                continue
            out.append((f"train_{fam}", train, ~train))
        return out
    if protocol == "leave_one_operator_out":
        out = []
        for op in sorted(set(a) | set(b)):
            test = (a == op) | (b == op)
            out.append((f"holdout_{op}", ~test, test))
        return out
    raise ValueError(
        f"{protocol!r} is not a frozen protocol; have "
        f"{list(prereg.GAP_PREDICTION_PROTOCOLS)}"
    )


# -- fitting and scoring ---------------------------------------------------------------


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    predictor: str,
    direction: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out predictions and an "outside the training range" flag.

    The constant predictor is the **training mean**: the L2-optimal constant, so it is
    fitted under the same criterion as isotonic and scored under another
    (:data:`prereg.GAP_ERROR_METRIC`), exactly as the method is.
    """
    if predictor == "constant":
        ok = np.isfinite(y_train)
        value = float(y_train[ok].mean()) if ok.any() else float("nan")
        return np.full(x_test.shape, value), np.zeros(x_test.shape, dtype=bool)
    fit = isotonic(x_train, y_train, direction)
    pred = fit.predict(x_test)
    outside = np.isfinite(x_test) & ((x_test < fit.lo) | (x_test > fit.hi))
    return pred, outside


def held_out(
    pairs: pd.DataFrame,
    x: np.ndarray,
    gap: np.ndarray,
    protocol: str,
    predictor: str,
    meta: Mapping[str, Mapping],
    keys: Sequence[str] = ("calibration", "deployment"),
    direction: str | None = None,
) -> pd.DataFrame:
    """One row per (fold, tested pair): prediction, truth, and the diagnostics.

    Rows where the predictor is undefined keep their place with a NaN prediction instead of
    vanishing, so a caller comparing two predictors can see *which* pairs each one can
    speak about and restrict the comparison to their intersection.
    """
    cal, dep = keys
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(gap, dtype=float)
    rows = []
    for name, train, test in folds(pairs, protocol, meta, keys):
        pred, outside = _fit_predict(
            xs[train], ys[train], xs[test], predictor, direction
        )
        idx = np.flatnonzero(test)
        rows.append(pd.DataFrame({
            "fold": name,
            "calibration": pairs[cal].astype(str).to_numpy()[idx],
            "deployment": pairs[dep].astype(str).to_numpy()[idx],
            "x": xs[test],
            "gap": ys[test],
            "prediction": pred,
            "extrapolated": outside,
            "n_train": int(np.isfinite(xs[train] + ys[train]).sum()),
            "direction": (
                prereg.GAP_PREDICTION_DIRECTION if direction is None else direction
            ),
        }))
    if not rows:
        return pd.DataFrame(
            columns=["fold", "calibration", "deployment", "x", "gap", "prediction",
                     "extrapolated", "n_train", "direction"]
        )
    return pd.concat(rows, ignore_index=True)


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def operator_weights(
    rng: np.random.Generator, operators: Sequence[str]
) -> dict[str, int]:
    """One dyadic bootstrap replicate: resample operators, weight pairs by the product.

    A pair ``(a, b)`` is an observation *about two operators*, so a replicate that draws
    ``a`` twice and ``b`` once should count that pair ``2 x 1`` times. Resampling the 72
    pairs directly would instead report an interval for 72 independent observations, and
    there are nine (:data:`prereg.GAP_CLUSTER_UNIT`).
    """
    ops = list(operators)
    draw = rng.integers(0, len(ops), size=len(ops))
    counts = {o: 0 for o in ops}
    for i in draw:
        counts[ops[int(i)]] += 1
    return counts


def _weighted_mae(err: np.ndarray, w: np.ndarray) -> float:
    total = float(w.sum())
    return float(np.sum(w * np.abs(err)) / total) if total > 0 else float("nan")


@dataclass(frozen=True)
class GapPrediction:
    """Held-out accuracy of one predictor, under one protocol, on one family scope."""

    protocol: str
    predictor: str
    scope: str
    n_observations: int
    n_pairs: int
    mae: float
    rmse: float
    extrapolated: float
    gap_sd: float
    defined: bool
    direction: str = prereg.GAP_PREDICTION_DIRECTION

    @property
    def pre_registered(self) -> bool:
        """Only the frozen direction may reach a verdict. See D89."""
        return self.direction == prereg.GAP_PREDICTION_DIRECTION

    @property
    def skill(self) -> float:
        """MAE as a fraction of the gaps' own spread. Above 1 is worse than saying nothing."""
        return self.mae / self.gap_sd if self.gap_sd > 0 else float("nan")

    def as_row(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "predictor": self.predictor,
            "scope": self.scope,
            "n_observations": self.n_observations,
            "n_pairs": self.n_pairs,
            "mae": self.mae,
            "rmse": self.rmse,
            "mae_over_gap_sd": self.skill,
            "extrapolated": self.extrapolated,
            "defined": self.defined,
            "direction": self.direction,
            "pre_registered": self.pre_registered,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Whether the distance beats one baseline, on the pairs where both can speak."""

    protocol: str
    scope: str
    baseline: str
    n_common: int
    n_dropped: int
    mae_distance: float
    mae_baseline: float
    delta: float
    lo: float
    hi: float
    level: float

    @property
    def beaten(self) -> bool:
        """The baseline is beaten only if the whole interval is on the right side of zero."""
        return bool(np.isfinite(self.lo) and self.lo > 0.0)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.delta):
            return "undefined_here"
        if self.beaten:
            return "distance_better"
        if np.isfinite(self.hi) and self.hi < 0.0:
            return "baseline_better"
        return "indistinguishable"

    def as_row(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "scope": self.scope,
            "baseline": self.baseline,
            "n_common": self.n_common,
            "n_dropped": self.n_dropped,
            "mae_distance": self.mae_distance,
            "mae_baseline": self.mae_baseline,
            "delta_mae": self.delta,
            "delta_lo": self.lo,
            "delta_hi": self.hi,
            "beaten": self.beaten,
            "verdict": self.verdict,
        }


def summarise(
    scored: pd.DataFrame,
    protocol: str,
    predictor: str,
    scope: str,
    gap_sd: float,
    direction: str | None = None,
) -> GapPrediction:
    """Collapse one predictor's held-out rows into its reported accuracy."""
    usable = scored[np.isfinite(scored["prediction"]) & np.isfinite(scored["gap"])]
    err = (usable["prediction"] - usable["gap"]).to_numpy(dtype=float)
    pairs = usable[["calibration", "deployment"]].drop_duplicates()
    return GapPrediction(
        protocol=protocol,
        predictor=predictor,
        scope=scope,
        n_observations=int(len(usable)),
        n_pairs=int(len(pairs)),
        mae=_mae(err),
        rmse=_rmse(err),
        extrapolated=(
            float(usable["extrapolated"].mean()) if len(usable) else float("nan")
        ),
        gap_sd=float(gap_sd),
        defined=bool(len(usable) > 0),
        direction=(
            prereg.GAP_PREDICTION_DIRECTION if direction is None else direction
        ),
    )


def compare_to_baseline(
    method: pd.DataFrame,
    baseline: pd.DataFrame,
    protocol: str,
    scope: str,
    name: str,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BaselineComparison:
    """Paired MAE difference with an operator-clustered interval.

    **Restricted to the rows both predictors can score**, with ``n_dropped`` saying what
    that cost. An unrestricted comparison would put two error rates on two different
    samples -- the baseline is undefined on cross-family pairs by construction, so the
    unrestricted version would compare the distance on hard pairs against the baseline on
    easy ones and call the difference skill.
    """
    join = ["fold", "calibration", "deployment"]
    merged = method.merge(baseline, on=join, how="inner", suffixes=("_m", "_b"))
    both = merged[
        np.isfinite(merged["prediction_m"])
        & np.isfinite(merged["prediction_b"])
        & np.isfinite(merged["gap_m"])
    ]
    dropped = int(len(merged) - len(both))
    if both.empty:
        nan = float("nan")
        return BaselineComparison(
            protocol, scope, name, 0, dropped, nan, nan, nan, nan, nan, level
        )
    err_m = (both["prediction_m"] - both["gap_m"]).to_numpy(dtype=float)
    err_b = (both["prediction_b"] - both["gap_m"]).to_numpy(dtype=float)
    cal = both["calibration"].to_numpy()
    dep = both["deployment"].to_numpy()
    operators = sorted(set(cal) | set(dep))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        counts = operator_weights(rng, operators)
        w = np.array([counts[a] * counts[b] for a, b in zip(cal, dep)], dtype=float)
        deltas[r] = _weighted_mae(err_b, w) - _weighted_mae(err_m, w)
    tail = (1.0 - level) / 2.0
    finite = deltas[np.isfinite(deltas)]
    lo, hi = (
        (float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail)))
        if finite.size
        else (float("nan"), float("nan"))
    )
    return BaselineComparison(
        protocol=protocol,
        scope=scope,
        baseline=name,
        n_common=int(len(both)),
        n_dropped=dropped,
        mae_distance=_mae(err_m),
        mae_baseline=_mae(err_b),
        delta=_mae(err_b) - _mae(err_m),
        lo=lo,
        hi=hi,
        level=level,
    )


def predictor_column(
    pairs: pd.DataFrame,
    name: str,
    meta: Mapping[str, Mapping],
    distance_col: str = "signed_distance",
    keys: Sequence[str] = ("calibration", "deployment"),
) -> np.ndarray:
    """The ``x`` one predictor sees, per pair. ``constant`` has none and gets NaN."""
    cal, dep = keys
    if name == "signed_distance":
        return pairs[distance_col].to_numpy(dtype=float)
    if name == "constant":
        return np.full(len(pairs), float("nan"))
    try:
        fn = PREDICTORS[name]
    except KeyError:
        raise ValueError(
            f"{name!r} is not a predictor; have "
            f"{['signed_distance', 'constant', *PREDICTORS]}"
        ) from None
    return np.array(
        [fn(str(a), str(b), meta) for a, b in zip(pairs[cal], pairs[dep])], dtype=float
    )


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho, written out because scipy is not usable in this project.

    Ties get their average rank, which is what makes this comparable to the isotonic fit:
    both treat a tie in ``x`` as one position rather than inventing an order for it.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if xs.size < 3:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        out = np.empty(v.size, dtype=float)
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def gap_sign_from_thresholds(
    pairs: pd.DataFrame,
    threshold_col: str = "threshold",
    gap_col: str = "gap",
    definition: str | None = None,
    keys: Sequence[str] = ("calibration", "deployment"),
) -> pd.DataFrame:
    """**A check on the bank, never a predictor.** See :data:`prereg.GAP_BARRED_PREDICTORS`.

    Coverage of the positives at ``b`` is the fraction whose score clears the threshold it
    was given, so with ``F_b`` the CDF of those scores::

        coverage(a->b) = 1 - F_b(tau_a)      coverage(b->b) = 1 - F_b(tau_b)
        gap            = F_b(tau_a) - F_b(tau_b)

    ``F_b`` is non-decreasing, so ``sign(gap) == sign(tau_a - tau_b)`` **exactly, at every
    ordered pair, as a theorem**. That is why ``tau_a - tau_b`` is barred from the baseline
    set: it would beat the distance and every baseline, and the win would be arithmetic.

    What the identity is good for is the other direction. Agreement below 100% cannot
    happen if the recorded thresholds and coverages came from the same scores, so a
    disagreeing pair means the bank contradicts the definition of coverage -- a defect that
    no coverage result survives, found for the price of one groupby. A check that can only
    find faults cannot flatter the hypothesis.

    This holds for ``matched_minus_transfer`` only; against a fixed nominal there is no
    ``tau_b`` in the expression and the identity says nothing.
    """
    definition = prereg.GAP_DEFINITION if definition is None else definition
    if definition != "matched_minus_transfer":
        raise ValueError(
            "the threshold identity is only defined for matched_minus_transfer; "
            f"got {definition!r}"
        )
    cal, dep = keys
    for col in (threshold_col, gap_col, cal, dep):
        if col not in pairs.columns:
            raise KeyError(f"no {col!r} column; need it for the threshold identity")

    tau: dict[str, float] = {}
    for op, part in pairs.groupby(pairs[cal].astype(str), sort=True):
        values = part[threshold_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        spread = float(values.max() - values.min())
        scale = max(abs(float(values.mean())), 1e-12)
        if spread / scale > 1e-9:
            raise ValueError(
                f"operator {op!r} carries {values.size} differing thresholds "
                f"(spread {spread:.3g}); tau_a must be one number per calibration "
                "operator or the identity has no left-hand side"
            )
        tau[str(op)] = float(values[0])

    tau_a = pairs[cal].astype(str).map(tau).to_numpy(dtype=float)
    tau_b = pairs[dep].astype(str).map(tau).to_numpy(dtype=float)
    gaps = pairs[gap_col].to_numpy(dtype=float)
    diff = tau_a - tau_b
    checkable = np.isfinite(diff) & np.isfinite(gaps)
    # A zero on either side is consistent with any sign on the other: F_b can be flat.
    decidable = checkable & (diff != 0.0) & (gaps != 0.0)
    agrees = np.where(decidable, np.sign(diff) == np.sign(gaps), False)
    return pd.DataFrame({
        cal: pairs[cal].astype(str).to_numpy(),
        dep: pairs[dep].astype(str).to_numpy(),
        "tau_calibration": tau_a,
        "tau_deployment": tau_b,
        "tau_difference": diff,
        "gap": gaps,
        "decidable": decidable,
        "agrees": agrees,
    })


@dataclass(frozen=True)
class DirectionCall:
    """What the threshold ordering says the direction of the fit must be.

    :attr:`determinate` is ``None`` until a measured per-operator threshold spread is
    supplied. It is not defaulted, because the alternative to measuring it is assuming the
    tau range is real -- and T2-A4 exists precisely because a conformal threshold is a
    single order statistic whose spread can be entirely sampling noise.
    """

    direction: str
    rho: float
    tau_range: float
    tau_noise: float | None = None
    determinate: bool | None = None
    source: str = prereg.GAP_DIRECTION_SOURCE
    bank: str | None = None

    @property
    def reportable(self) -> bool:
        """False on any bank whose per-pair gaps were already seen (amendment seven)."""
        if self.determinate is not True:
            return False
        return str(self.bank) not in set(prereg.GAP_DIRECTION_BANKS_BARRED)


def direction_from_thresholds(
    pairs: pd.DataFrame,
    threshold_col: str = "threshold",
    distance_col: str = "signed_distance",
    keys: Sequence[str] = ("calibration", "deployment"),
    tau_noise: float | None = None,
    bank: str | None = None,
) -> DirectionCall:
    """Read the direction of the gap-distance relation off the thresholds (D91).

    Takes the recorded thresholds and the signed distance and **nothing else**: no
    coverage, no gap, no fitted value. By D89 ``sign(gap) == sign(tau_a - tau_b)``, so a
    distance that orders the thresholds one way must order the gaps the other, and which
    way is a property of the bank. See :data:`prereg.GAP_DIRECTION_RULE`.

    This is deliberately *not* a predictor: it yields one bit, the same bit for all 72
    pairs of a bank, where :data:`prereg.GAP_BARRED_PREDICTORS` forbids the 72 per-pair
    values. The bit is still a bit, which is why :attr:`DirectionCall.reportable` is False
    on every bank in :data:`prereg.GAP_DIRECTION_BANKS_BARRED`.
    """
    cal, dep = keys
    for col in (threshold_col, distance_col, cal, dep):
        if col not in pairs.columns:
            raise KeyError(f"no {col!r} column; need it to read the direction")
    tau: dict[str, float] = {}
    for op, part in pairs.groupby(pairs[cal].astype(str), sort=True):
        values = part[threshold_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            tau[str(op)] = float(values[0])
    if len(tau) < 3:
        raise ValueError(
            f"only {len(tau)} calibration operators carry a threshold; the ordering "
            "is not readable from fewer than three"
        )

    tau_a = pairs[cal].astype(str).map(tau).to_numpy(dtype=float)
    tau_b = pairs[dep].astype(str).map(tau).to_numpy(dtype=float)
    dist = pairs[distance_col].to_numpy(dtype=float)
    rho = rank_correlation(dist, tau_b - tau_a)
    # tau rises with the distance  =>  tau_b > tau_a at positive distance  =>  gap falls.
    direction = "decreasing" if rho > 0 else "increasing"

    values = np.array(sorted(tau.values()), dtype=float)
    tau_range = float(values[-1] - values[0])
    determinate: bool | None = None
    if tau_noise is not None and np.isfinite(tau_noise):
        if tau_noise <= 0:
            determinate = True
        else:
            ratio = tau_range / float(tau_noise)
            determinate = bool(ratio >= prereg.GAP_DIRECTION_DETERMINACY_MIN_RATIO)
    if not np.isfinite(rho):
        determinate = False
    return DirectionCall(
        direction=direction,
        rho=float(rho),
        tau_range=tau_range,
        tau_noise=None if tau_noise is None else float(tau_noise),
        determinate=determinate,
        bank=bank,
    )


def arm_difference_identity(
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    matched_a: pd.DataFrame | None = None,
    matched_b: pd.DataFrame | None = None,
    coverage_col: str = "coverage",
    gap_col: str = "gap",
    keys: Sequence[str] = ("calibration", "deployment"),
) -> dict[str, float]:
    """**The two label arms' difference tables are not independent.** (D91)

    With ``level(a,b) = coverage(a->b)`` and ``gap(a,b) = level(b,b) - level(a,b)``, the
    difference between two label arms obeys, as algebra and not as an approximation::

        gap_diff(a,b) == level_diff(b,b) - level_diff(a,b)

    So the gap-difference table is the level-difference table minus its own diagonal, and
    the honest reading of "the gap agrees better than the level" is **the displacement at
    the matched cell is strongly correlated with the displacement at the transfer cell**.
    ``common_mode_correlation`` is that correlation, which is what D86's argument rests
    on. At zero correlation the gap difference is *no smaller* than the level difference,
    so this statistic can fail, and loudly.

    ``common_mode_correlation_lower_bound`` is the useful number, because it needs no
    assumption about the two displacements' relative spread: writing ``s`` for that ratio,
    ``sd(gap)/sd(level) = sqrt(s*s - 2*r*s + 1)``, which over all ``s`` bottoms out at
    ``sqrt(1 - r*r)``. An observed ratio of 0.36 therefore forces ``r >= 0.93`` however the
    spreads fall out. Inferring ``r`` from the ratio directly would instead assume equal
    spreads, and that assumption is false by a factor of 17 on this module's own fixture.

    ``residual`` is the identity's own violation and must be at machine precision. Anything
    larger means the two arms' tables disagree about their own diagonal.
    """
    cal, dep = keys
    left = arm_a.set_index([arm_a[cal].astype(str), arm_a[dep].astype(str)])
    right = arm_b.set_index([arm_b[cal].astype(str), arm_b[dep].astype(str)])
    shared = [k for k in left.index if k in set(right.index)]
    if len(shared) < 3:
        raise ValueError(
            f"the two arms share only {len(shared)} pairs; nothing to check"
        )
    lvl = np.array([float(left.loc[k, coverage_col]) for k in shared])
    lvl -= np.array([float(right.loc[k, coverage_col]) for k in shared])
    gap = np.array([float(left.loc[k, gap_col]) for k in shared])
    gap -= np.array([float(right.loc[k, gap_col]) for k in shared])

    out: dict[str, float] = {
        "n_pairs": float(len(shared)),
        "mean_abs_level_difference": float(np.abs(lvl).mean()),
        "mean_abs_gap_difference": float(np.abs(gap).mean()),
        "median_level_difference": float(np.median(lvl)),
        "median_gap_difference": float(np.median(gap)),
    }
    ratio = out["mean_abs_gap_difference"] / max(out["mean_abs_level_difference"], 1e-15)
    out["abs_difference_ratio"] = float(ratio)
    out["ratio_at_zero_common_mode"] = 1.0
    out["sd_level_difference"] = float(np.std(lvl))
    out["sd_gap_difference"] = float(np.std(gap))
    sd_ratio = out["sd_gap_difference"] / max(out["sd_level_difference"], 1e-15)
    out["sd_difference_ratio"] = float(sd_ratio)
    # With s = sd(matched)/sd(level) and r the correlation between the two
    # displacements, sd(gap)/sd(level) = sqrt(s^2 - 2 r s + 1), which is minimised at
    # s = r and takes the value sqrt(1 - r^2) there. So the observed ratio bounds r
    # from below WHATEVER the ratio of spreads turns out to be -- no equal-variance
    # assumption, which is the step that would otherwise be smuggled in here.
    out["common_mode_correlation_lower_bound"] = (
        float(np.sqrt(max(1.0 - min(sd_ratio, 1.0) ** 2, 0.0)))
    )
    out["spreads_ratio_at_which_the_bound_is_tight"] = (
        out["common_mode_correlation_lower_bound"]
    )

    if matched_a is not None and matched_b is not None:
        ma = matched_a.set_index(matched_a[dep].astype(str))[coverage_col]
        mb = matched_b.set_index(matched_b[dep].astype(str))[coverage_col]
        diag = np.array([float(ma.loc[k[1]]) - float(mb.loc[k[1]]) for k in shared])
        out["residual"] = float(np.abs(gap - (diag - lvl)).max())
        sd_l, sd_d = float(np.std(lvl)), float(np.std(diag))
        if sd_l > 0 and sd_d > 0:
            centred = (lvl - lvl.mean()) * (diag - diag.mean())
            out["common_mode_correlation"] = float(centred.mean() / (sd_l * sd_d))
    return out


def gap_prediction(
    pairs: pd.DataFrame,
    meta: Mapping[str, Mapping],
    distance_col: str = "signed_distance",
    gap_col: str = "gap",
    protocols: Sequence[str] | None = None,
    baselines: Sequence[str] | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    keys: Sequence[str] = ("calibration", "deployment"),
) -> dict[str, pd.DataFrame]:
    """T1-B end to end: accuracy, baseline comparisons, and the reporting verdict.

    Returns ``accuracy``, ``accuracy_by_fold``, ``comparisons``, ``verdict`` and
    ``reversal`` frames. ``reversal`` is the one to read first on the CT bank: the frozen
    direction is increasing, the measured relation within every family cell is decreasing,
    and a non-decreasing fit to a decreasing truth returns the training mean -- so without
    that frame the output would say "no information" where the truth is "the opposite
    information". It is a diagnostic and never reaches ``verdict``. Protocols and baselines
    default to the frozen tuples, and within-family and cross-family pairs are always both
    reported (:data:`prereg.GAP_PREDICTION_REPORT_BY_FAMILY_SCOPE`) because pooling them
    would let 36 pairs with a strong baseline average against 36 where it does not exist.

    The verdict is deliberately conjunctive: the distance is reported as predictive on a
    scope only if it beats **every** baseline defined there. One baseline surviving is
    enough to withhold the claim, because the objection T1-B answers is "any monotone
    function of the knob would do this".
    """
    protos = tuple(prereg.GAP_PREDICTION_PROTOCOLS if protocols is None else protocols)
    bases = tuple(prereg.GAP_PREDICTION_BASELINES if baselines is None else baselines)
    if gap_col not in pairs.columns:
        raise KeyError(f"no {gap_col!r} column; build it with gap_column()")
    scope = pair_scope(pairs, meta, keys)
    gaps = pairs[gap_col].to_numpy(dtype=float)
    columns = {
        name: predictor_column(pairs, name, meta, distance_col, keys)
        for name in ("signed_distance", *bases)
    }
    acc, comps, per_fold, reversal = [], [], [], []
    for protocol in protos:
        scored = {
            name: held_out(pairs, x, gaps, protocol, name, meta, keys)
            for name, x in columns.items()
        }
        for which in ("within_family", "cross_family"):
            keep = set(
                map(tuple, pairs.loc[scope == which, list(keys)].astype(str).to_numpy())
            )
            if not keep:
                continue
            sd = float(np.nanstd(gaps[(scope == which).to_numpy()], ddof=1))
            sliced = {}
            for name, frame in scored.items():
                mask = [
                    (str(a), str(b)) in keep
                    for a, b in zip(frame["calibration"], frame["deployment"])
                ]
                sliced[name] = frame[np.asarray(mask, dtype=bool)]
                acc.append(summarise(sliced[name], protocol, name, which, sd).as_row())
                # "Fit on sparse-view, predict limited-angle, then reverse" are two
                # different claims, and the work plan asks for both. Aggregating over
                # folds averages a fit on 30 pairs predicting 6 against a fit on 6
                # predicting 30, which is the asymmetry most worth seeing.
                for fold_name, part in sliced[name].groupby("fold", sort=True):
                    row = summarise(part, protocol, name, which, sd).as_row()
                    row["fold"] = fold_name
                    per_fold.append(row)
            for name in bases:
                comps.append(
                    compare_to_baseline(
                        sliced["signed_distance"], sliced[name], protocol, which, name,
                        n_boot=n_boot, level=level, seed=seed,
                    ).as_row()
                )
            # The reversal diagnostic. Held out the same way, scored the same way, and
            # kept out of the verdict: a direction chosen after seeing the direction is
            # one free bit, and one free bit on nine operators is not a result (D89).
            other = next(
                d for d in prereg.GAP_PREDICTION_DIRECTIONS
                if d != prereg.GAP_PREDICTION_DIRECTION
            )
            flipped = held_out(
                pairs, columns["signed_distance"], gaps, protocol,
                "signed_distance", meta, keys, direction=other,
            )
            fmask = [
                (str(a), str(b)) in keep
                for a, b in zip(flipped["calibration"], flipped["deployment"])
            ]
            flipped = flipped[np.asarray(fmask, dtype=bool)]
            frozen_fit = summarise(
                sliced["signed_distance"], protocol, "signed_distance", which, sd
            )
            other_fit = summarise(
                flipped, protocol, "signed_distance", which, sd, direction=other
            )
            acc.append(other_fit.as_row())
            in_scope = (scope == which).to_numpy()
            reversal.append({
                "protocol": protocol,
                "scope": which,
                "rho_distance_gap": rank_correlation(
                    columns["signed_distance"][in_scope], gaps[in_scope]
                ),
                "mae_frozen_direction": frozen_fit.mae,
                "mae_opposite_direction": other_fit.mae,
                "gap_sd": sd,
                "opposite_fits_better": bool(
                    np.isfinite(other_fit.mae)
                    and np.isfinite(frozen_fit.mae)
                    and other_fit.mae < frozen_fit.mae
                ),
                "frozen_direction_is_informative": bool(
                    np.isfinite(frozen_fit.mae) and sd > 0 and frozen_fit.mae < sd
                ),
            })
    accuracy = pd.DataFrame(acc)
    comparisons = pd.DataFrame(comps)
    rows = []
    for (protocol, which), part in comparisons.groupby(["protocol", "scope"], sort=True):
        # comparisons are built from the frozen-direction fit only; the opposite-direction
        # rows live in "accuracy" (flagged pre_registered=False) and "reversal".
        live = part[part["verdict"] != "undefined_here"]
        unbeaten = sorted(live.loc[~live["beaten"], "baseline"])
        rows.append({
            "protocol": protocol,
            "scope": which,
            "n_baselines_defined": int(len(live)),
            "n_baselines_beaten": int(live["beaten"].sum()),
            "unbeaten": ", ".join(unbeaten),
            "reportable": bool(len(live) > 0 and not unbeaten),
        })
    by_fold = pd.DataFrame(per_fold)
    if not by_fold.empty:
        lead = ["protocol", "fold", "scope", "predictor"]
        by_fold = by_fold[lead + [c for c in by_fold.columns if c not in lead]]
    return {
        "accuracy": accuracy,
        "accuracy_by_fold": by_fold,
        "comparisons": comparisons,
        "verdict": pd.DataFrame(rows),
        "reversal": pd.DataFrame(reversal),
    }
