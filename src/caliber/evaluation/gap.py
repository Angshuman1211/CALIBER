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
    """

    x: np.ndarray
    y: np.ndarray

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


def isotonic(x: Sequence[float], y: Sequence[float]) -> IsotonicFit:
    """Fit the frozen form. Ties in ``x`` are averaged into one knot, then weighted.

    Averaging ties rather than ordering them arbitrarily matters here: the null-space
    baseline ties four of the nine operators, so its ``x`` has heavy ties by construction
    and an arbitrary tie-break would let the fit read structure out of input order.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return IsotonicFit(np.empty(0), np.empty(0))
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    uniq, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=ys) / counts
    return IsotonicFit(uniq, _pava(means, counts.astype(float)))


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
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, predictor: str
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
    fit = isotonic(x_train, y_train)
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
        pred, outside = _fit_predict(xs[train], ys[train], xs[test], predictor)
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
        }))
    if not rows:
        return pd.DataFrame(
            columns=["fold", "calibration", "deployment", "x", "gap", "prediction",
                     "extrapolated", "n_train"]
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
    scored: pd.DataFrame, protocol: str, predictor: str, scope: str, gap_sd: float
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

    Returns ``accuracy``, ``comparisons`` and ``verdict`` frames. Protocols and baselines
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
    acc, comps, per_fold = [], [], []
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
    accuracy = pd.DataFrame(acc)
    comparisons = pd.DataFrame(comps)
    rows = []
    for (protocol, which), part in comparisons.groupby(["protocol", "scope"], sort=True):
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
    }
