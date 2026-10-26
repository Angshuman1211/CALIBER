"""Does a detector's threshold *transfer* across operators? (T1.4, D73)

**Track 1 owns this module.** It fixes an estimand, which is the protocol's call.

## Why this exists

The posterior-dispersion baseline ranks hallucinations within 0.001--0.012 AUROC of
``s_disagree`` at every operator (T2.6), which reads as "the decomposition buys nothing".
But that comparison is a *detection* comparison, and D67 ruled detection numbers
unreportable on this bank: the label is approximately a function of the score
(``err_free ~ sqrt(s_free^2 + free_content^2)`` at Spearman 0.99), so an AUROC measures
the project's own identity rather than a detector's merit. The baseline comparison was
therefore being held on the one axis the project had already disqualified, and never held
on the axis where the project has a claim at all.

This module asks the other question. Not *which detector ranks better*, but **whose
threshold survives a change of operator**. A detector that ranks hallucinations perfectly
at its calibration operator and loses coverage when the sampling pattern changes is
useless for the thing CALIBER exists to do; one that ranks slightly worse but holds its
promise is the better instrument. Nothing in the dispersion literature measures this,
because operator shift is precisely what that literature assumes away.

## The estimand, and why it is comparable across detectors

Absolute coverage is **not** comparable between two detectors: each has its own score
scale, its own threshold, and its own flag area. What is comparable is the *loss from
transferring*, per detector:

    penalty(d) = coverage(d, calibrated at the deployment operator)
               - coverage(d, calibrated at a different operator)

and the detectors are compared on **|penalty|** (D81), not on the signed value. The signed
form counts over-coverage as transferring well, so a detector whose transferred threshold
flags nearly everything beats one whose threshold is merely right. Both directions are
transfer failures -- one of validity, one of efficiency -- and a transfer comparison has to
charge for both.

Both arms use detector ``d``'s own threshold, fitted by the same split-conformal rule at
the same ``alpha`` on the same calibration patients, and both are scored against the same
label on the same test patients. The score's units cancel in the difference. Each detector
is measured against itself, and only then are the two self-comparisons compared:

    delta = |penalty(baseline)| - |penalty(CALIBER)|

``delta > 0`` means the baseline loses more coverage under shift than CALIBER does -- which
is the decomposition earning its keep on the axis that matters, whatever the AUROCs say.

:func:`detector_transfer` resamples clusters **once** per replicate and evaluates all four
cells (two detectors x two calibrations) on that one resample, so the patient's difficulty
cancels from both penalties *and* from their difference. This is the same reasoning as
:func:`caliber.evaluation.paired.paired_method_gap`, one level further out.

## What it cannot do

It inherits every defect of the label; it just does not *depend* on the label being
independent of the score. Both detectors are scored against the same label, so a label
that is partly a function of CALIBER's score inflates CALIBER's coverage **in both arms
equally** and cancels in the penalty. That cancellation is the reason this comparison is
reportable where the AUROC one is not -- and it is an argument about the estimand, not a
repair of the label, so it does not license reading the AUROCs again.

It also cannot say a detector is *good*; a detector that flags everything has a penalty of
exactly zero and transfers perfectly. :attr:`DetectorTransfer.interpretable` gates on both
arms reaching nominal coverage and on flag area, following D51's rule that an area means
something only once validity is reached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..calibration import apply_threshold
from ..calibration.conformal import split_threshold
from .paired import DEFAULT_N_BOOT, _per_cluster_counts

#: The smallest coverage penalty difference worth claiming, in coverage points.
#:
#: At ``alpha = 0.10`` the whole miss budget is 0.10, so 0.02 is a fifth of it: below that
#: the two detectors transfer equally well for any purpose this project serves. D72's
#: lesson applies -- a test that *rejects* on a difference this small reports precision,
#: not a finding.
#:
#: **This floor is a judgement about what is worth claiming, and it is not a noise floor.**
#: Threshold noise alone put ``|delta|`` at a mean of 0.039 and a maximum of 0.128 on a
#: 10-calibration-patient fixture where nothing drifted, so a magnitude test against this
#: constant would have called pure noise a CALIBER win most of the time. The noise is
#: handled where it belongs, in the interval: :func:`detector_transfer` resamples the
#: calibration patients and refits every threshold per replicate, so the interval carries
#: the threshold's own variability and spans zero when there is nothing there. Both gates
#: must pass (:attr:`TransferComparison.verdict`); neither is sufficient alone.
PENALTY_FLOOR = 0.02

#: A detector that flags this fraction of all regions or more is reaching nominal coverage
#: by flagging everything, so its zero penalty is an artefact of the trivial solution.
#: Deliberately loose -- it exists to catch the degenerate case, not to rank efficiency.
TRIVIAL_FLAG_AREA = 0.90


@dataclass(frozen=True)
class DetectorTransfer:
    """One detector's coverage penalty for a transferred threshold, at one operator.

    **A single penalty is not evidence of transfer failure.** The two thresholds are order
    statistics of different samples of a few dozen positives, so the difference between
    them carries that noise even when nothing has shifted: on a no-drift fixture the
    penalty is centred on zero but reaches 0.05 at 10 calibration patients, falling to
    0.014 at 100. What the noise does *not* survive is the difference between two
    detectors measured on the same patients, which is why :func:`detector_transfer` is
    the reportable quantity and this is its ingredient.
    """

    detector: str
    calibration_operator: str
    deployment_operator: str
    coverage_group: float
    coverage_naive: float
    coverage_group_lo: float
    coverage_group_hi: float
    penalty: float
    lo: float
    hi: float
    se_paired: float
    se_unpaired: float
    flag_area_group: float
    flag_area_naive: float
    n_clusters: int
    n_positive: int
    nominal: float

    @property
    def reaches_nominal(self) -> bool:
        """Is the in-group arm calibrated? If not, the penalty has no reference point.

        Tested on the **interval**, not the point estimate, which is D-056's rule in
        ``coverage_by_patient.py``. Split conformal promises nominal coverage in
        expectation; a single realisation on ~40 patients lands either side of it. The
        point-estimate version of this gate rejected 42% of a fixture built to be
        perfectly calibrated, which would have discarded most real cells as uncomparable.
        """
        return bool(np.isfinite(self.coverage_group_hi) and self.coverage_group_hi >= self.nominal)

    @property
    def trivial(self) -> bool:
        """Does the in-group arm reach coverage by flagging almost everything?"""
        return bool(self.flag_area_group >= TRIVIAL_FLAG_AREA)

    @property
    def abs_penalty(self) -> float:
        """Coverage lost in **either** direction. The quantity to compare on (D81).

        The signed penalty treats over-coverage as transferring well, and a detector whose
        transferred threshold flags nearly everything then scores *better* than one whose
        threshold is merely right: coverage 1.000 gives a negative penalty. Track 2
        measured the consequence -- on CT, CALIBER's transferred arm flags >= 99.7% of all
        regions at 6 of 8 operators and the signed verdict reads that as a win.

        A transferred threshold that over-flags has failed, in efficiency rather than in
        validity, and a *transfer* comparison should count both failures.
        """
        return float(abs(self.penalty))

    @property
    def transferred_over_flags(self) -> bool:
        """Does the transferred arm reach its coverage by flagging almost everything?

        The gap that let the above through: :attr:`trivial` guards only the in-group arm,
        so the transferred arm's flag area never entered the verdict. It is D51's rule --
        an area means nothing until validity is reached -- applied to the arm D51 was not
        looking at.
        """
        return bool(self.flag_area_naive >= TRIVIAL_FLAG_AREA)

    @property
    def interpretable(self) -> bool:
        return self.reaches_nominal and not self.trivial

    def as_row(self) -> dict[str, object]:
        return {
            "detector": self.detector,
            "calibrated_at": self.calibration_operator,
            "deployed_at": self.deployment_operator,
            "coverage_group": self.coverage_group,
            "coverage_group_lo": self.coverage_group_lo,
            "coverage_group_hi": self.coverage_group_hi,
            "coverage_naive": self.coverage_naive,
            "penalty": self.penalty,
            "lo": self.lo,
            "hi": self.hi,
            "se_paired": self.se_paired,
            "area_group": self.flag_area_group,
            "area_naive": self.flag_area_naive,
            "reaches_nominal": self.reaches_nominal,
            "interpretable": self.interpretable,
            "n_clusters": self.n_clusters,
            "n_positive": self.n_positive,
        }


@dataclass(frozen=True)
class TransferComparison:
    """Two detectors' penalties at one operator, and the paired difference between them."""

    baseline: DetectorTransfer
    caliber: DetectorTransfer
    delta: float
    lo: float
    hi: float
    se_paired: float
    se_unpaired: float
    floor: float

    @property
    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi)
                    and (self.lo > 0 or self.hi < 0))

    @property
    def exceeds_floor(self) -> bool:
        return bool(np.isfinite(self.delta) and abs(self.delta) > self.floor)

    @property
    def half_width(self) -> float:
        """Half the interval. Equivalence is unreachable below this, whatever the estimate."""
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            return float("nan")
        return float((self.hi - self.lo) / 2.0)

    @property
    def equivalent(self) -> bool:
        """Does the **whole interval** lie inside the floor? The only honest 'alike'.

        Not ``abs(delta) <= floor``. A point estimate of 0.015 with an interval of
        [-0.14, +0.17] is not evidence that two detectors transfer alike; it is no
        evidence either way. Reading the point estimate as equivalence is the project's
        recurring failure mode -- a statistic returning the hoped-for number -- pointed at
        the hope that the baseline comparison comes out level.
        """
        return bool(
            np.isfinite(self.lo) and np.isfinite(self.hi)
            and self.lo > -self.floor and self.hi < self.floor
        )

    @property
    def interpretable(self) -> bool:
        """Both detectors must be calibrated in-group for the difference to mean anything."""
        return self.baseline.interpretable and self.caliber.interpretable

    @property
    def over_flagging(self) -> tuple[str, ...]:
        """Detectors whose *transferred* arm flags almost everything (D81).

        Reported rather than silently excluded: a transferred threshold that over-flags is
        a real transfer failure and ``|penalty|`` already charges for it. This names it so
        a reader can see *how* the detector failed, since under- and over-coverage have
        the same magnitude here and very different meanings.
        """
        return tuple(
            d.detector for d in (self.baseline, self.caliber) if d.transferred_over_flags
        )

    @property
    def floor_for_equivalence(self) -> float:
        """The smallest floor at which this interval would support 'alike'.

        Reported because it is the actionable number when the answer is ``undetermined``:
        it says whether the gap to a conclusion is a few more patients or an order of
        magnitude more.
        """
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            return float("nan")
        return float(max(abs(self.lo), abs(self.hi)))

    @property
    def verdict(self) -> str:
        """``caliber_transfers_better``, ``baseline_transfers_better``, ``equivalent``,
        ``winner_over_flags``, ``not_interpretable``, or ``undetermined``.

        A directional verdict needs both gates: a magnitude above the floor (D72 -- a
        difference below it is precision, not a finding) *and* an interval excluding zero.
        ``equivalent`` needs the interval inside the floor. Everything else is
        ``undetermined``, which on a bank of this size is the common and correct answer,
        and :attr:`floor_for_equivalence` says what it would take to leave it.
        """
        if not self.interpretable:
            return "not_interpretable"
        if not np.isfinite(self.delta):
            return "undetermined"
        if self.equivalent:
            return "equivalent"
        if self.exceeds_floor and self.excludes_zero:
            # |penalty| charges over-coverage, but it cannot charge it *enough*: a
            # transferred threshold that over-flags moves coverage from ~0.90 to at most
            # 1.000, so it is charged ~0.10 however badly it flags, while under-covering
            # can cost the full 0.90. So a detector that wins by flagging everything can
            # still come out ahead on |penalty| alone. It is not a transfer result, and
            # the comparison it needs is on flag area at matched coverage, which is a
            # different estimand -- `area_at_matched_coverage`.
            winner = self.caliber if self.delta > 0 else self.baseline
            if winner.transferred_over_flags:
                return "winner_over_flags"
            return "caliber_transfers_better" if self.delta > 0 else "baseline_transfers_better"
        return "undetermined"

    def as_row(self) -> dict[str, object]:
        return {
            "deployed_at": self.caliber.deployment_operator,
            "calibrated_at": self.caliber.calibration_operator,
            "penalty_baseline": self.baseline.penalty,
            "penalty_caliber": self.caliber.penalty,
            "abs_penalty_baseline": self.baseline.abs_penalty,
            "abs_penalty_caliber": self.caliber.abs_penalty,
            "area_naive_baseline": self.baseline.flag_area_naive,
            "area_naive_caliber": self.caliber.flag_area_naive,
            "over_flagging": ",".join(self.over_flagging),
            "delta": self.delta,
            "lo": self.lo,
            "hi": self.hi,
            "se_paired": self.se_paired,
            "verdict": self.verdict,
            "interpretable": self.interpretable,
            "floor_for_equivalence": self.floor_for_equivalence,
            "n_clusters": self.caliber.n_clusters,
        }


def _thresholds(
    calib: pd.DataFrame,
    score_col: str,
    calibration_operator: str,
    deployment_operator: str,
    operator_col: str,
    label_col: str,
    alpha: float,
) -> tuple[float, float]:
    """``(naive, group)`` thresholds for one detector: fitted away, and fitted in place."""
    ops = calib[operator_col].astype(str)
    away = calib[ops == str(calibration_operator)]
    here = calib[ops == str(deployment_operator)]
    if away.empty:
        raise KeyError(
            f"no calibration-split rows at {calibration_operator!r}; the threshold to "
            "transfer cannot be fitted"
        )
    if here.empty:
        raise KeyError(
            f"no calibration-split rows at {deployment_operator!r}; the group-conditional "
            "threshold cannot be fitted, so there is no reference to lose coverage against"
        )
    naive = split_threshold(away[score_col], away[label_col], alpha=alpha)
    group = split_threshold(here[score_col], here[label_col], alpha=alpha)
    return float(naive.threshold), float(group.threshold)


def _positives_by_cluster(
    frame: pd.DataFrame, cluster_col: str, label_col: str, score_col: str, keys: np.ndarray
) -> list[np.ndarray]:
    """Each cluster's positive-region scores, sorted. The unit the bootstrap resamples."""
    positives = frame[frame[label_col].astype(bool)]
    by_cluster = {
        key: np.sort(part[score_col].to_numpy(dtype=float))
        for key, part in positives.groupby(cluster_col, observed=True)
    }
    empty = np.empty(0, dtype=float)
    return [by_cluster.get(k, empty) for k in keys]


def _tau_replicates(
    per_cluster: list[np.ndarray], idx: np.ndarray, alpha: float
) -> np.ndarray:
    """One split-conformal threshold per replicate of the **calibration** clusters.

    This is the term the first version of this module left out. Holding the threshold
    fixed while resampling only the test side produces an interval that measures how the
    test patients vary and says nothing about how the threshold itself would have varied
    on another calibration draw -- and the threshold is a single order statistic of a few
    dozen positives, so that is the larger term. An interval that ignores it reads as
    tight and is not.

    :func:`caliber.calibration.conformal.split_threshold` is called rather than
    reimplemented, so the flooring rule and the ``-inf`` for an uncertifiable group are
    identical to the ones the point estimate uses.
    """
    taus = np.empty(idx.shape[0], dtype=float)
    for b in range(idx.shape[0]):
        drawn = [per_cluster[c] for c in idx[b]]
        pooled = np.concatenate(drawn) if drawn else np.empty(0, dtype=float)
        taus[b] = split_threshold(pooled, np.ones(pooled.size, dtype=bool), alpha).threshold
    return taus


def _caught_matrix(per_cluster: list[np.ndarray], taus: np.ndarray) -> np.ndarray:
    """``(n_boot, n_clusters)`` positives caught, at each replicate's own threshold.

    ``apply_threshold`` flags on ``score >= tau``, so the count at or above ``tau`` is
    ``size - searchsorted(sorted_scores, tau, side="left")``.
    """
    out = np.zeros((taus.size, len(per_cluster)), dtype=float)
    for c, scores in enumerate(per_cluster):
        if scores.size:
            out[:, c] = scores.size - np.searchsorted(scores, taus, side="left")
    return out


def _coverage_replicates(
    caught: np.ndarray, positives: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    """Coverage per replicate: caught over positives, pooled across resampled clusters."""
    rows = np.arange(caught.shape[0])[:, None]
    num = caught[rows, test_idx].sum(axis=1)
    den = np.maximum(positives[test_idx].sum(axis=1), 1.0)
    return num / den


def _check_frames(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    score_cols: tuple[str, ...],
    label_col: str,
    cluster_col: str,
    operator_col: str,
) -> None:
    """The guards both entry points share, including the one that has bitten twice."""
    for frame, what in ((calib, "calibration"), (test, "test")):
        for col in (*score_cols, label_col, cluster_col, operator_col):
            if col not in frame.columns:
                raise KeyError(f"the {what} frame has no {col!r} column")
    shared = set(calib[cluster_col].unique()) & set(test[cluster_col].unique())
    if shared:
        raise ValueError(
            f"{len(shared)} cluster(s) appear in both the calibration and test frames "
            f"(e.g. {sorted(map(str, shared))[:3]}); a transferred threshold scored on its "
            "own patients is in-sample and will read as valid when it is not"
        )


@dataclass(frozen=True)
class _Arms:
    """One detector's two arms at one operator: replicates and point estimates."""

    penalty_reps: np.ndarray
    coverage_group_reps: np.ndarray
    coverage_group: float
    coverage_naive: float
    penalty: float
    flag_area_group: float
    flag_area_naive: float
    n_clusters: int
    n_positive: int


def _arms(
    calib: pd.DataFrame,
    here: pd.DataFrame,
    score_col: str,
    calibration_operator: str,
    deployment_operator: str,
    operator_col: str,
    label_col: str,
    cluster_col: str,
    alpha: float,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    test_keys: np.ndarray,
) -> _Arms:
    """Both arms of one detector, on the supplied resamples of each side.

    The resample indices are passed in rather than drawn here, which is what makes the
    comparison in :func:`detector_transfer` paired: two detectors evaluated on the same
    ``calib_idx`` and ``test_idx`` see the same patients in the same replicate.
    """
    naive_tau, group_tau = _thresholds(
        calib, score_col, calibration_operator, deployment_operator,
        operator_col, label_col, alpha,
    )
    flagged = apply_threshold(here, naive_tau, score_col, out_col="__flag_naive")
    flagged = apply_threshold(flagged, group_tau, score_col, out_col="__flag_group")
    caught_g, positives = _per_cluster_counts(
        flagged, cluster_col, "__flag_group", label_col, test_keys
    )
    caught_n, _ = _per_cluster_counts(
        flagged, cluster_col, "__flag_naive", label_col, test_keys
    )
    total = positives.sum()
    areas = (float(flagged["__flag_group"].mean()), float(flagged["__flag_naive"].mean()))
    if total == 0:
        nan = float("nan")
        empty = np.full(calib_idx.shape[0], nan)
        return _Arms(empty, empty, nan, nan, nan, areas[0], areas[1], int(test_keys.size), 0)

    ops = calib[operator_col].astype(str)
    away_pos = _positives_by_cluster(
        calib[ops == str(calibration_operator)], cluster_col, label_col, score_col,
        _calib_keys(calib, cluster_col),
    )
    here_pos = _positives_by_cluster(
        calib[ops == str(deployment_operator)], cluster_col, label_col, score_col,
        _calib_keys(calib, cluster_col),
    )
    taus_naive = _tau_replicates(away_pos, calib_idx, alpha)
    taus_group = _tau_replicates(here_pos, calib_idx, alpha)

    test_pos = _positives_by_cluster(flagged, cluster_col, label_col, score_col, test_keys)
    cov_naive = _coverage_replicates(_caught_matrix(test_pos, taus_naive), positives, test_idx)
    cov_group = _coverage_replicates(_caught_matrix(test_pos, taus_group), positives, test_idx)
    return _Arms(
        penalty_reps=cov_group - cov_naive,
        coverage_group_reps=cov_group,
        coverage_group=float(caught_g.sum() / total),
        coverage_naive=float(caught_n.sum() / total),
        penalty=float((caught_g.sum() - caught_n.sum()) / total),
        flag_area_group=areas[0],
        flag_area_naive=areas[1],
        n_clusters=int(test_keys.size),
        n_positive=int(total),
    )


def _calib_keys(calib: pd.DataFrame, cluster_col: str) -> np.ndarray:
    return np.array(sorted(calib[cluster_col].unique()), dtype=object)


def _resamples(
    calib: pd.DataFrame, test_keys: np.ndarray, cluster_col: str, n_boot: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One resample of the calibration clusters and one of the test clusters per replicate."""
    rng = np.random.default_rng(seed)
    calib_keys = _calib_keys(calib, cluster_col)
    return (
        rng.integers(0, calib_keys.size, size=(n_boot, calib_keys.size)),
        rng.integers(0, test_keys.size, size=(n_boot, test_keys.size)),
    )


def detector_penalty(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    score_col: str,
    calibration_operator: str,
    deployment_operator: str,
    detector_name: str | None = None,
    operator_col: str = "operator_id",
    label_col: str = "label",
    cluster_col: str = "patient_id",
    alpha: float = 0.10,
    n_boot: int = DEFAULT_N_BOOT,
    interval_alpha: float = 0.05,
    seed: int = 0,
) -> DetectorTransfer:
    """Coverage lost by transferring ``score_col``'s threshold to ``deployment_operator``.

    Both sides are resampled: the calibration clusters, which refits the two thresholds,
    and the test clusters, which re-evaluates them. See :func:`_tau_replicates` for why
    leaving the first out produces an interval that looks tight and is not.

    ``calib`` and ``test`` must be disjoint on ``cluster_col``; this is asserted rather
    than assumed, because the bug that has recurred most in this project is a calibration
    frame sharing patients with the frame it is scored on (P-23, and again in
    ``tool2_holdout.py``), and it shows up as coverage landing implausibly near nominal.
    """
    name = detector_name or score_col
    _check_frames(calib, test, (score_col,), label_col, cluster_col, operator_col)
    here = test[test[operator_col].astype(str) == str(deployment_operator)]
    if here.empty:
        raise KeyError(f"no test-split rows at {deployment_operator!r}")

    test_keys = np.array(sorted(here[cluster_col].unique()), dtype=object)
    calib_idx, test_idx = _resamples(calib, test_keys, cluster_col, n_boot, seed)
    arms = _arms(
        calib, here, score_col, calibration_operator, deployment_operator,
        operator_col, label_col, cluster_col, alpha, calib_idx, test_idx, test_keys,
    )
    nominal = 1.0 - alpha
    if arms.n_positive == 0:
        nan = float("nan")
        return DetectorTransfer(
            name, str(calibration_operator), str(deployment_operator), nan, nan, nan, nan,
            nan, nan, nan, nan, nan, arms.flag_area_group, arms.flag_area_naive,
            arms.n_clusters, 0, nominal,
        )

    lo, hi = np.quantile(arms.penalty_reps, [interval_alpha / 2, 1 - interval_alpha / 2])
    cov_lo, cov_hi = np.quantile(
        arms.coverage_group_reps, [interval_alpha / 2, 1 - interval_alpha / 2]
    )
    # The same estimand with the pairing deliberately broken, so the gain is measured.
    unpaired_calib, unpaired_test = _resamples(
        calib, test_keys, cluster_col, n_boot, seed + 1
    )
    broken = _arms(
        calib, here, score_col, calibration_operator, deployment_operator,
        operator_col, label_col, cluster_col, alpha, unpaired_calib, unpaired_test, test_keys,
    )
    return DetectorTransfer(
        detector=name,
        calibration_operator=str(calibration_operator),
        deployment_operator=str(deployment_operator),
        coverage_group=arms.coverage_group,
        coverage_naive=arms.coverage_naive,
        coverage_group_lo=float(cov_lo),
        coverage_group_hi=float(cov_hi),
        penalty=arms.penalty,
        lo=float(lo),
        hi=float(hi),
        se_paired=float(np.nanstd(arms.penalty_reps, ddof=1)),
        se_unpaired=float(np.nanstd(broken.penalty_reps, ddof=1)),
        flag_area_group=arms.flag_area_group,
        flag_area_naive=arms.flag_area_naive,
        n_clusters=arms.n_clusters,
        n_positive=arms.n_positive,
        nominal=nominal,
    )


def detector_transfer(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    baseline_col: str,
    caliber_col: str,
    calibration_operator: str,
    deployment_operator: str,
    baseline_name: str | None = None,
    caliber_name: str | None = None,
    operator_col: str = "operator_id",
    label_col: str = "label",
    cluster_col: str = "patient_id",
    alpha: float = 0.10,
    n_boot: int = DEFAULT_N_BOOT,
    interval_alpha: float = 0.05,
    seed: int = 0,
    floor: float = PENALTY_FLOOR,
) -> TransferComparison:
    """The difference of the two detectors' transfer penalties, paired on one resample.

    Four cells -- two detectors, each with a transferred and an in-group threshold -- are
    evaluated on **one** bootstrap resample of clusters per replicate. The patient effect
    cancels from each penalty and again from their difference, which is what makes a
    difference-of-differences on ~40 patients worth reporting at all.

    The label is shared by both detectors, so ``positives`` is one array; only the caught
    counts differ. That is also the formal reason the label's dependence on CALIBER's score
    does not favour CALIBER here: it moves both of CALIBER's arms together.
    """
    _check_frames(
        calib, test, (baseline_col, caliber_col), label_col, cluster_col, operator_col
    )
    here = test[test[operator_col].astype(str) == str(deployment_operator)]
    if here.empty:
        raise KeyError(f"no test-split rows at {deployment_operator!r}")

    kwargs = dict(
        operator_col=operator_col, label_col=label_col, cluster_col=cluster_col,
        alpha=alpha, n_boot=n_boot, interval_alpha=interval_alpha, seed=seed,
    )
    base = detector_penalty(
        calib, test, baseline_col, calibration_operator, deployment_operator,
        detector_name=baseline_name or baseline_col, **kwargs,
    )
    cal = detector_penalty(
        calib, test, caliber_col, calibration_operator, deployment_operator,
        detector_name=caliber_name or caliber_col, **kwargs,
    )
    if base.n_positive == 0 or cal.n_positive == 0:
        nan = float("nan")
        return TransferComparison(base, cal, nan, nan, nan, nan, nan, floor)

    # One resample of each side, shared by all four cells: this is the pairing. Both
    # detectors see the same calibration patients (so the same threshold noise) and the
    # same test patients (so the same anatomy) within a replicate, and both cancel from
    # the difference of their penalties.
    test_keys = np.array(sorted(here[cluster_col].unique()), dtype=object)
    shared = dict(
        operator_col=operator_col, label_col=label_col, cluster_col=cluster_col, alpha=alpha,
    )
    # The comparison is on |penalty| (D81), and the absolute value is taken **inside** the
    # bootstrap, per replicate, before differencing. Taking it outside -- on the two point
    # estimates alone -- would give an interval for the signed difference wearing the
    # absolute one's label, which is the same class of error as the defect it fixes.
    calib_idx, test_idx = _resamples(calib, test_keys, cluster_col, n_boot, seed)
    paired = (
        np.abs(_arms(calib, here, baseline_col, calibration_operator, deployment_operator,
                     calib_idx=calib_idx, test_idx=test_idx, test_keys=test_keys,
                     **shared).penalty_reps)
        - np.abs(_arms(calib, here, caliber_col, calibration_operator, deployment_operator,
                       calib_idx=calib_idx, test_idx=test_idx, test_keys=test_keys,
                       **shared).penalty_reps)
    )
    idx_b = _resamples(calib, test_keys, cluster_col, n_boot, seed + 1)
    idx_c = _resamples(calib, test_keys, cluster_col, n_boot, seed + 2)
    unpaired = (
        np.abs(_arms(calib, here, baseline_col, calibration_operator, deployment_operator,
                     calib_idx=idx_b[0], test_idx=idx_b[1], test_keys=test_keys,
                     **shared).penalty_reps)
        - np.abs(_arms(calib, here, caliber_col, calibration_operator, deployment_operator,
                       calib_idx=idx_c[0], test_idx=idx_c[1], test_keys=test_keys,
                       **shared).penalty_reps)
    )
    lo, hi = np.quantile(paired, [interval_alpha / 2, 1 - interval_alpha / 2])
    return TransferComparison(
        baseline=base,
        caliber=cal,
        delta=float(base.abs_penalty - cal.abs_penalty),
        lo=float(lo),
        hi=float(hi),
        se_paired=float(np.nanstd(paired, ddof=1)),
        se_unpaired=float(np.nanstd(unpaired, ddof=1)),
        floor=floor,
    )


def transfer_table(
    df: pd.DataFrame,
    baseline_col: str,
    caliber_col: str,
    calibration_operator: str,
    split_col: str = "split",
    operator_col: str = "operator_id",
    cluster_col: str = "patient_id",
    label_col: str = "label",
    alpha: float = 0.10,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    floor: float = PENALTY_FLOOR,
    baseline_name: str | None = None,
    caliber_name: str | None = None,
) -> tuple[pd.DataFrame, list[TransferComparison]]:
    """One comparison per deployment operator, never pooled across them.

    Pooling is the specific error that made a 0.499 "exactly chance" out of two opposite
    per-operator biases (D60, retracted). Every row here is one deployment operator, and
    the summary counts rows rather than averaging them.
    """
    if split_col not in df.columns:
        raise KeyError(f"bank has no {split_col!r} column")
    calib = df[df[split_col] == "calib"]
    test = df[df[split_col] == "test"]
    if calib.empty or test.empty:
        raise ValueError(
            "this bank has no calib/test split, so there is no out-of-sample side to "
            "measure transfer on"
        )

    results, rows = [], []
    for oid in sorted(test[operator_col].astype(str).unique()):
        if oid == str(calibration_operator):
            continue
        try:
            comparison = detector_transfer(
                calib, test, baseline_col, caliber_col, calibration_operator, oid,
                baseline_name=baseline_name, caliber_name=caliber_name,
                operator_col=operator_col, label_col=label_col, cluster_col=cluster_col,
                alpha=alpha, n_boot=n_boot, seed=seed, floor=floor,
            )
        except KeyError:
            continue
        results.append(comparison)
        rows.append(comparison.as_row())
    if not rows:
        raise ValueError(
            f"no deployment operator other than {calibration_operator!r} has both a "
            "calibration and a test side in this bank"
        )
    return pd.DataFrame(rows), results


def area_at_matched_coverage(
    frame: pd.DataFrame,
    score_cols: dict[str, str],
    target: float,
    label_col: str = "label",
) -> pd.DataFrame:
    """Flag area each detector needs to reach the **same** coverage. The efficiency axis.

    D39 ruled that detectors are compared on flag area rather than power, because a
    conformal threshold pins the miss rate -- so area is the only axis left. D51 then added
    the precondition: an area means nothing unless the arm is valid, because flagging
    little is cheap once you have stopped catching anything. The nearest-operator arm
    posted the smallest area in the project at coverage 0.547.

    This closes that loop by construction. Rather than fixing the threshold and comparing
    coverage, it fixes **coverage** and compares area: for each detector, the smallest
    flag area that still catches ``target`` of the positives. Both numbers are then
    answering one question -- at equal reliability, who flags less? -- and the degenerate
    arm cannot win, because reaching the target is what it is being charged for.

    Independent of any AUROC, so it stays available under D67: it is a statement about
    operating points, not about ranking.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target coverage must be in (0, 1], got {target}")
    if label_col not in frame.columns:
        raise KeyError(f"frame has no {label_col!r} column")
    labels = frame[label_col].to_numpy().astype(bool)
    n_positive = int(labels.sum())
    rows = []
    for name, col in score_cols.items():
        if col not in frame.columns:
            raise KeyError(f"frame has no {col!r} column")
        scores = frame[col].to_numpy(dtype=float)
        row: dict[str, object] = {"detector": name, "n_positive": n_positive,
                                  "target_coverage": float(target)}
        if n_positive == 0:
            rows.append({**row, "threshold": float("nan"), "flag_area": float("nan"),
                         "coverage": float("nan"), "reached": False})
            continue
        # The threshold that catches exactly ceil(target * n_positive) positives is the
        # k-th largest positive score, so the area follows from counting all regions at or
        # above it -- positives and negatives alike.
        needed = int(np.ceil(target * n_positive))
        positive_scores = np.sort(scores[labels])[::-1]
        tau = float(positive_scores[min(needed, n_positive) - 1])
        flagged = scores >= tau
        rows.append({
            **row,
            "threshold": tau,
            "flag_area": float(flagged.mean()),
            "coverage": float((flagged & labels).sum() / n_positive),
            "reached": True,
        })
    return pd.DataFrame(rows)


def transfer_verdict(results: list[TransferComparison]) -> dict[str, object]:
    """Count the per-operator verdicts. Counts, not a mean penalty difference.

    A mean over operators would be the pooling D60 was retracted for: two operators with
    opposite signs average to "no effect" and that average describes neither.
    """
    usable = [r for r in results if r.interpretable]
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    wins = counts.get("caliber_transfers_better", 0)
    losses = counts.get("baseline_transfers_better", 0)
    alike = counts.get("equivalent", 0)
    # `winner_over_flags` is a refusal, not a result, so it joins the undetermined count
    # rather than either side's. Counting it as a win is exactly the defect D81 fixes.
    undetermined = counts.get("undetermined", 0) + counts.get("winner_over_flags", 0)
    needed = [r.floor_for_equivalence for r in usable if np.isfinite(r.floor_for_equivalence)]
    floor_needed = float(max(needed)) if needed else float("nan")

    if not usable:
        headline = ("no operator has both detectors calibrated in-group; nothing is "
                    "comparable here")
    elif wins and not losses:
        headline = (f"CALIBER's threshold transfers better at {wins} of {len(usable)} "
                    "interpretable operators, and worse at none")
    elif losses and not wins:
        headline = (f"the baseline's threshold transfers better at {losses} of "
                    f"{len(usable)} interpretable operators")
    elif wins and losses:
        headline = (f"split: CALIBER better at {wins}, baseline better at {losses} of "
                    f"{len(usable)} -- report per operator, do not average")
    elif alike == len(usable):
        headline = (f"the two transfer alike at all {len(usable)} interpretable operators, "
                    "with the interval inside the floor: the decomposition buys no "
                    "transfer advantage, and that is measured rather than assumed")
    else:
        headline = (f"undetermined at {undetermined} of {len(usable)} interpretable "
                    f"operators: no difference is placed and none is excluded. The "
                    f"interval would have to reach {floor_needed:.3f} to call them alike")
    return {
        "n_operators": len(results),
        "n_interpretable": len(usable),
        "counts": counts,
        "caliber_better": wins,
        "baseline_better": losses,
        "equivalent": alike,
        "undetermined": undetermined,
        "floor_for_equivalence": floor_needed,
        "headline": headline,
    }
