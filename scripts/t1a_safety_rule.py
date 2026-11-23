"""T1-A and T1-D on a finished bank: the ordered-pair safety rule, and the worst cell.

    python scripts/t1a_safety_rule.py banks/<ct_bank> \
        --distances runs/t1a_ct_distances/distances.csv \
        --out runs/t1a_safety

Three things happen, in this order, and the order is the point.

**The matched-operator control runs first and gates everything after it.** For each
operator the threshold is fitted on that operator's own calibration-split patients and
scored on its own test-split patients. Split conformal says what that coverage must look
like -- ``Beta(n + 1 - k, k)``, with no free parameter -- so a cell outside its own Beta
interval means the bank is wrong, not that the method failed, and nothing downstream is
worth reading. The work plan's Sec. 3 asks for this check before any result from the bank
is interpreted; here it is a hard gate, and ``--ignore-control`` is the only way past it.
The gate's level is **family-wise over the matched cells** (:data:`CONTROL_FAMILY_LEVEL`),
Bonferroni-corrected per cell; before D89 the 98% was applied per cell to nine cells, which
is a 17% chance of failing a sound bank. The uncorrected band is printed beside it.

**Then the 72 ordered pairs.** Threshold fitted at the calibration operator, coverage
measured at the deployment operator, on disjoint patients. Joined to the signed operator
distance, which is a property of the two operators alone and so was computed before this
bank existed.

**Then the label, read over patient re-cuts rather than one draw (D86).** Split conformal
promises nominal coverage marginally over the draw of the calibration set, and T2-A1
measured what one draw costs: on T2-A's bank the registry split sits a near-constant 1.38 SD
low at all nine matched operators at once. That displacement is *shared* by the cells, so it
does not average out over the 72 pairs -- a single-draw label is biased, not merely noisy.
The bank holds every subject's scores at every operator, so this script makes its own
re-cuts (``--n-splits``, registry partition first) and reads the label from their mean, with
the interval widened for the overlap between re-cuts. The single-draw arm is printed beside
it every time, because the frozen choice is the one that *flatters* the rule.

**Then the rule, pooled and per family cell.** Every choice -- the threshold at zero, the
tolerance at zero, the permutation statistic, the resampling unit, which split the label
comes from -- comes from :mod:`caliber.evaluation.prereg` and none is a command-line option,
so no number printed here can have been obtained by picking a cut-point after seeing the
coverages.

Scale handling, which is easy to get wrong: the score's component scales are estimated once
on ``--scale-operator`` and frozen (D26/D27). Re-estimating them per deployment operator
pulls the score distributions towards each other and shrinks the very coverage gap this
analysis measures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from caliber.calibration.conformal import split_threshold
from caliber.data.splits import PROBE_SALTS, calib_sets_for_salts, split_for
from caliber.evaluation import add_label_args, derive_score_and_label, label_from_args, prereg
from caliber.evaluation.reliability import (
    across_split_columns,
    beta_reference,
    clustered_coverage,
    mean_pairwise_overlap,
    pair_coverage_from_splits,
    worst_operator_coverage,
)  # noqa: F401  -- beta_reference is reported beside the bootstrap as the idealisation
from caliber.evaluation.gap import gap_sign_from_thresholds
from caliber.evaluation.safety import (
    label_source_sensitivity,
    safety_by_family,
    safety_table,
    tolerance_sensitivity,
)
from caliber.reportability import COVERAGE, ClaimNotCarried, describe, require_claim, stamp
from caliber.scorebank import read_bank

#: Default anchor for the frozen score scales, and the CT bank of record's calibration
#: operator: the least aggressive geometry on the grid.
SCALE_OPERATOR = "ct.sparse_view.v180.a180.dfull"

#: Interval level for the matched-operator control. 0.98 rather than 0.95 because this is a
#: gate on roughly nine cells at once, and a 95% band would fail one cell in twenty by
#: construction and stop a correct bank.
CONTROL_LEVEL = 0.98

#: **The level the gate actually controls, over all matched cells at once.**
#:
#: Corrected 21 November 2026 (D89). :data:`CONTROL_LEVEL` was applied per cell to nine
#: cells, which leaves a family-wise spurious-failure rate of 1 - 0.98**9 = 17%, not 2%.
#: The docstring above it had already identified multiplicity as the thing to correct for
#: ("a 95% band would fail one cell in twenty") and then under-solved it by a factor of
#: nine. On T2-A's CT bank the gate duly failed one cell, limited-angle 90 deg, by 0.004 at
#: the upper end of its interval -- which is what a 17% family-wise rate looks like when it
#: fires.
#:
#: The per-cell level is now Bonferroni-derived from the number of checkable cells, so the
#: stated level means what it says however many operators the bank has. The uncorrected
#: interval stays in the output beside it: this correction makes the gate HARDER TO FAIL,
#: which is the self-interested direction, so it must not also make the old number
#: invisible. The cost is real and is the point of printing both -- a gate at 99.8% per
#: cell has less power against a single genuinely broken operator than one at 98%.
CONTROL_FAMILY_LEVEL = 0.98


def control_cell_level(n_cells: int, family_level: float = CONTROL_FAMILY_LEVEL) -> float:
    """Bonferroni per-cell level for a gate that tests ``n_cells`` cells simultaneously."""
    if n_cells < 1:
        return family_level
    return 1.0 - (1.0 - family_level) / float(n_cells)

#: Fewest clusters per side before the control's bootstrap interval is believed.
MIN_CLUSTERS_FOR_CONTROL = 8

#: Interval level for a single ordered pair's coverage. 0.95, not the control's 0.98: the
#: control is a simultaneous gate over nine cells, whereas this feeds a per-pair label and
#: a wider interval there would forgive genuine under-coverage.
PAIR_LEVEL = 0.95


def _load_distances(path: str, measure: str) -> dict[tuple[str, str], float]:
    """``(a, b) -> signed distance``, with the antisymmetric fill for the reverse order.

    ``null_space_growth`` is exactly antisymmetric -- swapping the operators swaps "newly
    hidden" for "newly revealed" on the same seeded probes -- so one direction per pair is
    enough and the other is its negation. Filling it here rather than computing it saves
    half of a day-long CPU job.
    """
    out: dict[tuple[str, str], float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["measure"] != measure:
                continue
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            a, b = row["calibration"], row["deployment"]
            out[(a, b)] = value
            out.setdefault((b, a), -value)
    if not out:
        raise SystemExit(f"no finite {measure!r} rows in {path}")
    return out


def _provenance(bank: str) -> dict:
    """The bank manifest's provenance block, or an empty one for a bare parquet."""
    root = Path(bank)
    manifest = root / "manifest.json" if root.is_dir() else root.parent / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("provenance") or {}
    except (OSError, ValueError):
        return {}


def _coverage(positives: pd.DataFrame, tau: float) -> tuple[int, int]:
    """``(caught, total)`` among hallucinated regions at threshold ``tau``."""
    scores = positives["score"].to_numpy(dtype=float)
    return int((scores >= tau).sum()), int(scores.size)


def _matched_control(
    df: pd.DataFrame, alpha: float, cluster_col: str = "sample_id", n_boot: int = 2000
) -> pd.DataFrame:
    """Each operator calibrated and scored on itself. The bank's own sanity check.

    The ``status`` column has three values, and keeping them apart matters more than it
    looks. ``pass`` and ``fail`` are statements about the bank. ``not_checkable`` is a
    statement about this script: too few calibration clusters for a bootstrap to mean
    anything, so the cell is neither evidence for the bank nor against it. Folding that
    third case into ``fail`` would report a thin bank as a broken one and send someone to
    debug a pipeline that is working.

    **The interval is a cluster bootstrap over subjects on both sides, not a Beta band on
    the region count.** The first version of this function used the latter and would have
    failed all nine cells of T2-A's CT bank: 16,400 calibration regions are not 16,400
    independent scores, they are about 35 subjects, so the Beta band came out at +/-0.011
    around nominal and rejected a bank that is fine. The Beta column stays in the output as
    the independence-assumed idealisation, with the design effect beside it, because the
    ratio between them is the interesting number and hiding it was how the mistake
    happened. See D85.
    """
    n_cells = int(df["operator_id"].nunique())
    cell_level = control_cell_level(n_cells)
    rows = []
    for op, part in df.groupby("operator_id", observed=True):
        calib = part[(part["split"] == "calib") & part["label"]]
        test = part[(part["split"] == "test") & part["label"]]
        row: dict[str, object] = {
            "operator_id": str(op),
            "n_calib_positives": int(len(calib)),
            "n_hallucinated": int(len(test)),
            "n_caught": 0,
            "coverage": float("nan"),
            "status": "not_checkable",
            "status_uncorrected": "not_checkable",
            "n_cells_gated": n_cells,
            "cell_level": cell_level,
            "note": "",
        }
        row.update(beta_reference(len(calib), alpha).as_row())
        if calib.empty or test.empty:
            row["note"] = "no calibration or no test positives"
            rows.append(row)
            continue

        res = split_threshold(calib["score"], np.ones(len(calib), bool), alpha=alpha)
        caught, total = _coverage(test, res.threshold)
        row["n_caught"], row["coverage"] = caught, caught / total

        by = lambda frame: [  # noqa: E731
            g["score"].to_numpy(dtype=float) for _, g in frame.groupby(cluster_col, observed=True)
        ]
        # Two intervals from the same replicates (same seed): the gate's corrected one,
        # and the uncorrected per-cell one that the gate used to read, kept visible.
        clustered = clustered_coverage(
            by(calib), by(test), alpha=alpha, n_boot=n_boot, level=cell_level, seed=0
        )
        per_cell = clustered_coverage(
            by(calib), by(test), alpha=alpha, n_boot=n_boot, level=CONTROL_LEVEL, seed=0
        )
        row.update(clustered.as_row())
        row["coverage_lo_uncorrected"] = per_cell.lo
        row["coverage_hi_uncorrected"] = per_cell.hi
        n_clusters = min(clustered.n_clusters_calib, clustered.n_clusters_test)

        if not np.isfinite(clustered.lo):
            row["note"] = (
                f"the bootstrap returned no usable replicate from {n_clusters} cluster(s); "
                "no threshold could be certified on the resamples"
            )
        elif n_clusters < MIN_CLUSTERS_FOR_CONTROL:
            row["note"] = (
                f"{n_clusters} cluster(s) per side is below the floor of "
                f"{MIN_CLUSTERS_FOR_CONTROL}; the interval would be an artefact of the "
                "resampling rather than a measurement"
            )
        else:
            row["status"] = "pass" if clustered.contains_nominal else "fail"
            row["status_uncorrected"] = (
                "pass" if per_cell.contains_nominal else "fail"
            )
            if not clustered.contains_nominal:
                row["note"] = (
                    f"cluster-bootstrap {cell_level:.2%} interval "
                    f"[{clustered.lo:.3f}, {clustered.hi:.3f}] excludes {prereg.NOMINAL:.2f} "
                    f"(design effect {clustered.design_effect:.0f} regions per unit)"
                )
            elif not per_cell.contains_nominal:
                row["note"] = (
                    f"passes the gate at its {CONTROL_FAMILY_LEVEL:.0%} family-wise level "
                    f"but not the uncorrected {CONTROL_LEVEL:.0%} per-cell one "
                    f"[{per_cell.lo:.3f}, {per_cell.hi:.3f}]; with {n_cells} cells the "
                    "uncorrected band fires on a sound bank about 17% of the time"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_pairs(
    df: pd.DataFrame,
    distances: dict[tuple[str, str], float],
    alpha: float,
    cluster_col: str = "sample_id",
    n_boot: int = 2000,
) -> pd.DataFrame:
    """One row per ordered operator pair: transferred threshold, coverage, and its interval.

    **The interval is the label, not decoration** (D85). T1-A asks whether a cell
    under-covers, and per-cell coverage on a real bank has a standard deviation of about
    0.05, so the question cannot be answered from a point estimate. Each pair therefore
    gets a cluster bootstrap over subjects on both sides with the threshold refitted inside
    every replicate, and :func:`caliber.evaluation.safety.safety_rule` thresholds the upper
    end of that interval.

    The calibration and test sides are different subjects by construction -- one comes from
    the calibration split at operator ``a``, the other from the test split at ``b`` -- so
    there is no leakage to guard against here beyond what the splits already give.
    """
    families = (
        df.drop_duplicates("operator_id").set_index("operator_id")["operator_family"].to_dict()
    )
    operators = sorted(set(df["operator_id"].astype(str)))

    def by_cluster(frame: pd.DataFrame) -> list[np.ndarray]:
        return [
            g["score"].to_numpy(dtype=float)
            for _, g in frame.groupby(cluster_col, observed=True)
        ]

    rows = []
    for a in operators:
        calib = df[(df["operator_id"] == a) & (df["split"] == "calib") & df["label"]]
        if calib.empty:
            continue
        res = split_threshold(calib["score"], np.ones(len(calib), bool), alpha=alpha)
        calib_clusters = by_cluster(calib)
        for b in operators:
            if a == b:
                continue
            test = df[(df["operator_id"] == b) & (df["split"] == "test") & df["label"]]
            if test.empty:
                continue
            caught, total = _coverage(test, res.threshold)
            interval = clustered_coverage(
                calib_clusters, by_cluster(test), alpha=alpha, n_boot=n_boot,
                level=PAIR_LEVEL, seed=0,
            )
            rows.append({
                "calibration": a,
                "deployment": b,
                "family_calibration": families.get(a, "?"),
                "family_deployment": families.get(b, "?"),
                "signed_distance": distances.get((a, b), float("nan")),
                "threshold": float(res.threshold),
                "certifiable": bool(res.certifiable),
                "n_calib_positives": int(len(calib)),
                "n_caught": caught,
                "n_hallucinated": total,
                "coverage": caught / total if total else float("nan"),
                "coverage_lo": interval.lo,
                "coverage_hi": interval.hi,
                "coverage_sd": interval.sd,
                "design_effect": interval.design_effect,
            })
    return pd.DataFrame(rows)


def _threshold_identity(
    pairs: pd.DataFrame, control: pd.DataFrame
) -> pd.DataFrame | None:
    """D89's identity, run as a check on the bank rather than reported as a result.

    ``gap = F_b(tau_a) - F_b(tau_b)`` with ``F_b`` non-decreasing, so the gap's sign is the
    threshold difference's sign at every ordered pair, exactly and by theorem. A pair that
    disagrees therefore means the recorded thresholds and coverages did not come from the
    same scores -- a defect no coverage number survives, caught for the price of a groupby,
    and strictly stronger than the matched control, which only asks about the diagonal.

    It is a check and never a predictor: ``tau_a - tau_b`` is in
    :data:`prereg.GAP_BARRED_PREDICTORS` because it would beat the distance and every
    baseline by arithmetic.
    """
    # The matched coverage comes from the CONTROL table, which is the only place the
    # diagonal exists: _ordered_pairs skips self-pairs. Taking it from the pair table
    # instead reads an arbitrary transfer cell as the matched one, which the 2-operator
    # fixture quietly turned into a table of zero gaps rather than an error.
    if "operator_id" not in control.columns or "coverage" not in control.columns:
        print("[t1a] threshold identity not checkable: the control carries no diagonal")
        return None
    matched = control.set_index(control["operator_id"].astype(str))["coverage"]
    matched = matched[~matched.index.duplicated()]
    frame = pairs.assign(
        gap=pairs["deployment"].astype(str).map(matched).to_numpy(dtype=float)
        - pairs["coverage"].to_numpy(dtype=float)
    )
    unmatched = int(np.isnan(frame["gap"].to_numpy(dtype=float)).sum())
    if unmatched:
        print(f"[t1a] threshold identity: {unmatched} pair(s) have no matched cell to "
              "difference against and are not checkable")
    try:
        check = gap_sign_from_thresholds(frame)
    except (KeyError, ValueError) as exc:
        print(f"[t1a] threshold identity not checkable: {exc}")
        return None
    live = check[check["decidable"]]
    if live.empty:
        print("[t1a] threshold identity: no decidable pair (every gap or difference is 0)")
        return check
    agree = int(live["agrees"].sum())
    print(f"[t1a] threshold identity (D89): sign(gap) == sign(tau_a - tau_b) at "
          f"{agree} of {len(live)} decidable pairs")
    if agree < len(live):
        worst = live[~live["agrees"]]
        names = ", ".join(
            str(a) + "->" + str(b)
            for a, b in zip(worst["calibration"], worst["deployment"])
        )
        print("[t1a] WARNING: the identity is a theorem, so a disagreement means the bank "
              "contradicts the definition of coverage, not that the method failed. "
              "Offending pairs: " + names)
    return check


def _recut_splits(
    df: pd.DataFrame, n_splits: int, cluster_col: str = "sample_id", seed: int = 0
) -> list[tuple[list[str], list[str]]]:
    """``n_splits`` calibration/test partitions of the bank's subjects, the registry one first.

    The bank already holds every subject's scores at every operator, so the calibration/test
    assignment is only a partition of subject ids and T1-A can make its own re-cuts without
    waiting on anything. The registry partition is kept as split 0 because it is one of the
    draws, not a privileged one, and because it is the arm the sensitivity table reports.

    The design subjects are excluded from both sides of every re-cut: they carry the label's
    own threshold (epsilon) and putting them in a calibration half would leak it.
    """
    reg_calib = sorted(set(df.loc[df["split"] == "calib", cluster_col].astype(str)))
    reg_test = sorted(set(df.loc[df["split"] == "test", cluster_col].astype(str)))
    pool = sorted(set(reg_calib) | set(reg_test))
    if not pool:
        return []
    k = len(reg_calib) or max(1, len(pool) // 2)
    rng = np.random.default_rng(seed)
    out = [(reg_calib, reg_test)]
    for _ in range(max(0, n_splits - 1)):
        perm = list(rng.permutation(pool))
        out.append((sorted(perm[:k]), sorted(perm[k:])))
    return out


def _verify_split_design(
    df: pd.DataFrame, cluster_col: str, calib_fraction: float, design_fraction: float
) -> None:
    """Refuse unless the bank's own split column is what these fractions reproduce.

    The overlap correction is reconstructed from :func:`split_for`, not stored, so it is
    only right if the fractions handed in are the ones the splits were actually drawn
    under. A mismatch would not raise anywhere -- it would return a *different* design's
    overlap, and a too-low overlap narrows the interval and calls more pairs unsafe. So the
    assumption is checked against the bank instead of trusted.
    """
    sides = {
        str(c): split_for(
            str(c), calib_fraction=calib_fraction, design_fraction=design_fraction
        )
        for c in df[cluster_col].unique()
    }
    rebuilt = df[cluster_col].astype(str).map(sides)
    disagree = int((rebuilt != df["split"].astype(str)).sum())
    if disagree:
        raise SystemExit(
            f"[t1a] the bank's split column does not match split_for at calib_fraction="
            f"{calib_fraction} design_fraction={design_fraction} ({disagree} of {len(df)} "
            "regions disagree). The overlap correction is reconstructed from those "
            "fractions, so reading an across-split label under the wrong ones would report "
            "another design's overlap. Pass the fractions the bank was split under."
        )


def _read_pair_coverage(
    directory: str,
    df: pd.DataFrame,
    cluster_col: str,
    calib_fraction: float,
    design_fraction: float,
    level: float,
) -> tuple[pd.DataFrame, float, int]:
    """T2-A3's per-split and registry tables, as the frame the unsafe label reads.

    Returns the frame, the measured calibration overlap, and the number of salts used.

    Both label sources are taken from this one artefact rather than mixing its across-split
    arm with this script's own bootstrap for the registry arm. The sensitivity table exists
    to show what *the label source* changes; if the two arms also came from two measurement
    pipelines, any difference between them would be part source and part pipeline, and the
    table would not answer the question it is there to answer. That is the same defect as
    scoring the two arms on different pair sets, caught once already.
    """
    root = Path(directory)
    per_split_path = root / "pair_coverage_by_split.csv"
    registry_path = root / "pair_coverage_registry.csv"
    for path in (per_split_path, registry_path):
        if not path.exists():
            raise SystemExit(f"[t1a] {path} is missing; run pair_coverage_by_split.py first")
    per_split = pd.read_csv(per_split_path)
    registry = pd.read_csv(registry_path)
    if "salt" not in per_split.columns:
        raise SystemExit(f"[t1a] {per_split_path} has no salt column; cannot name its draws")
    seen = [str(x) for x in per_split["salt"].unique()]
    unknown = sorted(set(seen) - set(PROBE_SALTS))
    if unknown:
        raise SystemExit(
            f"[t1a] {per_split_path} contains salts this protocol does not define: "
            f"{unknown}. Calibration membership is reconstructed from the salt, so an "
            "unnamed draw cannot be placed and its share of the overlap cannot be "
            "measured. Add it to PROBE_SALTS or drop it from the table."
        )
    _verify_split_design(df, cluster_col, calib_fraction, design_fraction)
    salts = [s for s in PROBE_SALTS if s in set(seen)]
    calib_splits = calib_sets_for_salts(
        df[cluster_col].astype(str).unique(),
        salts,
        calib_fraction=calib_fraction,
        design_fraction=design_fraction,
    )
    rho = mean_pairwise_overlap(calib_splits)
    frame = pair_coverage_from_splits(per_split, registry, overlap=rho, level=level)
    return frame, rho, len(salts)


def _pairs_across_splits(
    df: pd.DataFrame,
    splits: list[tuple[list[str], list[str]]],
    alpha: float,
    cluster_col: str = "sample_id",
) -> pd.DataFrame:
    """Long frame of per-pair coverage on every re-cut: the label's estimand (D86).

    One threshold fit and one comparison per (pair, split) -- no bootstrap, because the
    interval on the mean comes from the spread *across* splits, not from within one.

    **One difference from Track 2's 13-split run, in the conservative direction.** Theirs
    re-values the label's epsilon on each split's own design side; this holds the registry
    epsilon fixed, so the spread here misses that component and the interval is narrower.
    A narrower interval calls more pairs unsafe, which is harsher on the rule, so the gap
    cannot flatter the endpoint. It is reported rather than corrected.
    """
    operators = sorted(set(df["operator_id"].astype(str)))
    ids = df[cluster_col].astype(str)
    rows = []
    for s, (calib_ids, test_ids) in enumerate(splits):
        in_calib = ids.isin(set(calib_ids))
        in_test = ids.isin(set(test_ids))
        for a in operators:
            calib = df[(df["operator_id"] == a) & in_calib & df["label"]]
            if calib.empty:
                continue
            res = split_threshold(calib["score"], np.ones(len(calib), bool), alpha=alpha)
            for b in operators:
                if a == b:
                    continue
                test = df[(df["operator_id"] == b) & in_test & df["label"]]
                if test.empty:
                    continue
                caught, total = _coverage(test, res.threshold)
                rows.append({
                    "calibration": a,
                    "deployment": b,
                    "split": s,
                    "coverage": caught / total if total else float("nan"),
                })
    return pd.DataFrame(rows)


def _print_table(frame: pd.DataFrame, cols: list[str], title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 74 - len(title)))
    have = [c for c in cols if c in frame.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame[have].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank")
    ap.add_argument("--distances", required=True, help="ct_distances.py output CSV")
    ap.add_argument("--measure", default=prereg.GAP_PREDICTION_DISTANCE,
                    help="frozen by the pre-registration; overriding it is a new analysis")
    ap.add_argument("--scale-operator", default=SCALE_OPERATOR)
    ap.add_argument("--cluster", default="sample_id",
                    help="the resampling unit for the control's interval; regions within "
                         "one of these are not independent")
    ap.add_argument("--out", default=None, help="directory for the CSVs")
    ap.add_argument("--n-splits", type=int, default=13,
                    help="patient re-cuts the across-split label is averaged over (D86)")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for the re-cuts; the registry split is always split 0")
    ap.add_argument("--pair-coverage-dir", default=None,
                    help="read the across-split label from a pair_coverage_by_split.py run "
                         "(T2-A3) instead of re-cutting here; both label sources then come "
                         "from that one artefact, so the sensitivity table compares sources "
                         "and not pipelines")
    ap.add_argument("--calib-fraction", type=float, default=0.5,
                    help="the fraction the bank was split under; checked against it, "
                         "because the overlap correction is reconstructed from it")
    ap.add_argument("--design-fraction", type=float, default=0.10,
                    help="as --calib-fraction, for the label's own held-out side (D-051)")
    ap.add_argument("--ignore-control", action="store_true",
                    help="read the rule even though the matched-operator control failed")
    ap.add_argument("--development", action="store_true",
                    help="run on a bank that does not carry coverage claims (D82); every "
                         "row is stamped DEVELOPMENT ONLY and none may be written up")
    add_label_args(ap)
    args = ap.parse_args(argv)
    alpha = prereg.ALPHA

    print(f"[t1a] prereg {prereg.fingerprint()} frozen {prereg.FROZEN_ON} "
          f"(amended {', '.join(d for d, _ in prereg.AMENDMENTS)}) "
          f"alpha={alpha} threshold={prereg.SIGN_RULE_THRESHOLD}")

    # D82, via Track 2's reportability module: T1-A and T1-D report coverage numbers and
    # nothing else, so the claim this script asks for is `coverage`. A bank that does not
    # carry it can still be computed on -- the numbers are real -- but they cannot be
    # written up, and the refusal belongs here rather than in whoever remembers.
    prov = _provenance(args.bank)
    print(f"[t1a] bank claims: {describe(prov)}")
    try:
        require_claim(prov, COVERAGE)
    except ClaimNotCarried as exc:
        if not args.development:
            print(f"\n[t1a] {exc}")
            print("[t1a] Pass --development to compute them anyway, unreportably.")
            return 3
        print("[t1a] --development given; every number below is DEVELOPMENT ONLY.")
    claim_stamp = stamp(prov, COVERAGE)

    df = read_bank(args.bank)
    if args.scale_operator not in set(df["operator_id"]):
        raise SystemExit(
            f"--scale-operator {args.scale_operator!r} is not in this bank; have "
            f"{sorted(set(df['operator_id']))}"
        )
    spec = label_from_args(df, args)
    df = derive_score_and_label(df, args.scale_operator, **spec.kwargs)
    print(f"[t1a] {len(df)} regions, {int(df['label'].sum())} positives, "
          f"label={args.label}, scales frozen at {args.scale_operator}")

    control = _matched_control(df, alpha, cluster_col=args.cluster)
    n_cells = int(df["operator_id"].nunique())
    _print_table(control, ["operator_id", "n_clusters_calib", "n_positives_calib", "coverage",
                           "ci_lo", "ci_hi", "coverage_lo_uncorrected",
                           "coverage_hi_uncorrected", "effective_n", "design_effect",
                           "beta_sd", "status", "status_uncorrected", "note"],
                 f"matched-operator control at {CONTROL_FAMILY_LEVEL:.0%} family-wise over "
                 f"{n_cells} cells (per cell {control_cell_level(n_cells):.3%}); "
                 "runs first and gates everything below")
    split = control[control["status"] != control["status_uncorrected"]]
    if len(split):
        print(f"[t1a] {len(split)} cell(s) differ between the gate's corrected level and "
              f"the uncorrected {CONTROL_LEVEL:.0%} per-cell band: "
              f"{', '.join(split['operator_id'])}. The correction makes the gate harder "
              "to fail, so both are printed; see D89.")
        print("[t1a] The gate decides whether the rule's output is labelled diagnostic. "
              "It does not enter the false-safe rate, Youden's J, or the permutation p, "
              "all of which are computed from the pair table alone.")
    failed = control[control["status"] == "fail"]
    unchecked = control[control["status"] == "not_checkable"]
    if len(failed):
        print(f"\n[t1a] CONTROL FAILED at {len(failed)} of {len(control)} operator(s): "
              f"{', '.join(failed['operator_id'])}")
        print("[t1a] A matched cell outside its own Beta interval means the bank is wrong, "
              "not that the method failed. Fix the bank.")
    if len(unchecked):
        print(f"\n[t1a] CONTROL NOT CHECKABLE at {len(unchecked)} of {len(control)} "
              f"operator(s): {', '.join(unchecked['operator_id'])}")
        print("[t1a] This is not a failure of the bank. Each note above says what is "
              "missing; a cluster shortfall needs more subjects, not a code fix.")
    if len(failed) or len(unchecked):
        if not args.ignore_control:
            print("\n[t1a] Refusing to report the safety rule while any cell is "
                  "unverified. Pass --ignore-control to look anyway.")
            return 2
        print("\n[t1a] --ignore-control given; the numbers below are diagnostic only.")

    pairs = _ordered_pairs(
        df, _load_distances(args.distances, args.measure), alpha, cluster_col=args.cluster
    )
    missing = int(pairs["signed_distance"].isna().sum())
    print(f"\n[t1a] {len(pairs)} ordered pairs, {missing} without a {args.measure} distance; "
          f"label = {prereg.UNSAFE_LABEL} at the {PAIR_LEVEL:.0%} level")
    if len(pairs):
        print(f"[t1a] per-pair coverage sd: median {pairs['coverage_sd'].median():.4f}, "
              f"design effect: median {pairs['design_effect'].median():.0f} regions per unit")
        _threshold_identity(pairs, control)

    # D86: the label's estimand is coverage marginal over the calibration draw. One draw
    # displaces every pair the same way, so a single-split label is biased, not just noisy.
    if args.pair_coverage_dir:
        across, rho, n_salts = _read_pair_coverage(
            args.pair_coverage_dir, df, args.cluster,
            args.calib_fraction, args.design_fraction, PAIR_LEVEL,
        )
        print(f"[t1a] label source = {prereg.UNSAFE_LABEL_SOURCE} from "
              f"{args.pair_coverage_dir} over {n_salts} salted re-cuts, pairwise "
              f"calibration overlap {rho:.0%} (reconstructed from the salts, not read)")
        # Both arms now come from that artefact. This script's own bootstrap is kept
        # alongside under _local names: the two are independent implementations of the
        # same registry cell, so their largest disagreement is a free agreement check and
        # it costs one merge to have it.
        pairs = pairs.rename(
            columns={"coverage": "coverage_local", "coverage_hi": "coverage_hi_local"}
        ).merge(across, on=["calibration", "deployment"], how="left")
        both = pairs[["coverage", "coverage_local"]].dropna()
        if len(both):
            gap = (both["coverage"] - both["coverage_local"]).abs()
            print(f"[t1a] registry coverage agrees with this script's own bootstrap to "
                  f"{gap.max():.4f} at worst over {len(both)} pairs (median "
                  f"{gap.median():.4f}); the artefact's value is the one read")
    else:
        splits = _recut_splits(
            df, args.n_splits, cluster_col=args.cluster, seed=args.split_seed
        )
        rho = mean_pairwise_overlap([c for c, _ in splits])
        print(f"[t1a] label source = {prereg.UNSAFE_LABEL_SOURCE} over {len(splits)} "
              f"re-cuts made here (registry split first), pairwise calibration overlap "
              f"{rho:.0%}")
        long = _pairs_across_splits(df, splits, alpha, cluster_col=args.cluster)
        if long.empty:
            print("[t1a] no re-cut produced a usable pair; cannot form the frozen label")
            return 2
        across = across_split_columns(long, overlap=rho, level=PAIR_LEVEL)
        pairs = pairs.merge(across, on=["calibration", "deployment"], how="left")
    n_short = int((pairs["n_splits"] < prereg.UNSAFE_LABEL_MIN_SPLITS).sum())
    if n_short:
        print(f"[t1a] {n_short} pair(s) have fewer than "
              f"{prereg.UNSAFE_LABEL_MIN_SPLITS} usable splits; their label is not reportable")
    print(f"[t1a] across-split se is {pairs['overlap_inflation'].median():.2f}x the naive "
          f"one after the overlap correction (n_eff = {pairs['n_eff_splits'].median():.1f})")

    rules = safety_by_family(pairs)
    table = safety_table(rules)
    _print_table(table, ["cells", "n_pairs", "false_safe_rate", "false_safe_lo",
                         "false_safe_hi", "trivial_false_safe_rate", "green_light_fraction",
                         "youden_j", "p_value", "beats_trivial", "verdict"],
                 "T1-A ordered-pair safety rule")
    _print_table(tolerance_sensitivity(pairs),
                 ["tolerance", "frozen_primary", "n_unsafe", "false_safe_rate",
                  "trivial_false_safe_rate", "green_light_fraction", "beats_trivial"],
                 "sensitivity to what counts as under-coverage (primary is tolerance=0)")
    # Required by D86: the frozen source runs the favourable way for the rule, so the
    # single-draw arm is printed beside it every time rather than on request.
    _print_table(label_source_sensitivity(pairs),
                 ["label_source", "frozen_primary", "n_pairs", "n_dropped", "n_unsafe",
                  "false_safe_rate", "trivial_false_safe_rate", "green_light_fraction",
                  "verdict", "note"],
                 "which split the unsafe label is read from (D86)")

    print("\n== T1-D worst cell " + "=" * 56)
    for name, frame in (("matched", control), ("transfer", pairs)):
        usable = frame[frame["n_hallucinated"] > 0]
        if usable.empty:
            continue
        cell_col = "operator_id" if name == "matched" else "deployment"
        if name == "transfer":
            usable = usable.assign(
                _cell=usable["calibration"] + " -> " + usable["deployment"]
            )
            cell_col = "_cell"
        worst = worst_operator_coverage(usable, cell_col=cell_col)
        print(f"  {name:<9} worst={worst.cell}")
        print(f"            observed_min={worst.observed_min:.4f} "
              f"pointwise_lower={worst.pointwise_lower:.4f} "
              f"simultaneous_lower={worst.simultaneous_lower:.4f} "
              f"over {worst.n_cells} cells -> certifiable_at_0.90="
              f"{worst.certifiable_at_nominal}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for frame in (control, pairs, table):
            frame["claim_stamp"] = claim_stamp
            frame["prereg"] = prereg.fingerprint()
            # Two orthogonal senses of "reportable" meet in these files and D96
            # found them disagreeing: the per-cell column is STATISTICAL (is the
            # estimate well enough determined to quote), while claim_stamp is
            # PROVENANCE (may a number from this bank be quoted at all). Every
            # face-bank row came back reportable=True under a DEVELOPMENT ONLY
            # stamp, so `df[df.reportable]` selected development numbers for the
            # write-up. The column is conjoined here and the statistical bit kept
            # beside it, which makes any existing filter conservative rather than
            # wrong: on a development bank it now selects nothing.
            if "reportable" in frame.columns:
                frame["reportable_statistical"] = frame["reportable"]
                frame["reportable"] = frame["reportable"].astype(bool) & (
                    claim_stamp == "reportable"
                )
        control.to_csv(out / "matched_control.csv", index=False)
        pairs.to_csv(out / "ordered_pairs.csv", index=False)
        table.to_csv(out / "safety_rule.csv", index=False)
        print(f"\n[t1a] wrote {out}/matched_control.csv, ordered_pairs.csv, safety_rule.csv")

    pooled = rules[0]
    print(f"\n[t1a] pooled verdict: {pooled.verdict} [{claim_stamp}]")
    return 0 if pooled.verdict == "positive" else 1


if __name__ == "__main__":
    sys.exit(main())
