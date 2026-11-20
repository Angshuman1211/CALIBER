"""The T1-A runner's control gate: pass, fail, and the third state that is neither.

The gate exists because the work plan makes the matched-operator control a precondition --
nothing from the bank is read until it passes. A gate is only useful if it distinguishes
"this bank is wrong" from "this bank is too thin to check", because those send someone to
two different places: one to debug a pipeline, the other to acquire more data. The first
version of this script collapsed them, which is what these tests pin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliber.evaluation import prereg

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import t1a_safety_rule as T1A  # noqa: E402


def _bank(n_patients: int, per_patient: int, coverage: float, seed: int = 0) -> pd.DataFrame:
    """A labelled, scored frame shaped like a bank, with a chosen matched-cell coverage.

    Calibration scores are uniform; test scores are placed so that the given fraction of
    them lands above the calibration threshold. Only the columns the control reads are
    present, which is the point -- this tests the gate, not the bank reader.
    """
    rng = np.random.default_rng(seed)
    rows = []
    n_cal = n_patients // 2
    for p in range(n_patients):
        split = "calib" if p < n_cal else "test"
        for _ in range(per_patient):
            rows.append({
                "operator_id": "op",
                "operator_family": "fam",
                "sample_id": f"p{p:03d}",
                "split": split,
                "label": True,
                "score": float(rng.random()),
            })
    df = pd.DataFrame(rows)

    calib = df[df["split"] == "calib"]["score"].to_numpy()
    n = calib.size
    k = int(np.floor(prereg.ALPHA * (n + 1)))
    if k >= 1:
        tau = np.sort(calib)[k - 1]
        test_idx = df.index[df["split"] == "test"]
        n_test = len(test_idx)
        n_above = int(round(coverage * n_test))
        placed = np.concatenate([
            rng.uniform(tau, 1.0, size=n_above),
            rng.uniform(0.0, tau, size=n_test - n_above),
        ])
        df.loc[test_idx, "score"] = rng.permutation(placed)
    return df


def test_a_well_calibrated_cell_passes():
    control = T1A._matched_control(_bank(40, 10, coverage=0.90), prereg.ALPHA)
    row = control.iloc[0]
    assert row["status"] == "pass"
    assert row["note"] == ""
    assert row["coverage"] == pytest.approx(0.90, abs=0.02)


def test_a_badly_calibrated_cell_fails_and_names_the_interval():
    control = T1A._matched_control(_bank(40, 10, coverage=0.55), prereg.ALPHA, n_boot=300)
    row = control.iloc[0]
    assert row["status"] == "fail"
    assert "excludes 0.90" in row["note"]
    assert "design effect" in row["note"]


def test_too_few_clusters_is_not_checkable_not_failed():
    """Below the cluster floor the interval is an artefact, so the cell is not judged."""
    n = T1A.MIN_CLUSTERS_FOR_CONTROL - 2
    control = T1A._matched_control(_bank(n * 2, 20, coverage=0.90), prereg.ALPHA, n_boot=200)
    row = control.iloc[0]
    assert row["status"] == "not_checkable"
    assert "below the floor" in row["note"]


def test_the_control_reports_the_design_effect_so_the_clustering_is_visible():
    """The number whose absence caused the mistake this control was rewritten to avoid."""
    control = T1A._matched_control(_bank(40, 40, coverage=0.90), prereg.ALPHA, n_boot=300)
    row = control.iloc[0]
    assert np.isfinite(row["design_effect"])
    assert np.isfinite(row["effective_n"])
    assert row["effective_n"] <= row["n_positives_calib"]
    # The Beta column is still there, as the labelled idealisation.
    assert np.isfinite(row["beta_sd"])


def test_an_empty_side_is_not_checkable():
    df = _bank(40, 10, coverage=0.90)
    df = df[df["split"] == "calib"]
    control = T1A._matched_control(df, prereg.ALPHA)
    assert control.iloc[0]["status"] == "not_checkable"
    assert "no calibration or no test positives" in control.iloc[0]["note"]


def test_distance_loader_fills_the_reverse_direction(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "calibration,deployment,measure,value,signed,seconds,note\n"
        "a,b,null_space_growth,0.25,True,1,\n"
        "a,b,null_space_projection,0.40,False,1,\n"
        "a,c,null_space_growth,nan,True,1,probe failed\n",
        encoding="utf-8",
    )
    growth = T1A._load_distances(str(csv), "null_space_growth")
    assert growth[("a", "b")] == pytest.approx(0.25)
    assert growth[("b", "a")] == pytest.approx(-0.25)
    assert ("a", "c") not in growth, "a non-finite value must be dropped, not carried as NaN"
    assert T1A._load_distances(str(csv), "null_space_projection")[("a", "b")] == pytest.approx(0.40)


def test_distance_loader_refuses_a_measure_it_has_no_rows_for(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "calibration,deployment,measure,value,signed,seconds,note\n"
        "a,b,null_space_growth,0.25,True,1,\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="chordal_subspace_distance"):
        T1A._load_distances(str(csv), "chordal_subspace_distance")


def test_ordered_pairs_excludes_self_pairs_and_carries_families():
    df = _bank(40, 6, coverage=0.90)
    other = df.copy()
    other["operator_id"] = "op2"
    other["operator_family"] = "fam2"
    both = pd.concat([df, other], ignore_index=True)
    pairs = T1A._ordered_pairs(both, {("op", "op2"): 0.3, ("op2", "op"): -0.3}, prereg.ALPHA)
    assert len(pairs) == 2
    assert (pairs["calibration"] != pairs["deployment"]).all()
    assert set(pairs["family_deployment"]) == {"fam", "fam2"}
    assert pairs["signed_distance"].notna().all()


# -- D82 / T2-B integration ----------------------------------------------------


def test_the_claim_t1a_needs_is_the_claim_t2a_s_bank_carries():
    """The pin between Track 1's analyses and Track 2's reportability rules.

    T1-A and T1-D report coverage numbers and nothing else, so the script asks for the
    ``coverage`` claim. T2-A's configuration -- TotalSegmentator under the LIDC chest prior
    -- is cross-anatomy, so D80 gives it coverage and refuses detection. Those two facts
    have to agree or the analysis is unreportable on the only bank built for it, and the
    agreement is currently implicit in two modules owned by two tracks. This makes it
    explicit: if the admissibility rules move, this fails rather than the pipeline silently
    refusing on the day the bank lands.
    """
    from caliber.data.totalsegmentator import TotalSegmentatorDataset
    from caliber.reportability import COVERAGE, DETECTION, claims_for

    carried = claims_for(
        TotalSegmentatorDataset,
        "diffusion",
        "checkpoints/song_inverse_problems/ct2d_320_ncsnpp_ve",
    )
    assert COVERAGE in carried, "T1-A's coverage numbers would be unreportable on T2-A"
    assert DETECTION not in carried, "cross-anatomy must still refuse detection (D80)"


def test_provenance_is_read_from_the_manifest_beside_the_bank(tmp_path):
    import json

    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "manifest.json").write_text(
        json.dumps({"provenance": {"reportable_claims": ["coverage"], "dataset": "x"}}),
        encoding="utf-8",
    )
    assert T1A._provenance(str(bank))["reportable_claims"] == ["coverage"]


def test_a_missing_or_broken_manifest_yields_no_claims_not_a_crash(tmp_path):
    """A bare parquet, or a manifest that will not parse, must fail closed."""
    from caliber.reportability import claims_of

    empty = tmp_path / "nomanifest"
    empty.mkdir()
    assert T1A._provenance(str(empty)) == {}
    assert claims_of(T1A._provenance(str(empty))) == frozenset()

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    assert T1A._provenance(str(broken)) == {}


def test_the_amendment_trail_is_dated_and_explained():
    """An amendment to the freeze is only legitimate if it is recorded as one."""
    assert len(prereg.AMENDMENTS) >= 2
    for date, reason in prereg.AMENDMENTS:
        assert date >= prereg.FROZEN_ON, f"amendment {date} predates the freeze"
        assert len(reason) > 60, f"amendment {date} has no usable reason"
    assert prereg.as_dict()["amendments"] == [d for d, _ in prereg.AMENDMENTS]


def test_baseline_scope_covers_every_declared_baseline():
    """A baseline with no stated scope is one that can be defined after the fact."""
    assert set(prereg.BASELINE_SCOPE) == set(prereg.GAP_PREDICTION_BASELINES)
    assert "undefined across families" in prereg.BASELINE_SCOPE["acquisition_parameter"]


# -- D86: the runner must produce the arm the frozen label reads --------------


def _multi_operator_bank(n_patients: int = 40, per_patient: int = 30, seed: int = 0):
    """Three operators of increasing severity on shared patients, registry split by half."""
    rng = np.random.default_rng(seed)
    rows = []
    n_cal = n_patients // 2
    for i, (op, loc) in enumerate((("op0", 0.0), ("op1", 0.25), ("op2", 0.55))):
        for p in range(n_patients):
            for _ in range(per_patient):
                rows.append({
                    "operator_id": op,
                    "operator_family": "sv" if i < 2 else "la",
                    "sample_id": f"p{p:03d}",
                    "split": "calib" if p < n_cal else "test",
                    "label": True,
                    "score": float(rng.normal(loc, 1.0)),
                })
    return pd.DataFrame(rows)


def test_recuts_keep_the_registry_split_first_and_exclude_design_subjects():
    df = _multi_operator_bank()
    df.loc[df["sample_id"] == "p000", "split"] = "design"
    splits = T1A._recut_splits(df, n_splits=5)

    assert len(splits) == 5
    reg_calib = sorted(set(df.loc[df["split"] == "calib", "sample_id"]))
    assert splits[0][0] == reg_calib                     # registry partition is split 0
    for calib, test in splits:
        assert "p000" not in calib and "p000" not in test   # design held out everywhere
        assert not set(calib) & set(test)                   # the sides stay disjoint
        assert len(calib) == len(reg_calib)                 # same calibration size


def test_the_runner_produces_exactly_the_columns_the_frozen_label_reads():
    """The gap this closes: nothing previously ran the runner's frame through the label."""
    from caliber.evaluation import safety

    df = _multi_operator_bank()
    splits = T1A._recut_splits(df, n_splits=9)
    long = T1A._pairs_across_splits(df, splits, prereg.ALPHA)
    assert set(long["split"]) == set(range(9))
    assert len(long) == 9 * 6                            # 3 operators -> 6 ordered pairs

    across = T1A.across_split_columns(
        long, overlap=T1A.mean_pairwise_overlap([c for c, _ in splits])
    )
    wanted = safety.LABEL_SOURCE_COLUMNS[prereg.UNSAFE_LABEL_SOURCE]
    assert set(wanted) <= set(across.columns)
    assert across["n_splits"].eq(9).all()


def test_the_merged_frame_runs_the_rule_without_tripping_the_no_fall_back_guard():
    """End to end on the runner's own helpers, which is what main does."""
    from caliber.evaluation import safety

    df = _multi_operator_bank()
    sev = {"op0": 0, "op1": 1, "op2": 2}
    splits = T1A._recut_splits(df, n_splits=9)
    rho = T1A.mean_pairwise_overlap([c for c, _ in splits])
    across = T1A.across_split_columns(T1A._pairs_across_splits(df, splits, prereg.ALPHA),
                                      overlap=rho)
    registry = T1A._ordered_pairs(
        df,
        {(a, b): float(sev[b] - sev[a]) for a in sev for b in sev if a != b},
        prereg.ALPHA,
        n_boot=200,
    )
    frame = registry.merge(across, on=["calibration", "deployment"], how="left")

    rules = safety.safety_by_family(frame, with_null=False)
    assert rules[0].label_source == prereg.UNSAFE_LABEL_SOURCE
    assert rules[0].n_pairs == 6
    both = safety.label_source_sensitivity(frame)
    assert both["verdict"].ne("unavailable").all()   # both arms present, neither skipped


# -- reading the across-split label from T2-A3's artefact (D86) --------------------------


def _split_bank(n_patients: int = 60, design_fraction: float = 0.10) -> pd.DataFrame:
    """A frame whose ``split`` column really is what ``split_for`` assigns."""
    from caliber.data.splits import split_for

    return pd.DataFrame([
        {
            "sample_id": f"LIDC-IDRI-{i:04d}",
            "split": split_for(
                f"LIDC-IDRI-{i:04d}", calib_fraction=0.5, design_fraction=design_fraction
            ),
        }
        for i in range(1, n_patients + 1)
    ])


def _artefact(tmp_path, salts=None, coverage=0.89):
    """A directory shaped like ``pair_coverage_by_split.py``'s output."""
    from caliber.data.splits import PROBE_SALTS

    use = list(PROBE_SALTS) if salts is None else list(salts)
    pairs = (("a", "b"), ("b", "a"))
    pd.DataFrame([
        {"salt": s, "calibration": x, "deployment": y, "coverage": coverage + 0.003 * i}
        for i, s in enumerate(use)
        for x, y in pairs
    ]).to_csv(tmp_path / "pair_coverage_by_split.csv", index=False)
    pd.DataFrame([
        {"calibration": x, "deployment": y, "coverage": 0.84, "lo": 0.79, "hi": 0.88}
        for x, y in pairs
    ]).to_csv(tmp_path / "pair_coverage_registry.csv", index=False)
    return tmp_path


def test_the_split_design_is_checked_against_the_bank_not_trusted():
    """The overlap is reconstructed from the fractions, so the wrong ones silently report
    another design's overlap -- and a too-low overlap narrows the interval."""
    df = _split_bank(design_fraction=0.10)
    T1A._verify_split_design(df, "sample_id", 0.5, 0.10)   # the truth: no raise
    with pytest.raises(SystemExit, match="does not match split_for"):
        T1A._verify_split_design(df, "sample_id", 0.5, 0.30)


def test_reading_the_artefact_measures_the_overlap_from_its_salts(tmp_path):
    from caliber.data.splits import PROBE_SALTS

    frame, rho, n_salts = T1A._read_pair_coverage(
        str(_artefact(tmp_path)), _split_bank(100), "sample_id", 0.5, 0.10, 0.95
    )
    assert n_salts == len(PROBE_SALTS)
    assert 0.2 < rho < 0.6, rho
    # both label sources, from the one artefact
    for col in ("coverage", "coverage_hi", "coverage_across", "coverage_across_hi"):
        assert col in frame.columns
    assert float(frame["overlap_inflation"].iloc[0]) > 2.0


def test_a_salt_the_protocol_does_not_name_is_refused(tmp_path):
    """Membership is reconstructed from the salt, so an unnamed draw cannot be placed and
    its share of the overlap cannot be measured. Guessing it would widen or narrow the
    interval by an unknown amount."""
    from caliber.data.splits import PROBE_SALTS

    bad = _artefact(tmp_path, salts=[*PROBE_SALTS[:12], "probe-rerun-with-new-seed"])
    with pytest.raises(SystemExit, match="salts this protocol does not define"):
        T1A._read_pair_coverage(str(bad), _split_bank(100), "sample_id", 0.5, 0.10, 0.95)


def test_a_subset_of_the_salts_is_read_as_the_splits_it_has(tmp_path):
    """A split whose design side could not value the label is skipped upstream, so the
    table arrives short. It is described by the draws it has, not credited with all 13."""
    from caliber.data.splits import PROBE_SALTS

    short = _artefact(tmp_path, salts=PROBE_SALTS[:9])
    frame, _, n_salts = T1A._read_pair_coverage(
        str(short), _split_bank(100), "sample_id", 0.5, 0.10, 0.95
    )
    assert n_salts == 9
    assert int(frame["n_splits"].iloc[0]) == 9
    assert bool(frame["reportable"].iloc[0]) is True


def test_a_missing_artefact_says_which_file_and_what_makes_it(tmp_path):
    with pytest.raises(SystemExit, match="pair_coverage_by_split"):
        T1A._read_pair_coverage(
            str(tmp_path), _split_bank(60), "sample_id", 0.5, 0.10, 0.95
        )
