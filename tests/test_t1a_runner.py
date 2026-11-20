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
