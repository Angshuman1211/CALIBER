"""`coverage_by_patient.py` shows the control alone unless transfer is asked for (P-44).

D-070's rule for a new bank: the matched-operator control is read first, and nothing else
until it passes. On the LIDC bank that rule was honoured by a person looking away, because
the script printed the naive arm in the same table. These pin that the default run does not
compute the naive arm at all, and that `--transfer` still gives the table every earlier
record was built from.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import coverage_by_patient  # noqa: E402

LOW, HIGH = "ct.low", "ct.high"


@pytest.fixture
def frame(monkeypatch):
    """A bank already carrying `score` and `label`: 12 patients a side, two operators, the
    harder operator's positives scoring lower so naive transfer has something to lose."""
    rng = np.random.default_rng(0)
    rows = []
    for split in ("calib", "test"):
        for p in range(12):
            for op, shift in ((LOW, 0.0), (HIGH, -1.0)):
                for _ in range(30):
                    label = bool(rng.random() < 0.4)
                    rows.append({"split": split, "patient_id": f"{split}{p}", "operator_id": op,
                                 "label": label, "score": rng.normal(2.0 * label + shift, 1.0)})
    df = pd.DataFrame(rows)
    monkeypatch.setattr(coverage_by_patient, "read_bank", lambda _: df)
    monkeypatch.setattr(coverage_by_patient, "label_from_args",
                        lambda raw, args: SimpleNamespace(text="synthetic", kwargs={}))
    monkeypatch.setattr(coverage_by_patient, "derive_score_and_label", lambda raw, op, **k: raw)
    return df


def test_the_default_run_is_the_control_and_never_computes_the_naive_arm(frame, tmp_path, capsys):
    out = tmp_path / "control.csv"
    assert coverage_by_patient.main(["bank", "--low", LOW, "--high", HIGH, "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert set(table["method"]) == {"group"}
    printed = capsys.readouterr().out
    assert "matched-operator control" in printed
    assert "naive transfer misses" not in printed
    assert "--transfer" in printed          # says how to get the rest, once the control is read


def test_transfer_adds_the_naive_arm_and_leaves_the_control_unchanged(frame, tmp_path):
    control, both = tmp_path / "control.csv", tmp_path / "both.csv"
    args = ["bank", "--low", LOW, "--high", HIGH]
    assert coverage_by_patient.main([*args, "--out", str(control)]) == 0
    assert coverage_by_patient.main([*args, "--transfer", "--out", str(both)]) == 0
    c, b = pd.read_csv(control), pd.read_csv(both)
    assert set(b["method"]) == {"naive", "group"}
    group = b[b["method"] == "group"].reset_index(drop=True)
    pd.testing.assert_frame_equal(c.reset_index(drop=True), group)


def test_the_matched_arm_is_the_same_in_both_directions(frame, tmp_path):
    """The control does not depend on which operator naive transfer is calibrated at, which
    is why the default output can show one row per operator."""
    out = tmp_path / "control.csv"
    coverage_by_patient.main(["bank", "--low", LOW, "--high", HIGH, "--out", str(out)])
    table = pd.read_csv(out)
    for _, part in table.groupby("deployed_at"):
        assert part["coverage"].nunique() == 1
