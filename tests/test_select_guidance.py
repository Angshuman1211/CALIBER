"""D-031's guidance choice, which decides an unattended DPS bank (T2.10).

The choice is made by a script, overnight, from a design-split sweep. So the rule is pinned:
an inadmissible setting (D-033: error not mostly in the null space) is never chosen however
well it fits the data, and a sweep with no admissible setting builds no bank at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import select_guidance  # noqa: E402


def _artefact(directory: Path, z: float, null_shares, dc_errors) -> None:
    ops = [{"operator_id": f"op{i}", "null_error_share": s, "data_consistency_error": d}
           for i, (s, d) in enumerate(zip(null_shares, dc_errors, strict=True))]
    name = f"stage2-w11_dps_sweep_z{z:g}-diffusion.dps.brats_ncsnpp_ve.t24.json"
    (directory / name).write_text(json.dumps({
        "config": "configs/experiments/w11_m4raw_brats_dps_sweep.yaml",
        "n_items": 8, "operators": ops}), encoding="utf-8")


def test_the_best_fitting_admissible_setting_is_chosen_over_a_better_fitting_inadmissible_one(
    tmp_path, capsys
):
    _artefact(tmp_path, 1, [0.90, 0.85, 0.80], [0.30, 0.30, 0.30])
    _artefact(tmp_path, 5, [0.80, 0.70, 0.60], [0.20, 0.20, 0.20])
    # fits best, but one operator's error is range-dominated: D-033 excludes it
    _artefact(tmp_path, 20, [0.70, 0.55, 0.40], [0.05, 0.05, 0.05])
    assert select_guidance.main([str(tmp_path), "--prefix", "w11_dps_sweep_z"]) == 0
    out = capsys.readouterr().out
    assert "STEP_SIZE=5" in out
    assert "STEP_SIZE=20" not in out


def test_a_choice_at_the_edge_of_the_sweep_says_the_optimum_is_not_bracketed(tmp_path, capsys):
    _artefact(tmp_path, 1, [0.9, 0.9, 0.9], [0.3, 0.3, 0.3])
    _artefact(tmp_path, 50, [0.9, 0.9, 0.9], [0.1, 0.1, 0.1])
    assert select_guidance.main([str(tmp_path)]) == 0
    assert "not bracketed" in capsys.readouterr().out


def test_a_sweep_with_no_admissible_setting_builds_no_bank(tmp_path, capsys):
    _artefact(tmp_path, 5, [0.6, 0.45, 0.3], [0.2, 0.2, 0.2])
    assert select_guidance.main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "STEP_SIZE=" not in captured.out
    assert "no DPS bank" in captured.err
