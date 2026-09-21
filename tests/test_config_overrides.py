"""An override that names nothing is refused, not silently ignored (D-047, D-048).

Track 1 found that `load_config` accepted any `key=value` override. A mistyped
`run.limitt=5` was added to the config as a brand-new key that nothing read, so the run
used the default item count — and the config hash, manifest and provenance all agreed with
each other, because they record what ran. Nothing afterwards could show the mistake.

The fix has to know what is valid, not merely what is present: some keys the runner reads
are deliberately absent from `base.yaml` (adding a default there would change every
existing bank's config hash, D-040), and solver or dataset parameters depend on which
solver or dataset is configured. These tests pin both directions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from caliber.config import OPTIONAL_KEYS, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1] / "configs/experiments"
W3 = str(ROOT / "w03_mri_fixed_operator.yaml")      # ddnm on fastmri
DPS = str(ROOT / "w03_dps_step_budget.yaml")        # dps
SMOKE = str(ROOT / "w01_smoke_phantom.yaml")        # reference sampler on phantoms


# -- Track 1's three examples, verbatim ------------------------------------------


def test_a_nonsense_key_is_refused():
    with pytest.raises(ConfigError, match="silently ignored"):
        load_config(W3, ["nonsense.key.that.does.not.exist=99"])


def test_a_typo_is_refused_and_the_real_key_is_suggested():
    with pytest.raises(ConfigError, match=r"Did you mean run\.limit"):
        load_config(W3, ["run.limitt=5"])


def test_list_indexing_is_refused_with_a_way_forward():
    with pytest.raises(ConfigError, match="Override the whole list"):
        load_config(W3, ["experiment.operators[0].acceleration=[4]"])


# -- nothing legitimate breaks ------------------------------------------------------


@pytest.mark.parametrize("override", [
    "run.limit=3",
    "run.output_root=elsewhere",
    "experiment.solver.n_steps=2",
    "data.image_shape=[128,128]",
    "data.dataset.image_shape=[128,128]",
])
def test_keys_that_exist_in_the_config_are_accepted(override):
    load_config(W3, [override])


@pytest.mark.parametrize("key", sorted(OPTIONAL_KEYS))
def test_the_runner_s_declared_optional_keys_are_accepted(key):
    """Absent from base.yaml on purpose (D-040), and read by the runner."""
    load_config(W3, [f"{key}=1"])


def test_the_optional_keys_are_exactly_what_the_runner_reads():
    """If the runner starts reading a new optional key, it must be declared — or refused.

    Reads the runner's source rather than trusting a list kept by hand.
    """
    import re

    source = (Path(__file__).resolve().parents[1] / "src/caliber/runner.py").read_text("utf-8")
    base = {"cache_reconstructions", "limit", "output_root", "seed"}      # in base.yaml
    read = set(re.findall(r'run_cfg\.get\("([a-z_]+)"', source)) - base
    assert {f"run.{k}" for k in read} <= OPTIONAL_KEYS


def test_a_solver_parameter_inherited_through_kwargs_is_accepted():
    """DDNM takes `**kwargs` and forwards to its base, so its own signature understates it."""
    load_config(W3, ["experiment.solver.max_draws_per_batch=1"])


def test_a_dataset_parameter_the_yaml_does_not_set_is_accepted():
    load_config(W3, ["data.dataset.calib_fraction=0.5"])


def test_a_solver_specific_key_is_checked_against_the_solver_actually_configured():
    """Validation runs after every override, so changing the solver first counts."""
    load_config(DPS, ["experiment.solver.step_size=20.0"])
    load_config(W3, ["experiment.solver.name=dps", "experiment.solver.step_size=20"])


# -- solver and dataset parameters are checked against the class -----------------


def test_a_parameter_the_configured_solver_does_not_take_is_refused():
    """DDNM has no guidance step size; accepting one would silently do nothing."""
    with pytest.raises(ConfigError, match="not a parameter of the configured solver"):
        load_config(W3, ["experiment.solver.step_size=20"])


def test_a_solver_parameter_typo_is_refused_with_a_suggestion():
    with pytest.raises(ConfigError, match="n_steps"):
        load_config(W3, ["experiment.solver.n_step=2"])


def test_a_dataset_parameter_typo_is_refused():
    with pytest.raises(ConfigError, match="not a parameter of the configured dataset"):
        load_config(W3, ["data.dataset.slice_per_volume=2"])


def test_a_catch_all_kwargs_is_not_treated_as_accepting_everything():
    """The reference sampler swallows `**_`. That is how a typo would slip through, so it
    must not count as accepting an unknown key."""
    with pytest.raises(ConfigError, match="not a parameter"):
        load_config(SMOKE, ["experiment.solver.max_draws_per_batch=2"])


# -- the property that matters ----------------------------------------------------


def test_a_refused_override_never_reaches_the_run(tmp_path):
    """The failure D-047 describes is a run that proceeds. It must not proceed."""
    from caliber.runner import main

    rc = main([SMOKE, "--set", "run.limitt=1", "--set", f"run.output_root={tmp_path.as_posix()}"])
    assert rc != 0
    assert not any(tmp_path.iterdir()), "nothing should have been written"
