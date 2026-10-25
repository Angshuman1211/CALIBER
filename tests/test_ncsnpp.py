"""Gates for the medical prior: the NCSN++ port, its converter, and the VE adapter (D79, T2.10).

The port was checked once against Song et al.'s own Flax model on the real checkpoint, and
agreed to ~1e-5 of the score's magnitude at sigma = 0.05, 1 and 50. That check needs JAX,
which the project does not depend on. So these tests pin what can drift without it:

* the **converter's mapping**, round-tripped through a Flax-layout tree, and its refusal
  to produce a network when a layer is missing or left over;
* the **VE-to-VP algebra**, against a Gaussian prior whose score and whose optimal noise
  prediction are both known in closed form, so the adapter is checked against the answer
  rather than against itself;
* the **schedule** and the **prior-domain gate**;
* and, where the converted checkpoint is present, the port against values recorded from
  the Flax reference run, so a regression in the port is caught on a machine that has it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from caliber.solvers.denoiser import (
    DiffusersUNetDenoiser,
    SongVEScoreDenoiser,
    build_denoiser,
    prior_domain,
    prior_is_admissible,
    ve_schedule,
)
from caliber.solvers.ncsnpp import (
    NIN,
    NCSNpp,
    NCSNppConfig,
    downsample_2d,
    upsample_2d,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import convert_song_checkpoint as convert_mod  # noqa: E402

TINY = NCSNppConfig(image_size=32, nf=8, ch_mult=(1, 2), num_res_blocks=1,
                    attn_resolutions=(16,), sigma_max=50.0)
BRATS_PT = REPO_ROOT / "checkpoints/song_inverse_problems/brats_ncsnpp_ve/ncsnpp_brats240.pt"


def _randomised(config: NCSNppConfig, seed: int = 0) -> NCSNpp:
    torch.manual_seed(seed)
    model = NCSNpp(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return model.eval()


# -- the network ----------------------------------------------------------------


def test_the_network_maps_an_image_and_a_noise_level_to_a_score_of_the_same_shape():
    model = _randomised(TINY)
    x = torch.rand(2, 1, 32, 32)
    out = model(x, torch.tensor([0.1, 10.0]))
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_a_resolution_the_architecture_cannot_halve_is_refused():
    with pytest.raises(ValueError, match="halve"):
        NCSNpp(NCSNppConfig(image_size=30, nf=8, ch_mult=(1, 2, 2), num_res_blocks=1))


def test_fir_resampling_halves_and_doubles_and_keeps_a_flat_image_flat():
    """The FIR kernel is normalised, so a constant interior stays constant -- the property
    that makes the input pyramid a pyramid rather than a gain."""
    x = torch.full((1, 1, 16, 16), 3.0)
    down, up = downsample_2d(x, (1, 3, 3, 1)), upsample_2d(x, (1, 3, 3, 1))
    assert down.shape[-1] == 8 and up.shape[-1] == 32
    assert torch.allclose(down[..., 2:-2, 2:-2], torch.tensor(3.0))
    assert torch.allclose(up[..., 4:-4, 4:-4], torch.tensor(3.0))


# -- the converter ----------------------------------------------------------------


def _to_flax(model: NCSNpp) -> dict:
    """The inverse of the converter: a torch model written out in Flax's names and layouts."""
    counters: dict[str, int] = {}

    def export(module: nn.Module) -> dict:
        out: dict = {}
        if isinstance(module, nn.Conv2d):
            return {"kernel": module.weight.detach().numpy().transpose(2, 3, 1, 0),
                    "bias": module.bias.detach().numpy()}
        if isinstance(module, nn.Linear):
            return {"kernel": module.weight.detach().numpy().T,
                    "bias": module.bias.detach().numpy()}
        if isinstance(module, nn.GroupNorm):
            return {"scale": module.weight.detach().numpy().reshape(1, 1, 1, -1),
                    "bias": module.bias.detach().numpy().reshape(1, 1, 1, -1)}
        if isinstance(module, NIN):
            return {"W": module.W.detach().numpy(), "b": module.b.detach().numpy()}
        for name, p in module.named_parameters(recurse=False):
            out[name] = p.detach().numpy()
        for name, child in module.named_children():
            out[name] = export(child)
        return out

    tree = {}
    for module in model.all_modules:
        cls = convert_mod.FLAX_CLASS[type(module)]
        k = counters.get(cls, 0)
        counters[cls] = k + 1
        tree[f"{cls}_{k}"] = export(module)
    return tree


def test_a_flax_layout_tree_converts_back_to_the_same_network():
    """Every layout rule (conv transpose, dense transpose, GroupNorm reshape) and the
    call-order mapping, checked as a round trip on the network's actual output."""
    source = _randomised(TINY, seed=1)
    weights = convert_mod.convert(_to_flax(source), TINY)
    rebuilt = NCSNpp(TINY)
    rebuilt.load_state_dict(weights)
    x, sigma = torch.rand(2, 1, 32, 32), torch.tensor([0.05, 5.0])
    with torch.no_grad():
        assert torch.allclose(rebuilt.eval()(x, sigma), source(x, sigma), atol=1e-6)


def test_a_checkpoint_missing_a_layer_is_refused():
    tree = _to_flax(_randomised(TINY))
    del tree["AttnBlockpp_0"]
    with pytest.raises(KeyError, match="AttnBlockpp_0"):
        convert_mod.convert(tree, TINY)


def test_parameter_free_layers_serialised_as_empty_dicts_are_not_a_leftover():
    """Some Flax versions write the pyramids' FIR resamplers as empty dicts (the LIDC
    checkpoint does). Nothing to map, so not a refusal -- but only because they are empty."""
    source = _randomised(TINY, seed=2)
    tree = _to_flax(source)
    tree["Downsample_0"], tree["Upsample_0"] = {}, {}
    rebuilt = NCSNpp(TINY)
    rebuilt.load_state_dict(convert_mod.convert(tree, TINY))
    x, sigma = torch.rand(1, 1, 32, 32), torch.tensor([1.0])
    with torch.no_grad():
        assert torch.allclose(rebuilt.eval()(x, sigma), source(x, sigma), atol=1e-6)


def test_a_checkpoint_with_a_layer_left_over_is_refused():
    """The silent failure this guards: a plausible network built from the wrong subset."""
    tree = _to_flax(_randomised(TINY))
    tree["Conv_99"] = tree["Conv_0"]
    with pytest.raises(ValueError, match="never mapped"):
        convert_mod.convert(tree, TINY)


# -- the VE adapter -----------------------------------------------------------------


class _GaussianPriorScore(nn.Module):
    """The exact VE score of ``x0 ~ N(mu, tau^2)`` per pixel: ``-(x - mu) / (tau^2 + s^2)``."""

    def __init__(self, mu: float, tau: float):
        super().__init__()
        self.mu, self.tau = mu, tau

    def forward(self, x, sigma):
        s2 = sigma.reshape(-1, 1, 1, 1) ** 2
        return -(x - self.mu) / (self.tau**2 + s2)


@pytest.fixture()
def adapter(tmp_path):
    """A SongVEScoreDenoiser over a saved tiny checkpoint, its network swapped for the
    Gaussian prior's exact score."""
    torch.save({"config": TINY.to_dict(), "state_dict": NCSNpp(TINY).state_dict(),
                "source": {}}, tmp_path / "ncsnpp_tiny.pt")
    den = SongVEScoreDenoiser(tmp_path)
    den._model = _GaussianPriorScore(mu=0.4, tau=0.2)
    return den


@pytest.mark.parametrize("t", [0, 250, 600, 999])
def test_the_adapter_returns_the_optimal_noise_prediction_for_a_known_prior(adapter, t):
    """For ``x_t = sqrt(a) x0 + sqrt(1-a) eps`` with ``x0 ~ N(mu, tau^2)``, the optimal
    prediction is ``E[eps | x_t] = sqrt(1-a) (x_t - sqrt(a) mu) / (a tau^2 + 1 - a)``. The
    adapter reaches it only through the VE change of variables, so this checks the
    algebra rather than the adapter against itself."""
    a = float(adapter.default_schedule.alpha_bar(t))
    x_t = torch.randn(2, 32, 32, generator=torch.Generator().manual_seed(t))
    expected = (1 - a) ** 0.5 * (x_t - a ** 0.5 * 0.4) / (a * 0.2**2 + 1 - a)
    assert torch.allclose(adapter.predict_noise(x_t, t), expected, rtol=1e-4, atol=1e-5)


def test_the_adapter_refuses_an_image_size_the_network_was_not_built_for(adapter):
    with pytest.raises(ValueError, match="32"):
        adapter.predict_noise(torch.zeros(1, 64, 64), 10)


def test_the_ve_schedule_spans_the_trained_noise_range_log_uniformly():
    sched = ve_schedule(0.01, 128.0)
    sigma = ((1 - sched.alphas_cumprod) / sched.alphas_cumprod).sqrt()
    assert sigma[0] == pytest.approx(0.01, rel=1e-3)
    assert sigma[-1] == pytest.approx(128.0, rel=1e-3)
    ratios = sigma[1:] / sigma[:-1]
    assert torch.allclose(ratios, ratios.mean(), rtol=1e-3)
    assert (sched.alphas_cumprod[1:] < sched.alphas_cumprod[:-1]).all()


def test_the_factory_picks_the_adapter_by_what_the_checkpoint_directory_holds(tmp_path):
    song = tmp_path / "brats_ncsnpp_ve"
    song.mkdir()
    torch.save({"config": TINY.to_dict(), "state_dict": NCSNpp(TINY).state_dict()},
               song / "ncsnpp_tiny.pt")
    assert isinstance(build_denoiser(song), SongVEScoreDenoiser)
    plain = tmp_path / "ddpm-celebahq-256"
    plain.mkdir()
    assert isinstance(build_denoiser(plain), DiffusersUNetDenoiser)  # loaded lazily


def test_the_brain_prior_is_recorded_as_brain_and_admitted_nowhere_yet():
    """Recorded as "mri" it would stamp knee banks reportable; whether the anatomy-matched
    case is admissible is Track 1's ruling (T2.10)."""
    assert prior_domain("checkpoints/song_inverse_problems/brats_ncsnpp_ve") == "mri_brain"
    assert not prior_is_admissible("checkpoints/song_inverse_problems/brats_ncsnpp_ve", "mri")


# -- the real checkpoint, where it is present -------------------------------------------

#: From Song et al.'s own Flax NCSN++ with checkpoint_26's params_ema, on
#: rng = np.random.default_rng(0); x = rng.uniform(0, 1, (3, 1, 240, 240)) as float32,
#: sigma = (0.05, 1.0, 50.0). Recorded in T2.10.
FLAX_REFERENCE = {
    0: (16.1265, [-32.0244, -8.20843, -10.6592, -30.7882]),
    1: (0.810083, [-0.080588, 1.00248, -0.778664, -0.170506]),
    2: (0.0164311, [-0.0448863, -0.000279603, -0.0177925, 0.00711996]),
}
PIXELS = [(0, 0), (60, 100), (120, 120), (200, 37)]


@pytest.mark.skipif(not BRATS_PT.is_file(), reason="converted BraTS checkpoint not present")
def test_the_port_reproduces_song_et_al_s_flax_model_on_the_real_checkpoint():
    payload = torch.load(BRATS_PT, weights_only=True)
    model = NCSNpp(NCSNppConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    x = np.random.default_rng(0).uniform(0, 1, size=(3, 1, 240, 240)).astype(np.float32)
    with torch.no_grad():
        out = model.eval()(torch.from_numpy(x), torch.tensor([0.05, 1.0, 50.0])).numpy()
    for i, (mean_abs, pixels) in FLAX_REFERENCE.items():
        assert np.abs(out[i]).mean() == pytest.approx(mean_abs, rel=1e-4)
        got = [out[i, 0, r, c] for r, c in PIXELS]
        np.testing.assert_allclose(got, pixels, rtol=1e-3, atol=1e-3 * mean_abs)
