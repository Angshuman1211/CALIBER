"""``err_free_two_solve``: the error column that keeps cos theta inside [-1, 1] (D94, T2.11 1j).

``cos theta`` is recovered from three region magnitudes by the law of cosines. The bank's
``err_free`` is a third null-space solve, on the difference. On CT that solve is a stopped
CG, which is not additive, so the three magnitudes need not be the sides of one triangle
and ``cos theta`` can leave [-1, 1]. ``err_free_two_solve`` is the difference of the two
solves the other columns already come from. These pin that it equals ``err_free`` for a
linear projector, that it bounds ``cos theta`` for a projector that is not additive, and
that the column is additive in the bank format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from caliber.scorebank import schema
from caliber.scoring.labels import compute_region_truth
from caliber.scoring.regions import RegionGrid


class _Projector:
    """Stands in for an operator: only ``project_null`` is read by the truth columns."""

    def __init__(self, additive: bool):
        self.additive = additive

    def project_null(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x)
        mask[..., ::2] = 1.0
        out = x * mask
        if not self.additive:
            # a gain that depends on the input's own size, lower on a small input: what a
            # stopped solver does, and the reason P_N(a) - P_N(b) != P_N(a - b)
            out = out * (0.7 + 0.3 * torch.tanh(torch.linalg.vector_norm(x) / 20.0))
        return out


def _cosines(operator, scale: float, seed: int = 0, recovered: bool = False):
    """``recovered``: the reconstruction is the truth rescaled, so its hidden content is
    parallel to the truth's. That is where the bank's values above 1 sit (T2.11 Sec. 1j)."""
    gen = torch.Generator().manual_seed(seed)
    grid = RegionGrid(image_shape=(32, 32), region_size=8)
    truth = torch.rand((32, 32), generator=gen) + 0.5
    noise = torch.randn((32, 32), generator=gen)
    recon = truth * (1.0 + scale) + 0.01 * scale * noise if recovered else truth + scale * noise
    t = compute_region_truth(recon[None], truth, operator, grid, ref_intensity=1.0)
    s = grid.reduce(operator.project_null(recon).abs(), how="rms").numpy().astype(float)
    c = t.free_content.numpy().astype(float)
    three = (s**2 + c**2 - t.err_free.numpy().astype(float) ** 2) / (2 * s * c)
    two = (s**2 + c**2 - t.err_free_two_solve.numpy().astype(float) ** 2) / (2 * s * c)
    return t, three, two


def test_two_solve_equals_err_free_for_a_linear_projector():
    t, three, two = _cosines(_Projector(additive=True), scale=0.2)
    assert torch.allclose(t.err_free_two_solve, t.err_free, atol=1e-6)
    assert np.allclose(three, two, atol=1e-5)


@pytest.mark.parametrize("scale", [0.002, 0.02, 0.2])
def test_two_solve_cos_theta_stays_in_range_when_the_projector_is_not_additive(scale):
    t, _, two = _cosines(_Projector(additive=False), scale=scale)
    assert np.all(np.abs(two) <= 1.0 + 1e-6)
    # the third solve is a different number
    assert not torch.allclose(t.err_free, t.err_free_two_solve, rtol=1e-2)


def test_three_solve_form_leaves_the_range_where_the_reconstruction_is_close():
    _, three, two = _cosines(_Projector(additive=False), scale=0.05, recovered=True)
    assert np.any(np.abs(three) > 1.0 + 1e-4)
    assert np.all(np.abs(two) <= 1.0 + 1e-6)


def test_err_free_itself_is_not_replaced():
    op = _Projector(additive=False)
    gen = torch.Generator().manual_seed(3)
    grid = RegionGrid(image_shape=(32, 32), region_size=8)
    truth = torch.rand((32, 32), generator=gen) + 0.5
    recon = truth + 0.1 * torch.randn((32, 32), generator=gen)
    t = compute_region_truth(recon[None], truth, op, grid, ref_intensity=1.0)
    third = grid.reduce(op.project_null(recon - truth).abs(), how="rms")
    assert torch.allclose(t.err_free, third.to(torch.float32))


def test_the_column_is_in_the_format_and_nullable():
    assert schema.COLUMNS["err_free_two_solve"] == "float32"
    assert "err_free_two_solve" in schema.NULLABLE_COLUMNS
    row = {name: (0 if dtype.startswith("int") else 0.5 if dtype.startswith("float") else "x")
           for name, dtype in schema.COLUMNS.items()}
    old = pd.DataFrame([row]).drop(columns=["err_free_two_solve"])   # a bank written before it
    out = schema.coerce(old)
    assert np.isnan(out["err_free_two_solve"].iloc[0])
    assert out["err_free"].iloc[0] == pytest.approx(0.5)
