"""
T3.4 (L2) numerical check: does an ENERGY-WEIGHTED null-space fraction explain
the rise-then-plateau shape T3.3 found for E|delta score| / d_op, better than
raw null-space dimension does?

Motivation. `caliber.operators.spectrum` (decision D-011) already found that
fitting a singular-value decay exponent to A itself is unreliable and that L2
should prefer `null_space_dim` instead -- but for a masked-Fourier operator,
singular values are exactly {0, 1} (see `MRIUndersamplingOperator.descriptor`:
"all singular values are 1 or 0 for a mask operator"), so there is no decay to
fit in the first place, and raw null-space dimension carries no information
about WHICH frequencies were dropped, only how many. The score's free
component is `RMS(P_N(mean_recon))` (`caliber/scoring/score.py`), i.e. energy
of the reconstruction inside the null space -- so the quantity that should
actually control how much the score moves as A moves is not null-space
dimension but null-space ENERGY: how much of a typical image's spectral
energy falls into the newly-hidden frequencies.

This script computes, using the exact operators, exact masks and the exact
PhantomDataset population T3.3's real-score check used:
  - null_space_dim(R)              -- exact, dimension only
  - rho(R) = E_X[||P_N(A_R) X||^2] / E_X[||X||^2]   -- energy-weighted fraction,
    with E_X estimated empirically from PhantomDataset by Parseval (fft2 is
    orthonormal here, so ||P_N X||^2 in image domain = sum of |X_hat(k)|^2 over
    unmeasured k)
  - d_op(R) = null_space_projection_distance(cal_op, deploy_op)  -- same
    aggregate distance T3.3 used

and reports delta_rho(R)/d_op(R) alongside null_dim(R)/d_op(R), to see which
one tracks the shape of T3.3's observed ratio table (rises R=3->6, plateaus
through R=16).

Same sys.modules bypass as `t3_lipschitz_real_score_check.py` for the
unrelated `caliber.data.splits` import bug (Track 2, not touched here).
"""
import sys
import types

import torch

for name, path in [("caliber", "src/caliber"), ("caliber.data", "src/caliber/data")]:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m

from caliber.data.synthetic import PhantomDataset  # noqa: E402
from caliber.operators.mri import MRIUndersamplingOperator  # noqa: E402
from caliber.operators.distance import null_space_projection_distance  # noqa: E402

IMAGE_SHAPE = (128, 128)
CENTER_FRACTION = 0.04
MASK_SEED = 1  # matches t3_lipschitz_real_score_check.py's make_op default


def make_op(acceleration: float) -> MRIUndersamplingOperator:
    return MRIUndersamplingOperator(
        IMAGE_SHAPE, family="random1d", acceleration=acceleration,
        center_fraction=CENTER_FRACTION, mask_seed=MASK_SEED,
    )


def population_spectrum(n_items: int, seed: int) -> torch.Tensor:
    """E_X[|FFT(X)|^2], same fft2 convention as MRIUndersamplingOperator._fft."""
    phantoms = PhantomDataset(n_items=n_items, image_shape=IMAGE_SHAPE, seed=seed)
    total = torch.zeros(IMAGE_SHAPE, dtype=torch.float64)
    for i in range(n_items):
        x = phantoms[i].image
        xhat = torch.fft.fft2(x.to(torch.complex64), norm="ortho")
        total += xhat.abs().to(torch.float64) ** 2
    return total / n_items


def main():
    # Same population T3.3's real-score check drew its test items from
    # (PhantomDataset(seed=7)); use its full range here purely to estimate the
    # population spectrum, not to reproduce individual score values.
    s_bar = population_spectrum(n_items=200, seed=7)
    total_energy = float(s_bar.sum())

    cal_op = make_op(2.0)
    cal_mask = cal_op.mask
    rho_cal = float(s_bar[~cal_mask].sum()) / total_energy

    accel_sweep = [2, 3, 4, 6, 8, 12, 16]
    print(f"population energy in null space at calibration (R=2): rho_cal={rho_cal:.5f}\n")
    print(f"{'R':>4} {'null_dim':>9} {'null_frac':>10} {'rho':>8} {'d_op':>8} "
          f"{'d_rho/d_op':>11} {'d_nullfrac/d_op':>16}")

    ambient = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    rows = []
    for R in accel_sweep:
        deploy_op = cal_op if R == 2 else make_op(float(R))
        d = null_space_projection_distance(cal_op, deploy_op)
        null_dim = deploy_op.descriptor().null_space_dim
        null_frac = null_dim / ambient
        rho = float(s_bar[~deploy_op.mask].sum()) / total_energy

        d_rho = rho - rho_cal
        d_nullfrac = null_frac - (cal_op.descriptor().null_space_dim / ambient)
        ratio_rho = d_rho / d if d > 1e-9 else float("nan")
        ratio_nullfrac = d_nullfrac / d if d > 1e-9 else float("nan")

        print(f"{R:>4} {null_dim:>9} {null_frac:>10.4f} {rho:>8.4f} {d:>8.4f} "
              f"{ratio_rho:>11.4f} {ratio_nullfrac:>16.4f}")
        rows.append((R, null_dim, null_frac, rho, d, ratio_rho, ratio_nullfrac))
    return rows


if __name__ == "__main__":
    main()
