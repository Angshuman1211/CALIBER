"""
Synthetic check of T3.3 assumption (ii): is the T1.1-style hallucination score
Lipschitz in the operator distance d_op (aggregate null-space projection
distance, ratified in T1.3)?

Self-contained: does not depend on the real caliber package, and does not call
src/caliber/scoring/score.py directly -- this is a proxy score for a first-pass
directional check, not a validation of T1.1's actual component implementation.
Uses a synthetic 1D masked-sampling operator (same structure as MRI k-space
masking, for which d_op has the exact closed form
sqrt(symmetric_difference)/sqrt(n)) and a reference score matching T1.1's
three-part design (null-space energy, cross-sample disagreement, measurement
residual), with a smoothness-informed prior fill (linear interpolation from
measured neighbours + draw noise) instead of pure noise, so the score actually
depends on image content and on which indices the operator hides -- not just
on i.i.d. randomness.

Note on scope: T1.1's score.py (D25) makes the three-component combination an
affine, non-negative weighted sum specifically so that the combined score is
Lipschitz whenever each component is -- see combine_components() there, which
cites this task by name. This script tests the combined proxy score, not the
three components separately, so a finding here about the combined behaviour
does not by itself say which component is responsible -- see
Documentation/T3.3_stability_result_L1.md for the resulting discussion.
"""
import numpy as np

N = 256


_PERM = np.random.default_rng(42).permutation(N)  # one fixed ordering -> nested mask family


def make_mask(keep_frac):
    """Nested by construction: mask(R) subset-of mask(R') whenever R > R'."""
    k = int(round(keep_frac * N))
    mask = np.zeros(N, dtype=bool)
    mask[_PERM[:k]] = True
    return mask


def op_distance(mask_a, mask_b):
    """Closed form for diagonal mask operators (matches src/caliber/operators/distance.py)."""
    sym_diff = np.sum(mask_a ^ mask_b)
    return np.sqrt(sym_diff) / np.sqrt(N)


def smooth_image(seed, smoothness=9):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(N)
    kernel = np.ones(smoothness) / smoothness
    padded = np.concatenate([raw[-(smoothness // 2):], raw, raw[:smoothness // 2]])
    smoothed = np.convolve(padded, kernel, mode="valid")[:N]
    return (smoothed - smoothed.mean()) / smoothed.std()


def score(x, mask, rng, n_draws=8, noise_std=0.05, fill_noise_std=0.3):
    positions = np.arange(N)
    meas_pos = positions[mask]
    meas_vals = x[mask] + rng.normal(0, noise_std, size=meas_pos.shape)

    # circular linear interpolation from measured neighbours -> smoothness-informed prior mean
    ext_pos = np.concatenate([meas_pos - N, meas_pos, meas_pos + N])
    ext_vals = np.concatenate([meas_vals, meas_vals, meas_vals])
    mean_fill_all = np.interp(positions, ext_pos, ext_vals)

    null_pos = positions[~mask]
    mean_fill_null = mean_fill_all[null_pos]
    draws = mean_fill_null[None, :] + rng.normal(0, fill_noise_std, size=(n_draws, null_pos.size))

    null_energy = np.mean(draws ** 2)
    disagreement = np.mean(np.var(draws, axis=0))

    recon_meas = meas_vals[None, :] + rng.normal(0, noise_std, size=(n_draws, meas_pos.size))
    residual = np.mean((recon_meas - meas_vals[None, :]) ** 2)

    return null_energy + disagreement + residual


def main():
    cal_mask = make_mask(0.25)  # R=4 calibration operator
    accel_factors = [4, 5, 6, 8, 12, 16, 20, 32, 64]  # R=4 is the identical / zero-distance check

    n_x_trials = 600

    print(f"{'R':>4} {'d_op':>8} {'E|delta|':>10} {'ratio':>10}")
    results = []
    for R in accel_factors:
        keep_frac = 1.0 / R
        deploy_mask = make_mask(keep_frac)
        d = op_distance(cal_mask, deploy_mask)
        abs_deltas = []
        for t in range(n_x_trials):
            x = smooth_image(seed=t)
            rng_cal = np.random.default_rng(10_000 + t)
            rng_dep = np.random.default_rng(10_000 + t)  # same seed: paired evaluation noise
            s_cal = score(x, cal_mask, rng_cal)
            s_dep = score(x, deploy_mask, rng_dep)
            abs_deltas.append(abs(s_dep - s_cal))
        e_abs_delta = float(np.mean(abs_deltas))
        ratio = e_abs_delta / d if d > 1e-9 else float("nan")
        print(f"{R:>4} {d:>8.4f} {e_abs_delta:>10.5f} {ratio:>10.4f}")
        results.append((R, d, e_abs_delta, ratio))
    return results


if __name__ == "__main__":
    main()
