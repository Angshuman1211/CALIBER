"""T3.4 check: closed form for the free-component ratio under nested masks.

For nested masks d_op^2 = null_frac(dep) - null_frac(cal), and in the placeholder harness
s_free ~ c * sqrt(null_frac). So the free-component ratio should follow

    r(d) = (sqrt(a + d^2) - sqrt(a)) / d,   a = null_frac(cal)

up to a constant c. This prints r(d) next to the measured free ratios from
t3_illposedness_component_check.py, and their quotient, which should be flat if the
shape comes from the square root itself.
"""
from caliber.operators.distance import null_space_projection_distance
from caliber.operators.mri import MRIUndersamplingOperator

SHAPE = (128, 128)
MEASURED = {3: 0.375, 4: 0.454, 6: 0.523, 8: 0.546, 12: 0.569, 16: 0.582}  # T3.4 sec. 3


def make_op(r: float) -> MRIUndersamplingOperator:
    return MRIUndersamplingOperator(
        SHAPE, family="random1d", acceleration=r, center_fraction=0.04, mask_seed=1
    )


def main():
    cal = make_op(2.0)
    a = cal.descriptor().null_space_dim / (SHAPE[0] * SHAPE[1])
    print(f"{'R':>3} {'d_op':>7} {'closed':>8} {'measured':>9} {'ratio':>7}")
    for r, meas in MEASURED.items():
        dep = make_op(float(r))
        d = null_space_projection_distance(cal, dep)
        nf = dep.descriptor().null_space_dim / (SHAPE[0] * SHAPE[1])
        closed = (nf**0.5 - a**0.5) / d
        print(f"{r:>3} {d:>7.4f} {closed:>8.4f} {meas:>9.4f} {meas / closed:>7.3f}")


if __name__ == "__main__":
    main()
