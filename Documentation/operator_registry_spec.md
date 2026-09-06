# Operator Registry Specification v0

**Status:** draft for Week 1 sign-off by all three tracks
**Owner:** Track 2 (T2.1) with Track 1 (T1.3) and Track 3 (T3.2)

Blueprint references: Sec. 19 (shared interfaces), Sec. 12.1 (distance
candidates), Week 9 card (consistency audit).

---

## 1. Why this exists

Blueprint Sec. 19 requires **one definition of each acquisition operator and of
the distance between operators, used identically in both modalities and in the
derivations.** The Week 9 card turns that into an explicit audit: confirm that
the same registered operator representation and distance measure are used
consistently across MRI, CT, and the theory (T1.3 / T3.2).

That audit is only passable if there is a single place operators are defined.
This is that place. All operator construction goes through
`caliber.operators.registry`, and nothing constructs an operator inline.

## 2. What an operator must provide

Every operator implements `ForwardOperator`:

| Method | Meaning |
|---|---|
| `forward(x)` | `A x` — image to measurements |
| `adjoint(y)` | `A^H y` — the exact transpose |
| `pinv(y)` | `A^+ y` — exact where possible, best available otherwise; the *display* reconstruction |
| `decomposition_pinv(y)` | The inverse the decomposition below is built from. Defaults to `pinv`; overridden when `pinv` is not a genuine pseudo-inverse and a better one exists for this purpose specifically (CT, D-012) |
| `descriptor()` | Data-free summary registered here |

and inherits the decomposition of Sec. 2.3, built from `decomposition_pinv`,
**not** `pinv`, so that a display-only approximation (CT's FBP) never leaks
into `project_range` / `project_null`:

```
project_range(x) = decomposition_pinv(A x)          the measured component
project_null(x)  = x - project_range(x)             the free component
```

## 3. The descriptor

`OperatorDescriptor` is the **data-free** summary. Track 3 computes distances
from descriptors and never touches a dataset or a GPU.

| Field | Meaning |
|---|---|
| `operator_id` | Stable, human-readable primary key |
| `modality`, `family` | `mri`/`ct`; mask or geometry family |
| `params` | Every parameter needed to rebuild the operator exactly |
| `image_shape`, `ambient_dim`, `measurement_dim` | Dimensions |
| `null_space_dim` | Size of the unmeasured subspace |
| `exact_pinv` | Whether `pinv(y)`, the display reconstruction, is the exact Moore-Penrose inverse — also gates the idempotence health check, unchanged by D-012 |
| `sampling_support` | MRI: k-space mask. CT: angle set |
| `singular_values` | Optional spectral summary |

### Operator ids

```
mri.<family>.R<acceleration>.c<center_fraction>.s<mask_seed>
ct.<family>.v<n_views>.a<angular_range>.d<dose>
```

Stable across runs and machines, and tested as such. An id is the join key in
the score bank, so it must never change meaning.

## 4. Two honest caveats, carried in the descriptor

These are recorded as data, not as comments, because they must reach the
write-up and the Assumptions and Honest Scope section (Sec. 13).

**`exact_pinv`.** For mask-based MRI, `M` is an orthogonal projection and `F`
is unitary, so `A^+ = F^H M` exactly and `P_R`, `P_N` are true orthogonal
projections; `exact_pinv` is `True`. For CT, `pinv` is filtered
backprojection, which is *not* the Moore-Penrose inverse — `exact_pinv` is
`False`, and any claim resting on the exactness of the *display*
reconstruction, or on idempotence of the decomposition, must be qualified for
CT.

**This is deliberately unchanged by D-012.** D-012 fixed CT's decomposition
being built from FBP — `decomposition_pinv` now uses conjugate gradient on the
normal equations (`CTProjectionOperator.pinv_cg`) instead, which reliably
fixes the *contraction* property (`null_projection_gain() <= 1`, verified
across the Sec. 16 operator grid, often from as few as 5-10 CG iterations, and
now a hard gate — see `contraction_ok` below). It does not fix idempotence:
CG is iterative and only approximately idempotent at any budget cheap enough
to run per item, and — because applying an incompletely-converged solve twice
compounds error in a way that doesn't simply shrink with more iterations of
the first solve — pushing further towards convergence is not the free lunch
it looks like, and past a point is not even safe (`pinv_cg`'s docstring has
the numerical-divergence story). So `exact_pinv` correctly stays `False` for
CT, and the idempotence gate stays bypassed for it, exactly as before D-012.

**`null_space_dim` for CT is a bound, not a rank.** It is the dimension count
`ambient - measurements`, recorded in `params` as
`null_space_dim_method: dimension_count`. The true rank needs a spectral
estimate (`caliber.operators.spectrum.estimate_spectrum`). This matters
directly for L2, which scales the bound with null-space dimension: L2 must
either use the estimated rank or state that it uses the bound.

## 5. Health checks — a gate, not a nicety

`OperatorRegistry.check_health()` runs before any score-bank production job and
the runner aborts on failure:

- **Adjoint test.** `<A x, v> == <x, A^H v>` to a relative tolerance.
- **Idempotence.** `P_R(P_R x) == P_R x`, required only where `exact_pinv` —
  MRI only; unchanged by D-012 (see §4).
- **Contraction.** `‖P_N x‖ / ‖x‖ <= 1`, required for every operator. Gated
  (not merely reported) since D-012 — this is the check the FBP-based CT
  decomposition used to fail.

A wrong adjoint, a non-idempotent projector, or an amplifying null projector
corrupts every score in the bank in a way that is invisible downstream — the
numbers stay plausible. All three checks are cheap; there is no reason not to
run them every time.

The CT adjoint is obtained as the vector-Jacobian product of the forward map.
Because the forward map is linear, the VJP **is** the exact transpose, so the
dot-product test passes to floating-point precision despite bilinear
interpolation in the projector.

## 6. Operator grids

Grids are declared in config; list-valued axes expand to the Cartesian product.

```yaml
- modality: mri
  family: [random1d, equispaced1d, variable_density2d]
  acceleration: [2, 3, 4, 6, 8, 10]
  center_fraction: 0.08
```

Grid axes are named in `registry.GRID_AXES`. MRI varies acceleration and mask
family; CT varies view count, angular range, and dose (Sec. 16).

One grid deserves attention: `mri_mask_family_shift` holds acceleration fixed
and varies only the mask family. Same undersampling budget, different null
space. It separates the *direction* of operator change from its *severity*,
which the acceleration sweep alone confounds — and the theoretical programme
needs that separation, since a distance measure that only tracks severity
cannot predict shifts that change direction.

## 7. Distance measures

All five Sec. 12.1 candidates are implemented in
`caliber.operators.distance` and selected by name in config:

| Name | Measure |
|---|---|
| `null_space_projection` | `‖P_N(A) − P_N(B)‖_F` |
| `sampling_support` | Symmetric difference of what is sampled |
| `chordal_subspace` | Principal-angle / chordal subspace distance |
| `operator_norm` | `‖A − B‖_F` |
| `spectral_divergence` | Difference in singular-value decay |

**Modality-agnosticism is a hard requirement** by Week 7: the same function
must accept an MRI pair and a CT pair. `null_space_projection` achieves this by
using a closed form when both operators expose a k-space mask — for masks,
`‖P_N(A) − P_N(B)‖_F` is exactly the square root of the mask symmetric
difference — and Hutchinson's estimator otherwise. The two paths are tested
against each other to within 15%.

Normalisation (`none` / `ambient` / `nullspace`) is configurable because Week 9
tests which choice makes the MRI and CT coverage-gap curves collapse onto a
shared trend. If none of them does, Sec. 21 R5 says revise the distance
definition before adding domains.

## 8. Open questions for Week 1 sign-off

- [ ] **Primary distance measure** (T1.3, locked Week 3). Working default is
      `null_space_projection` with `ambient` normalisation.
- [ ] **`calib_group` definition.** Currently derived in
      `runner.calibration_group`. Should it be a registry field instead, so
      Track 1 controls grouping without touching the runner?
- [ ] **Mask seeds.** Is a mask family at fixed acceleration one group or
      several? Affects both grouping and the meaning of a mask-family shift.
- [ ] **CT spectral estimation budget.** How many singular values does L2 need,
      and is randomised estimation accurate enough at the tail that matters?
