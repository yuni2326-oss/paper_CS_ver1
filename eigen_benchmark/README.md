# eigen_benchmark — computation for the eigensolver reliability benchmark

Computation code for a benchmark that measures the **reliability and error attribution** of
Rayleigh-quotient neural eigensolvers against stabilized classical bases.

The design record and the scope boundary document are internal and are not part of this
distribution. The rules that matter are summarized under "Design contract" below, and they
are enforced by `tests/` rather than by prose.

## Design contract

Every basis and every solver exchanges **only the discrete (K, M) that come from one weak
form per problem**. That is what makes `e_model` zero by construction in the decomposition
`e_total = e_model + e_approx + e_algebraic + e_optimization`, leaving the other three
components as the measured quantities.

- `problems/*` knows nothing about bases. `bases/*` knows nothing about problems.
- `metrics.py`, `conditioning.py` and `cost.py` are pure functions: arrays in, scalars or
  dicts out.
- The classical side uses only NumPy/SciPy/mpmath. The neural arms add PyTorch and need a
  CUDA GPU; nothing else does.

## Running

```bash
export PYTHONPATH=$PWD

# the suite — 359 host tests; the GPU tests need a CUDA container with PyTorch
python -m pytest eigen_benchmark/tests -q

# regenerate the tables: one driver per problem produces all of that problem's tables
for d in p1 p2 p3 p4; do python -m eigen_benchmark.drivers.run_$d; done
# p2 and p3 take minutes to ~20 min each because of the mpmath high-precision root finding
```

`main(outdir=..., quick=True)` is a reduced-grid smoke run; the tests use it.

## Scope

P3 uses the rotational spring **only as a generic interface benchmark** and never
parameterizes it by a damage mechanism such as a crack depth.

**Dimensionless rule.** No absolute dimensions, material properties or frequencies appear
anywhere. Nothing is lost, because every conclusion is a dimensionless quantity — Λ, e_λ,
MAC, principal angles, condition numbers, success probabilities, cost ratios — and quoting
absolute dimensions would tie the test problems to one particular machine. P2 is fixed by
a/b and ν alone (with b = 1), and P4 sets L = E = ρ = 1 so the solver returns
Λ_el = ω²ρL²/E directly. `tests/test_paper_boundary.py` enforces this: the suite breaks if
absolute units or a damage mechanism re-enter.

Artifacts are written to `docs/_generated/data/paper2/`. Each driver records the git SHA,
branch, library versions, platform and precision in its own manifest JSON.

## Numerical facts established while implementing this

Trusting the computation means knowing that the following are not accidents. Every row is
pinned by a regression test.

| fact | evidence |
|---|---|
| `1 + cos β cosh β` suffers catastrophic cancellation by β₁₀ ≈ 29.8 → **use `cos β + sech β`** | agrees with the mpmath dps=50 roots to 1e-14 relative |
| **A C¹ global basis makes ⟦u′⟧ ≡ 0 on P3, so the k_θ term cancels identically** → refining the space converges to the *uniform* beam, not the spring beam | the error against the spring reference stays flat at 6.3e-2 from 12 to 24 degrees of freedom |
| A ramp enrichment has zero bending energy in the broken form → using it on P1 introduces a **hinge mechanism** and K becomes singular | the enriched bases are P3-only |
| Raw monomials lose fp64 Cholesky of the mass matrix from N = 12, the smallest size tested (Hilbert-like M) | recorded as `cholesky_ok=False` plus NaN rather than raised as an exception |
| Orthonormalizing the same span restores **factorability** but not accuracy | the parent basis has already lost rank when it is evaluated in fp64 |
| Integrated-Legendre bubbles must start at **L₂** | L₀ and L₁ vanish identically once the cubic Hermite correction is removed, so a zero function enters the basis and produces a spurious λ ≈ 0 |
| Quadrature that is not aligned to the basis break points underestimates the stiffness and **breaks the Ritz upper-bound property** | `Basis.breaks` aligns it automatically; misalignment is kept only as a control |
| Fewer Gauss points than basis degrees of freedom makes M rank-deficient by construction | the quadrature order is floored at `n_dof` |
| A high-precision solve **cannot recover information lost at assembly** | the fp64 Hilbert(14) matrix is not positive definite even in exact arithmetic, so the mpmath Cholesky refuses it and the general pencil path is reported instead |
| **The saturation gap of the C¹ bases follows the expected 1/k̂ scaling** to the reported precision | 2.108, 0.2106, 0.02105, 0.002104 at k̂ = 1, 10, 100, 1000 (ratio 10.00) — the `e_approx` floor tracks the first-order effect of interface compliance |
| An fp64 function cannot be refined by mpmath | the residual stalls at 1e-50, so the Bessel reference needs a **separate high-precision path** built on mpmath recurrences |
| Placing Q2 midside nodes at the **graded parameter midpoint** degenerates the mapping | for β ≥ 2 the node leaves the middle half of the edge and the Jacobian turns non-positive; vertices are graded, midside nodes stay at the geometric midpoint |
| **Over-grading is counterproductive** | on P4, β = 5 is worse than β = 3 (6/6 versus 1/6 modes meeting the target). Accuracy is governed by refinement level rather than by grading strength |
| Mirror symmetry (Z₂) does not force degeneracy | it has one-dimensional irreducible representations only; double roots are a property of non-abelian groups such as C₄ᵥ. The correct consequence is that each mode is symmetric or antisymmetric |
| The Bessel 4×4 determinant **requires column scaling** | I_m and K_m differ by 10¹¹ at kb ≈ 26, so without scaling the determinant is meaningless through cancellation |

## Layout

| path | responsibility |
|---|---|
| `quadrature.py` | piecewise Gauss–Legendre, separating basis order p from quadrature order n_q |
| `bases/base.py` | basis protocol (`eval`/`d1`/`d2`/`d1_jump`/`breaks`) and coordinate-transform wrappers |
| `bases/monomial.py` | raw monomials and their orthonormalization |
| `bases/recurrence.py` | shifted Legendre, integrated Legendre, Chebyshev |
| `bases/bspline.py` | C⁰ B-splines with controlled continuity |
| `bases/enriched.py` | interface-enriched and split-domain bases |
| `bases/fem.py` | Hermite C¹ beam elements with a duplicated rotation DOF at x_c |
| `bases/radial_fem.py` | radial Hermite C¹ elements for P2 (r-dependent coefficients, so per-element Gauss quadrature) |
| `reference/transfer_matrix.py` | transfer-matrix exact reference (fp64 scan refined by mpmath) |
| `reference/bessel_annulus.py` | P2 exact Bessel reference (4×4 determinant, column scaling, mpmath recurrences) |
| `problems/p1_beam.py` | P1 prismatic beam — analytic reference for modes 1–14, (K, M) assembly |
| `problems/p2_annulus.py` | P2 annular Kirchhoff plate — one 1-D radial problem per nodal diameter m |
| `problems/p3_spring.py` | P3 zero-width rotational spring, single broken-H² weak form, k̂ ∈ {1, 10, 100, 1000} |
| `problems/p4_lshape.py` | P4 L-shaped plane elasticity — three-block graded Q2 mesh with Richardson extrapolation |
| `neural/` | the neural arms: shared Rayleigh-quotient engine, deflation strategies, curriculum, subspace objectives, Eig-PIELM, space–time PINN, 2-D P4 arm |
| `degeneracy.py` | degenerate eigenspace construction in (r, θ) and the subspace metrics |
| `conditioning.py` | four normalizations of κ, backward error, Cholesky, fp64 versus mpmath |
| `metrics.py` | e_λ, MAC, r_h, orthogonality, principal angles, the **pre-specified mode classification**, Wilson intervals |
| `cost.py` | t(ε) line items and E[T_success] |
| `render/` | committed CSV → markdown tables and PNG figures. Computes nothing |
| `drivers/` | per-problem regeneration drivers and manifests |

## Values fixed before the runs

`metrics.MAC_MIN = 0.9` and `metrics.ELAM_MAX = 0.05`. They define the four outcome
categories (`correct` / `lower_mode_basin` / `spurious` / `non_converged`) and are **not
changed after seeing data**. The constants themselves are pinned by a test, so changing one
fails the suite.

There is no third-party timestamped registration of this study; the specification is in the
repository history, in which the commit fixing these thresholds precedes the first committed
benchmark data.
