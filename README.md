# Reliability and error attribution in Rayleigh-quotient neural eigensolvers — code and data

Computational artifacts for a benchmark of Rayleigh-quotient neural eigensolvers against
stabilized classical bases: solvers, committed data, the render pipeline and the regression
suite.

**This repository contains the computation only.** The manuscript and its internal research
records are not included; they live in a private repository. Everything needed to reproduce
every number, table and figure of the paper is here: solvers, drivers, committed data with
per-artifact manifests, the render pipeline, and the regression suite.

## Layout

| path | contents |
|---|---|
| `eigen_benchmark/` | the library: test problems, classical bases, neural arms, metrics, conditioning, drivers |
| `eigen_benchmark/tests/` | regression suite, 359 host tests + 71 GPU tests |
| `docs/_generated/data/paper2/` | every committed CSV/JSONL, each with a JSON manifest (git SHA, library versions, platform, precision) |
| `docs/_generated/` | figure and document generators |

## Test problems

All four are dimensionless. Absolute dimensions, material properties and frequencies are
deliberately absent, so no conclusion is tied to one machine.

- **P1** clamped–free Euler–Bernoulli beam, analytic reference, modes 1–10
- **P2** annular Kirchhoff plate with degenerate nodal-diameter pairs, exact Bessel reference
- **P3** zero-width rotational spring in a broken-H² weak form that every solver discretizes
  identically
- **P4** L-shaped plane-stress domain, `Ω = (0,2)² \ ([1,2] × [1,2])`, reference by
  Richardson-extrapolated graded Q2 FEM

## Solver arms

`(a)` penalty deflation · `(a′)` diagonal-Gram subtraction map · `(b)` exact M-orthogonal
projection · `(c)` coarse-Ritz curriculum · `(d)` separate-net neural basis ·
`(e)` shared-trunk subspace · `(f)` Eig-PIELM random features · `(g)` space–time PINN.
Fifty seeds per stochastic cell, with the outcome classification, convergence rule and
thresholds fixed before the runs.

## Reproducing

```
export PYTHONPATH=$PWD

# figures that are generated from equations rather than from the data
python docs/_generated/make_fig1_benchmark_problems.py

# tables and result figures, from the committed CSVs
python -m eigen_benchmark.drivers.render_paper

# the suite (GPU tests need a CUDA container with PyTorch)
python -m pytest eigen_benchmark/tests -q

# record the execution environment
python -m eigen_benchmark.drivers.run_environment
```

Generated binaries (PNG, PDF, DOCX) are not committed, so the figure generators must run
once in a fresh clone. The renderers fail loudly and name the command to run rather than
skipping silently.

Tests that check manuscript-to-data consistency skip here with a stated reason — the
manuscript is not part of this distribution. Everything else runs.

Code paths that name the manuscript resolve it at runtime and degrade to a skip when it is
missing; nothing depends on it being present.

## Recomputing from scratch

The drivers below rewrite the committed data. Neural arms need a CUDA GPU; the classical
side, the references and the render pipeline are CPU-only.

```
python -m eigen_benchmark.drivers.run_p1          # classical bases, conditioning, references
python -m eigen_benchmark.drivers.run_p2          # annulus, degeneracy scoring
python -m eigen_benchmark.drivers.run_p3          # spring interface
python -m eigen_benchmark.drivers.run_p4          # L-shape reference, Richardson
python -m eigen_benchmark.drivers.run_p1_neural   # neural arms (GPU)
python -m eigen_benchmark.drivers.run_p4_neural   # 2-D exact-projection arm (GPU)
python -m eigen_benchmark.drivers.run_p1_spacetime
python -m eigen_benchmark.drivers.run_p1_compare  # accuracy-versus-cost, canonical ratios
```

## Conventions worth knowing before reading the code

- **The renderers compute nothing.** `eigen_benchmark/render/` turns committed CSVs into
  markdown tables and PNGs. A number that is not in a CSV cannot reach the paper; adding one
  means fixing a driver first.
- **`render_paper --check` fails the build** if any rendered number disagrees with the data.
  That is the mechanism behind every quoted figure.
- **Every artifact carries its own manifest** with the git SHA of the commit that produced
  it, the library versions, the platform and the precision.
- **fp64 throughout.** NumPy defaults to float64; the Torch solvers call
  `set_default_dtype(torch.float64)` at entry because the process default is float32.
  High-precision references use mpmath at `dps=50`.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE` for the full text,
or https://creativecommons.org/licenses/by/4.0/ for a summary. `SPDX-License-Identifier:
CC-BY-4.0`.

You may share and adapt this material for any purpose, including commercially, provided you
give appropriate credit, link to the license, and indicate whether changes were made.

Please cite the accompanying paper when you use this code or data. Note that CC BY 4.0 grants
no patent rights and carries no source-distribution requirement; it covers the data, the
documentation and the code in this repository alike.
