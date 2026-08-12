"""이 세션에서 실제로 돌린 계산의 커버리지 지도 — 커밋된 CSV에서만 읽는다.

추정하지 않는다. 칸의 내용은 해당 CSV에 실제로 존재하는 (arm, mode, seed) 조합에서
직접 세고, 없는 칸은 비워 둔다.
"""
from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

from eigen_benchmark.drivers import manifest         # noqa: E402

D = "docs/_generated/data/paper2"
OUT = "docs/_out/paper2-computation-coverage.png"

FAM = {"classical": "#1b5e20", "randfeat": "#0d47a1", "gradient": "#b71c1c",
       "none": "#e0e0e0"}


def rd(name):
    return manifest.read_csv(os.path.join(D, f"{name}.csv"))


def _span(rows, key="mode"):
    v = sorted({int(float(r[key])) for r in rows if r.get(key) not in (None, "")})
    if not v:
        return ""
    return f"{v[0]}" if len(v) == 1 else f"{v[0]}-{v[-1]}"


def _seeds(rows, keys=("n", "n_seeds")):
    for k in keys:
        s = {int(float(r[k])) for r in rows if r.get(k) not in (None, "")}
        if s:
            return max(s)
    return None


# ---------------------------------------------------------------- 패널 A 자료
ms = rd("p1_neural_mode_sweep")
pen = rd("p1_neural_penalty_sweep")
p4n = rd("p4_neural")
st = rd("p1_spacetime")
col = rd("p1_neural_collinearity")


def cell(rows, fam, extra=""):
    if not rows:
        return None
    s = _seeds(rows)
    txt = f"modes {_span(rows)}"
    if s:
        txt += f"\n{s} seeds"
    if extra:
        txt += f"\n{extra}"
    return (txt, fam)


def arm_rows(arm_prefix):
    return [r for r in ms if str(r["arm"]).startswith(arm_prefix)]


P1_DOF = sorted({int(r["n_dof"]) for r in rd("p1_basis_study")})
P2_DOF = sorted({int(r["n_dof"]) for r in rd("p2_basis_study")})
P3_DOF = sorted({int(r["n_dof"]) for r in rd("p3_basis_study")})
P4_GRID = sorted({int(r["n"]) for r in rd("p4_convergence")})
NB1 = len({r["basis"] for r in rd("p1_basis_study")})
NB2 = len({r["basis"] for r in rd("p2_basis_study")})
NB3 = len({r["basis"] for r in rd("p3_basis_study")})
NB4 = len({r["solver"] for r in rd("p4_classical_vs_neural")
           if r["family"] == "classical"})
P4_DOF = sorted({int(r["n_dof"]) for r in rd("p4_convergence")})

ROWS = [
    ("classical admissible bases",
     (f"{NB1} families\nN = {P1_DOF[0]}-{P1_DOF[-1]}", "classical"),
     (f"{NB2} families\nN = {P2_DOF[0]}-{P2_DOF[-1]}", "classical"),
     (f"{NB3} families\nN = {P3_DOF[0]}-{P3_DOF[-1]}\n4 stiffnesses", "classical"),
     (f"graded Q2, {NB4} gradings\nn = {P4_GRID[0]}-{P4_GRID[-1]}"
      f"\n{P4_DOF[0]}-{P4_DOF[-1]} dof", "classical")),
    ("(a) penalty deflation",
     cell(pen, "gradient", "7 weights 1e-1..1e5"), None, None, None),
    ("(a') diagonal-approx. projection",
     cell(arm_rows("a_prime"), "gradient"), None, None, None),
    ("(b) exact M-orthogonal projection",
     cell(arm_rows("b_projection"), "gradient"), None, None,
     cell(p4n, "gradient", "2-D vector field")),
    ("(c) coarse-Ritz curriculum",
     cell(arm_rows("c_curriculum"), "gradient"), None, None, None),
    ("(d) separate-net basis (M=9)",
     cell(arm_rows("d_neural"), "gradient"), None, None, None),
    ("(e) shared-trunk subspace",
     cell(arm_rows("e_simultaneous"), "gradient", "2 objectives"),
     None, None, None),
    ("(f) Eig-PIELM (random features)",
     cell(arm_rows("f_eig_pielm"), "randfeat", "n_f = 20-160"), None, None, None),
    ("(g) space-time PINN",
     (f"mode 1\n{len(st) // len({r['n_periods'] for r in st})} seeds"
      f"\n4 windows", "gradient"), None, None, None),
]
COLS = ["P1\nEuler-Bernoulli beam", "P2\nannular plate",
        "P3\nspring interface", "P4\nL-shaped domain"]

# ---------------------------------------------------------------- 패널 B 자료
orth = rd("p1_orthogonality")
wc = rd("p1_neural_warm_cold")
oab = rd("p1_neural_orth_ablation")
SWEEPS = [
    ("penalty weight w",
     sorted({float(r["penalty_weight"]) for r in pen}), "log", "gradient"),
    ("orthogonality penalty w_orth",
     sorted({float(r["w_orth"]) for r in oab if float(r["w_orth"]) > 0}),
     "log", "gradient"),
    ("iteration budget",
     [200, 400, 4000, 20000], "log", "gradient"),
    ("stored-mode depth k (Gram diagnostics)",
     sorted({int(float(r["n_prev_modes"])) for r in orth
             if r.get("kappa_gram_median") not in (None, "")}), "linear",
     "gradient"),
    ("time window (periods of mode 1)",
     sorted({float(r["n_periods"]) for r in st}), "log", "gradient"),
    ("interface stiffness k-hat (P3)",
     sorted({float(r["k_hat"]) for r in rd("p3_basis_study")}), "log",
     "classical"),
    ("random features n_f",
     sorted({int(str(r["arm"]).split("nf=")[-1].rstrip(")"))
             for r in ms if str(r["arm"]).startswith("f_eig")}), "log",
     "randfeat"),
]
LADDER = sorted({str(r["ladder"]) for r in wc})


EXPERIMENTS = [
    ("mode sweep", "arm x mode 1-10, 50 seeds", "p1_neural_mode_sweep", "gradient"),
    ("penalty weight sweep", "7 decades x modes 1-3", "p1_neural_penalty_sweep",
     "gradient"),
    ("quadrature control", "Gauss vs Monte-Carlo, same arm",
     "p1_neural_quadrature + audit", "gradient"),
    ("initialization ladder", "I0-I5 x 3 budgets", "p1_neural_warm_cold(+200,+400)",
     "gradient"),
    ("long-run budget", "failing cells at 20000 iters", "p1_neural_longrun",
     "gradient"),
    ("Gram diagnostics", "stored depth k = 2-10, both rules", "p1_orthogonality",
     "gradient"),
    ("collinearity + orth. penalty", "(d),(e) ranks x 4 weights",
     "p1_neural_collinearity, _orth_ablation", "gradient"),
    ("objective ablation", "trace vs log-trace on (e)",
     "p1_e_objective_ablation", "gradient"),
    ("space-time PINN", "4 windows x 3 seeds", "p1_spacetime(_bins)", "gradient"),
    ("single-seed cost", "direct timing, median of three",
     "p1_single_seed_cost, p1p4_..._repeats", "gradient"),
    ("basis + conditioning studies", "P1/P2/P3 families x sizes",
     "p{1,2,3}_basis_study, _conditioning", "classical"),
    ("precision + quadrature separation", "fp64 vs mp, rule vs error",
     "p3_precision_fp64_vs_mp, p3_quadrature_separation", "classical"),
    ("degeneracy scoring", "nodal-diameter pairs on P2", "p2_degeneracy",
     "classical"),
    ("reference values", "analytic / Bessel / transfer-matrix / FEM+Richardson",
     "p{1,2,3,4}_reference, p4_convergence", "classical"),
]


def draw():
    fig = plt.figure(figsize=(13.2, 13.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.45, 0.98, 1.62],
                          hspace=0.24)
    ax = fig.add_subplot(gs[0])
    nr, nc = len(ROWS), len(COLS)
    for i, row in enumerate(ROWS):
        label, cells = row[0], row[1:]
        y = nr - 1 - i
        ax.text(-0.06, y + 0.5, label, ha="right", va="center", fontsize=9.5)
        for j, c in enumerate(cells):
            if c is None:
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=FAM["none"],
                                       edgecolor="white", lw=1.5, hatch="///",
                                       alpha=0.55))
                continue
            txt, fam = c
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=FAM[fam],
                                   edgecolor="white", lw=1.5, alpha=0.88))
            ax.text(j + 0.5, y + 0.5, txt, ha="center", va="center",
                    fontsize=8.2, color="white", linespacing=1.35)
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)])
    ax.set_xticklabels(COLS, fontsize=9.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("What was actually run: solver arm x test problem",
                 fontsize=12, pad=34, loc="left", x=-0.235)
    h = [Rectangle((0, 0), 1, 1, facecolor=FAM[k], alpha=0.88)
         for k in ("classical", "randfeat", "gradient")]
    h.append(Rectangle((0, 0), 1, 1, facecolor=FAM["none"], hatch="///",
                       alpha=0.55))
    ax.legend(h, ["classical (deterministic solve)",
                  "random features (no backprop)",
                  "gradient-trained", "not run - no claim made"],
              loc="lower left", bbox_to_anchor=(-0.235, -0.085), ncol=4,
              fontsize=8.6, frameon=False)

    bx = fig.add_subplot(gs[1])
    for i, (name, vals, scale, fam) in enumerate(SWEEPS):
        y = len(SWEEPS) - 1 - i
        v = [v for v in vals if v > 0]
        xs = [math.log10(x) for x in v] if scale == "log" else list(v)
        lo, hi = min(xs), max(xs)
        span = (hi - lo) or 1.0
        nx = [(x - lo) / span for x in xs]
        bx.plot([0, 1], [y, y], color="#cccccc", lw=1.0, zorder=1)
        bx.scatter(nx, [y] * len(nx), s=34, color=FAM[fam], zorder=3)
        bx.text(-0.012, y, name, ha="right", va="center", fontsize=9)
        fmt = (lambda z: f"{z:g}")
        bx.text(1.012, y, f"{fmt(min(v))} -> {fmt(max(v))}   ({len(v)} levels)",
                ha="left", va="center", fontsize=8.6, color="#444444")
    y = -1
    bx.plot([0, 1], [y, y], color="#cccccc", lw=1.0, zorder=1)
    bx.scatter([k / (len(LADDER) - 1) for k in range(len(LADDER))],
               [y] * len(LADDER), s=34, color=FAM["gradient"], zorder=3)
    bx.text(-0.012, y, "initialization ladder", ha="right", va="center",
            fontsize=9)
    bx.text(1.012, y, ", ".join(LADDER) + f"   ({len(LADDER)} levels)",
            ha="left", va="center", fontsize=8.6, color="#444444")
    y = -2
    bx.plot([0, 1], [y, y], color="#cccccc", lw=1.0, zorder=1)
    bx.scatter([0.0, 1.0], [y, y], s=34, color=FAM["gradient"], zorder=3)
    bx.text(-0.012, y, "quadrature rule", ha="right", va="center", fontsize=9)
    bx.text(1.012, y, "Gauss 256 -> Monte-Carlo   (2 levels)", ha="left",
            va="center", fontsize=8.6, color="#444444")
    bx.set_xlim(0, 1)
    bx.set_ylim(-2.55, len(SWEEPS) - 0.45)
    bx.set_xticks([])
    bx.set_yticks([])
    for s in bx.spines.values():
        s.set_visible(False)
    bx.set_title("Axes that were swept (dots = levels actually present in the "
                 "data; spacing is to scale)",
                 fontsize=11, pad=10, loc="left", x=-0.235)

    cx = fig.add_subplot(gs[2])
    n = len(EXPERIMENTS)
    for i, (name, what, files, fam) in enumerate(EXPERIMENTS):
        y = n - 1 - i
        cx.add_patch(Rectangle((0.0, y - 0.36), 0.008, 0.72,
                               facecolor=FAM[fam], edgecolor="none"))
        cx.text(0.020, y, name, ha="left", va="center", fontsize=9.2)
        cx.text(0.335, y, what, ha="left", va="center", fontsize=8.8,
                color="#333333")
        cx.text(0.70, y, files, ha="left", va="center", fontsize=8.0,
                color="#777777", family="monospace")
    cx.set_xlim(0, 1.0)
    cx.set_ylim(-0.6, n - 0.35)
    cx.set_xticks([])
    cx.set_yticks([])
    for s_ in cx.spines.values():
        s_.set_visible(False)
    cx.set_title("Experiments behind the map (what varied, and the artifact it "
                 "wrote)", fontsize=11, pad=10, loc="left", x=0.0)
    cx.set_facecolor("white")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(draw())
