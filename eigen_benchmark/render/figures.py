"""논문 그림 — CSV → PNG. 계산은 하지 않는다.

헤드리스 컨테이너·호스트 어디서든 돌아야 하므로 import 시점에 Agg를 강제한다.
축 라벨은 영문(제출용)이고 한글을 쓰지 않는다 — 폰트 누락으로 두부글자가 되는 것을 피한다.
"""
from __future__ import annotations

import math
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

from ..drivers import manifest       # noqa: E402
from .tables import ARM_LABEL, DATA_DIR, FAMILY_LABEL, _base   # noqa: E402

FIG_DIR = os.path.join("docs", "_generated", "fig", "paper2")
FAMILY_STYLE = {"classical": ("o", "#1b5e20"),
                "neural_randfeat": ("s", "#0d47a1"),
                "neural_gradient": ("^", "#b71c1c")}


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def _rows(data_dir, name):
    return manifest.read_csv(os.path.join(data_dir, f"{name}.csv"))


def fig1_mode_reliability(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """모드 1–10 × arm의 p_correct와 Wilson 구간. (a′)의 벽과 (b)의 무벽이 한눈에."""
    rows = _rows(data_dir, "p1_neural_mode_sweep")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for a in ("a_prime_projection_diagonal", "b_projection_exact", "c_curriculum",
              "d_neural_basis_galerkin", "e_simultaneous_subspace"):
        g = sorted((r for r in rows if _base(r["arm"]) == a),
                   key=lambda r: int(r["mode"]))
        if not g:
            continue
        m = [int(r["mode"]) for r in g]
        p = [float(r["p_correct"]) for r in g]
        lo = [p_ - float(r["wilson_lo"]) for p_, r in zip(p, g)]
        hi = [float(r["wilson_hi"]) - p_ for p_, r in zip(p, g)]
        ax.errorbar(m, p, yerr=[lo, hi], marker="o", capsize=2.5, lw=1.4, ms=4,
                    label=ARM_LABEL[a])
    ax.set_xlabel("target mode")
    ax.set_ylabel(r"$p_{\mathrm{correct}}$ (50 seeds, 95 % Wilson)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    return _save(fig, outdir, "fig1_mode_reliability.png")


def fig2_error_vs_dof(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """P3 공통 약형식의 기저별 e_λ vs n_dof. 포화가 보이는 그림."""
    rows = _rows(data_dir, "p3_basis_study")
    kc = sorted({float(r["k_hat"]) for r in rows})[len(set(
        float(r["k_hat"]) for r in rows)) // 2]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b in sorted({str(r["basis"]) for r in rows}):
        g = sorted((r for r in rows
                    if str(r["basis"]) == b and float(r["k_hat"]) == kc),
                   key=lambda r: int(r["n_dof"]))
        xy = [(int(r["n_dof"]), float(r["e_lam_mode1"])) for r in g
              if r["e_lam_mode1"] not in (None, "")
              and math.isfinite(float(r["e_lam_mode1"]))
              and float(r["e_lam_mode1"]) > 0]
        if len(xy) < 2:
            continue
        ax.loglog(*zip(*xy), marker="o", ms=3.5, lw=1.2, label=b)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel(r"relative eigenvalue error $e_\lambda$ (mode 1)")
    ax.set_title(rf"P3, $\hat k$ = {kc:g}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    return _save(fig, outdir, "fig2_error_vs_dof.png")


def fig3_accuracy_cost_pareto(data_dir: str = DATA_DIR,
                              outdir: str = FIG_DIR) -> str:
    """정확도-대-비용 평면. 세 계열과 Pareto 전선 — 논문 §5.4의 그림."""
    rows = _rows(data_dir, "p1_classical_vs_neural")
    par = _rows(data_dir, "p1_pareto")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for fam, (mk, col) in FAMILY_STYLE.items():
        g = [r for r in rows if r["family"] == fam and int(r["mode"]) == 1
             and float(r["p_correct"]) > 0
             and math.isfinite(float(r["e_lam"]))
             and float(r["e_lam"]) > 0]
        if not g:
            continue
        ax.scatter([float(r["seconds_per_attempt"]) for r in g],
                   [float(r["e_lam"]) for r in g],
                   marker=mk, s=26, alpha=0.75, color=col,
                   label=FAMILY_LABEL[fam])
    pf = sorted((r for r in par if int(r["pareto_mode"]) == 1),
                key=lambda r: float(r["seconds_per_attempt"]))
    if pf:
        ax.plot([float(r["seconds_per_attempt"]) for r in pf],
                [float(r["e_lam"]) for r in pf], "k--", lw=1.0, alpha=0.6,
                label="Pareto front")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock per attempt [s]")
    ax.set_ylabel(r"relative eigenvalue error $e_\lambda$")
    ax.set_title("P1 mode 1: accuracy vs cost", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right")
    return _save(fig, outdir, "fig3_accuracy_cost_pareto.png")


def fig4_conditioning(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """κ vs n_dof, 정규화별. Cholesky 실패점은 ✗."""
    rows = _rows(data_dir, "p1_conditioning")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b in sorted({str(r["basis"]) for r in rows}):
        g = sorted((r for r in rows if str(r["basis"]) == b),
                   key=lambda r: int(r["n_dof"]))
        xy = [(int(r["n_dof"]), float(r["kappa_A_transformed"])) for r in g
              if r.get("kappa_A_transformed") not in (None, "")
              and math.isfinite(float(r["kappa_A_transformed"]))]
        if len(xy) >= 2:
            ax.semilogy(*zip(*xy), marker="o", ms=3.5, lw=1.2, label=b)
        bad = [(int(r["n_dof"]), float(r["kappa_M_raw"])) for r in g
               if not bool(r["cholesky_ok"])]
        if bad:
            ax.scatter(*zip(*bad), marker="x", s=48, color="k", zorder=5)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel(r"$\kappa_2$ of the transformed operator")
    ax.set_title("P1 conditioning (x marks Cholesky failure, at raw kappa(M))",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    return _save(fig, outdir, "fig4_conditioning.png")


def fig5_rank_collapse(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """(d) vs (e)의 유효 rank와 상관 공선성 — §5.3.4의 두 패널."""
    rows = _rows(data_dir, "p1_neural_collinearity")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    tol = ["corr_rank_1e12", "rank_1e12"]
    arms = sorted({str(r["arm"]) for r in rows})
    for arm in arms:
        g = [r for r in rows if str(r["arm"]) == arm and float(r["w_orth"]) == 0.0]
        if not g:
            continue
        nb = int(g[0]["n_basis"])
        vals = [sum(float(r[t]) for r in g) / len(g) for t in tol]
        a1.bar([f"{_base(arm)[:12]}\n{t.replace('_1e12','')}" for t in tol],
               vals, alpha=0.8)
        a1.axhline(nb, ls=":", lw=1, color="k")
    a1.set_ylabel("effective rank (mean over seeds)")
    a1.set_title("nominal (dotted) vs numerical rank", fontsize=9)
    a1.tick_params(axis="x", labelsize=6)
    for arm in arms:
        g = [r for r in rows if str(r["arm"]) == arm and float(r["w_orth"]) == 0.0]
        if g:
            a2.hist([float(r["offdiag_ratio"]) for r in g], bins=12, alpha=0.6,
                    label=_base(arm)[:24])
    a2.set_xlabel(r"correlation off-diagonal ratio $\Sigma\rho^2/(M^2-M)$")
    a2.set_ylabel("seeds")
    a2.set_title("1 = every pair collinear", fontsize=9)
    a2.legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, outdir, "fig5_rank_collapse.png")


def fig6_spacetime_negative(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """(g) 잔차와 고유값 오차의 역상관 — 이 논문에서 가장 선명한 귀속 그림."""
    rows = _rows(data_dir, "p1_spacetime")
    g = {}
    for r in rows:
        g.setdefault(float(r["n_periods"]), []).append(r)
    T = sorted(g)
    med = lambda rs, k: sorted(float(r[k]) for r in rs)[len(rs) // 2]   # noqa: E731
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(T, [med(g[t], "loss_final") for t in T], "o-", color="#0d47a1",
              label="final PDE residual")
    ax.loglog(T, [med(g[t], "e_lam") for t in T], "s-", color="#b71c1c",
              label=r"eigenvalue error $e_\lambda$")
    ax.loglog(T, [float(g[t][0]["rel_lambda_bin"]) for t in T], ":", color="grey",
              label="FFT bin width (not a floor)")
    ax.set_xlabel("periods of mode 1 inside the time window")
    ax.set_ylabel("value")
    ax.set_title("Space-time PINN: residual falls while the target error rises",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    return _save(fig, outdir, "fig6_spacetime_negative.png")


def fig1_benchmark_problems(data_dir: str = DATA_DIR, outdir: str = FIG_DIR) -> str:
    """**미리 렌더된 개요도**를 경로로만 돌려준다 — 여기서 그리지 않는다.

    이 그림은 지배방정식에서 모드형을 풀어 그리므로 계산이 들어간다. `render/`는
    "계산하지 않는다"가 규약이므로 생성은 `docs/_generated/make_fig1_benchmark_problems.py`
    가 맡고, 이 함수는 그 산출물의 존재를 확인해 번호·캡션·출처 부여만 파이프라인에
    맡긴다. 파일이 없으면 조용히 넘어가지 않고 실패한다.
    """
    name = "fig1_benchmark_problems.png"
    p = os.path.join(outdir, name)
    if os.path.exists(p):
        return p
    # outdir가 기본값이 아닐 때(테스트의 임시 디렉터리 등)는 정본에서 복사한다.
    src = os.path.join(FIG_DIR, name)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"{src} 가 없다 — 먼저 `python docs/_generated/"
            f"make_fig1_benchmark_problems.py`를 돌려라")
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(src, p)
    return p


ALL = {"fig1_benchmark_problems": fig1_benchmark_problems,
       "fig1_mode_reliability": fig1_mode_reliability,
       "fig2_error_vs_dof": fig2_error_vs_dof,
       "fig3_accuracy_cost_pareto": fig3_accuracy_cost_pareto,
       "fig4_conditioning": fig4_conditioning,
       "fig5_rank_collapse": fig5_rank_collapse,
       "fig6_spacetime_negative": fig6_spacetime_negative}
