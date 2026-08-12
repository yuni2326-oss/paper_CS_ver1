"""논문 §5의 표를 CSV에서 조립한다. **여기서 새로 계산하는 숫자는 없다.**

각 함수는 마크다운 표 문자열 하나를 낸다. 출처 CSV와 git sha를 캡션에 남긴다 — 표만 보고도
어느 실행에서 나왔는지 추적할 수 있어야 한다(§3.7).

집계가 필요하면 **드라이버에서** 한다. 렌더러가 집계하면 논문의 숫자가 어느 코드에서
나왔는지 추적선이 끊긴다.
"""
from __future__ import annotations

import json
import math
import os

from ..drivers import manifest
from . import md

DATA_DIR = os.path.join("docs", "_generated", "data", "paper2")

ARM_LABEL = {
    "a_penalty": "(a) penalty deflation",
    "a_prime_projection_diagonal": "(a′) diagonal-approx. projection (pilot)",
    "b_projection_exact": "(b) exact M-orthogonal projection",
    "c_curriculum": "(c) coarse-Ritz curriculum",
    "d_neural_basis_galerkin": "(d) separate-net neural basis + Galerkin",
    "e_simultaneous_subspace": "(e) shared-trunk subspace trace",
    "f_eig_pielm": "(f) Eig-PIELM (random features, no backprop)",
    "g_spacetime_pinn": "(g) space–time PINN (time domain)",
}
ARM_ORDER = list(ARM_LABEL)

# P1·P2·P3의 참조해는 초월방정식의 근을 mpmath dps=50으로 찾은 것이라 불확도가 근찾기
# 허용오차이고, fp64로 보고하는 순간 **표현 자체가 하한**이 된다. 공란으로 두면 "재지
# 않았다"로 읽히므로 그 값을 명시한다.
HIGHPREC_UNC = "< 1e-15 (fp64 representation)"

FAMILY_LABEL = {"classical": "classical basis + `eigh`",
                "neural_randfeat": "neural, random features (no training)",
                "neural_gradient": "neural, gradient-trained"}


def sha(data_dir: str = DATA_DIR, name: str = "manifest_p1_neural.json") -> str:
    with open(os.path.join(data_dir, name), encoding="utf-8") as f:
        return json.load(f).get("git_sha", "unknown")


def _base(arm) -> str:
    return str(arm).split("(")[0]


def _rows(data_dir: str, name: str) -> list:
    return manifest.read_csv(os.path.join(data_dir, f"{name}.csv"))


def _frac(r, key: str) -> str:
    return f"{r[key]}/{r['n']}"


def _ci(r) -> str:
    return f"[{md.fmt(r['wilson_lo'])}, {md.fmt(r['wilson_hi'])}]"


# ---------------------------------------------------------------- 표 1


def table1_attribution(data_dir: str = DATA_DIR, n_table: int = 1) -> str:
    """arm × 목표모드의 4분류 카운트 — 논문의 귀속 표.

    v3 초안의 껍데기는 `arm × 초기화 × mode 3`이었으나, 초기화 사다리는 (a′)에서 I5
    (해석해 oracle)까지 전부 0.00이라 변별력이 없었다. 변별이 실제로 일어나는 축은
    **모드 번호**이므로 격자를 arm × mode로 바꿨다. 초기화는 표 4에서 예산 축과 함께 다룬다.
    """
    rows = _rows(data_dir, "p1_neural_mode_sweep")
    by = {}
    for r in rows:
        by.setdefault((_base(r["arm"]), int(r["mode"])), r)
    out = []
    for a in ARM_ORDER:
        for m in sorted({int(r["mode"]) for r in rows}):
            r = by.get((a, m))
            if r is None:
                continue
            out.append([ARM_LABEL[a], m, _frac(r, "correct"),
                        _frac(r, "lower_mode_basin"), _frac(r, "spurious"),
                        _frac(r, "non_converged"), r["p_correct"], _ci(r),
                        r.get("p_accurate")])
    body = md.table(["arm", "mode", "correct", "lower-mode basin", "spurious",
                     "non-converged", "p_correct", "95 % Wilson", "p_accurate"],
                    out, align="llrrrrrcr")
    cap = md.caption(
        n_table,
        "Outcome attribution on P1, 50 seeds per cell, I0 initialization, fixed "
        "Gauss–Legendre quadrature, 4000 iterations. Categories and thresholds "
        "(MAC ≥ 0.9, |Δλ|/λ ≤ 0.05) were pre-specified before any run and were not "
        "adjusted after seeing data. `p_accurate` — a post-hoc diagnostic, not part of "
        "the pre-specification — counts seeds whose recovered pair meets the same "
        "accuracy thresholds while skipping the convergence gate.",
        "p1_neural_mode_sweep.csv", sha(data_dir))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 표 2–5


def table2_penalty(data_dir: str = DATA_DIR, n_table: int = 2) -> str:
    rows = _rows(data_dir, "p1_neural_penalty_sweep")
    # **non-converged 열을 반드시 싣는다.** w = 1e5·모드 2는 correct 7, lower 0,
    # spurious 0이므로 나머지 43/50이 어디로 갔는지 표에서 사라진다 — 그 43이
    # 이 결과의 핵심(인증 실패)이므로 숨은 잔여가 있어서는 안 된다.
    out = [[r["penalty_weight"], r["mode"], _frac(r, "correct"),
            _frac(r, "lower_mode_basin"), _frac(r, "spurious"),
            _frac(r, "non_converged"),
            r["p_correct"], _ci(r), r.get("p_accurate")] for r in rows]
    body = md.table(["penalty weight w", "mode", "correct", "lower-mode basin",
                     "spurious", "non-converged", "p_correct", "95 % Wilson",
                     "p_accurate"],
                    out, align="rrrrrrrcr")
    cap = md.caption(
        n_table,
        "Penalty deflation on P1, 50 seeds per cell, six decades of weight. A one-line "
        "argument predicts the thresholds. If the stored modes were exact and "
        "mass-normalized, a trial function sitting on stored mode j would see an objective of "
        "λ_j + w, against λ_m for the target, so excluding mode 1 requires w > λ_m − λ₁. "
        "That bound is 473 for mode 2 and 3794 for mode 3, and the sweep turns those modes on "
        "at w = 1e3 and w = 1e4 — in each case the first decade above the bound. "
        "**Penalty deflation therefore does recover mode 3, given a large enough weight** — "
        "an earlier draft of this benchmark stopped at w = 1e3 and concluded it never did. "
        "What the mechanism costs is not capability but the need to know that weight: it is "
        "mode-dependent, it is set by a gap that is not available before the modes are "
        "computed, and the window closes from above — at w = 1e5 mode 2 falls to 0.14, with "
        "43 of 50 seeds failing the pre-specified convergence criterion while `p_accurate` "
        "stays at 1.00, so the recovered mode is still right and only the convergence gate "
        "fails. The contrast is arm (a′), which holds mode 2 "
        "at 1.00 with no weight to choose at all.",
        "p1_neural_penalty_sweep.csv", sha(data_dir))
    return body + "\n" + cap + "\n"


def table3_quadrature(data_dir: str = DATA_DIR, n_table: int = 3) -> str:
    rows = _rows(data_dir, "p1_neural_quadrature")
    audit = _rows(data_dir, "p1_neural_audit")
    noise = [r for r in audit if r.get("study") == "snapshot_noise_floor"]
    # (a′)의 MC 자체 대조군은 감사 후속에서 나왔다 — 같은 표에 실어야 증거가 보인다.
    rows = rows + [r for r in audit if r.get("study") == "a_prime_mc_control"]
    out = [[ARM_LABEL.get(_base(r["arm"]), r["arm"]), r["nodes"], r["mode"],
            r["p_correct"], r.get("p_accurate"), _frac(r, "non_converged"), _ci(r)]
           for r in rows]
    body = md.table(["arm", "quadrature", "mode", "p_correct", "p_accurate",
                     "non-converged", "95 % Wilson"], out, align="llrrrrc")
    extra = ""
    if noise:
        nrows = [[r["nodes"], r["rule_rel_change_median"], r["rule_rel_change_p95"],
                  r["prereg_tol"], r["ratio_median_to_tol"]] for r in noise]
        extra = ("\n" + md.table(
            ["quadrature", "tail relative change (median)", "(p95)",
             "pre-specified tolerance", "ratio"], nrows, align="lrrrr"))
    cap = md.caption(
        n_table,
        "Monte-Carlo collocation yields p_accurate = 1.00 with p_correct = 0.00: the "
        "recovered pairs meet the accuracy thresholds but the pre-specified "
        "convergence rule cannot certify them. The lower panel measures why — over the "
        "same tail window the rule uses, the stochastic objective's relative change is "
        "8.6e-2, about 86× the 1e-3 tolerance, while deterministic Gauss sits at 2.3e-6 "
        "(0.002×). The rule presumes a deterministic objective; under it the MC arm is "
        "inconclusive rather than wrong (§6). The (a′) rows are the same-arm control for "
        "the quadrature question: only the quadrature changed, and `p_accurate` separates "
        "the two effects cleanly — modes 1 and 2 stay accurate and are merely "
        "uncertifiable, while mode 3 is genuinely wrong. The mode-3 failure therefore "
        "survives a change of quadrature and cannot be attributed to it.",
        ["p1_neural_quadrature.csv", "p1_neural_audit.csv"], sha(data_dir))
    return body + extra + "\n" + cap + "\n"


def table4_warm_cold(data_dir: str = DATA_DIR, n_table: int = 4) -> str:
    """초기화 축 — 예산과 함께 보아야 의미가 있고, 사전적합 비용을 청구해야 순위가 맞다."""
    rows = [r for r in _rows(data_dir, "p1_warm_cold_summary")
            if _base(r["arm"]) == "b_projection_exact"]
    out = [[r["budget_iters"], r["ladder"], r["start"], r["p_correct"],
            r["train_seconds_per_attempt"], r["init_seconds_per_attempt"],
            r["E_T_success"], r["speedup_vs_cold"],
            r["E_T_success_if_init_free"]] for r in rows]
    body = md.table(["budget (iters)", "level", "start", "p_correct",
                     "train [s/attempt]", "pre-fit [s/attempt]",
                     "E[T_success] [s]", "vs cold",
                     "E[T] if pre-fit were free [s]"],
                    out, align="rllrrrrrr")
    cap = md.caption(
        n_table,
        "Initialization on arm (b), mode 3, 50 seeds. At the 4000-iteration budget every "
        "level reaches 1.00, so that grid has no headroom to detect an initialization "
        "effect; the axis only separates once the budget is tight enough for cold starts "
        "to fail. Two readings follow. First, what helps is not knowledge of the answer: "
        "a BC-satisfying random polynomial (I1) lifts 0.58 → 1.00 while knowing the "
        "eigenvalue window (I2) changes nothing. Second, charging the pre-fit reverses "
        "the ranking — the analytic oracle (I5) reaches 1.00 but costs about a third more "
        "per success than a cold start; the last column shows what the table would say if "
        "the pre-fit were not charged. All times are per attempt, on the same basis as the cost "
        "table: the "
        "single-seed cost measured directly at the 4000-iteration budget and carried to "
        "the other budgets as a ratio, not converted by a batching factor. The pre-fit "
        "uses max(iters//2, 200) iterations, so it is tied to the main budget by design "
        "and is not a fixed one-off cost.",
        "p1_warm_cold_summary.csv", sha(data_dir, "manifest_p1_warm_cold_summary.json"))
    return body + "\n" + cap + "\n"


def table5_longrun(data_dir: str = DATA_DIR, n_table: int = 5) -> str:
    rows = _rows(data_dir, "p1_neural_longrun")
    out = [[ARM_LABEL.get(_base(r["arm"]), r["arm"]), r["mode"],
            r.get("p_correct_base"), r["p_correct"], _ci(r), r["iters"],
            r.get("budget_resolved")] for r in rows]
    body = md.table(["arm", "mode", "p_correct @ 4000", "p_correct @ 20000",
                     "95 % Wilson", "iters", "budget-resolved"],
                    out, align="lrrrcrl")
    cap = md.caption(
        n_table,
        "Cells below p_correct = 1 in the base grid, re-run at five times the iteration "
        "budget. A cell is budget-resolved when the extra budget lifts it; a cell that "
        "does not move is limited by something the budget cannot buy.",
        "p1_neural_longrun.csv", sha(data_dir, "manifest_p1_neural_longrun.json"))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 표 6–7


def _construct_solve(data_dir: str) -> dict:
    """(기저, dof) → (구성초, 풀이초). 고전 계열만 있다."""
    try:
        rows = _rows(data_dir, "p1_basis_study")
    except FileNotFoundError:
        return {}
    return {(str(r["basis"]), str(r["n_dof"])):
            (float(r["seconds_construct"]), float(r["seconds_solve"])) for r in rows}


def table6_cost(data_dir: str = DATA_DIR, n_table: int = 6) -> str:
    """정확도 ε를 가장 싸게 달성하는 경로 — 세 계열을 같은 축에서."""
    rows = _rows(data_dir, "p1_cheapest_to_epsilon")
    out = []
    for r in rows:
        reached = bool(r["reached"])
        cs = _construct_solve(data_dir).get(
            (str(r["solver"]), str(r["size"]))) if reached else None
        # size는 계열마다 뜻이 다르므로 단위를 칸에 붙여 쓴다. 반복예산은 별개 열이다
        # — 이전에는 두 값이 한 열에 섞여 PIELM 행에 4000(=반복예산)이 찍혔다.
        size = (f"{md.fmt(r['size'])} {r.get('size_label', '')}".strip()
                if reached else None)
        it = r.get("iterations")
        out.append([r["mode"], r["eps"], FAMILY_LABEL[r["family"]],
                    r["solver"] if reached else "not reached", size,
                    (int(float(it)) if reached and it not in (None, "") else None),
                    r["e_lam"] if reached else None,
                    r["p_correct"] if reached else None,
                    (cs[0] * 1e3) if cs else None,
                    (cs[1] * 1e3) if cs else None,
                    (r["seconds_per_attempt"] * 1e3) if reached else None,
                    r["E_T_success"] if reached else None])
    body = md.table(["mode", "target ε", "family", "cheapest solver", "trial size",
                     "iterations", "e_λ", "p_correct", "construct [ms]", "solve [ms]",
                     "per attempt [ms]", "E[T_success] [s]"],
                    out, align="rrllrrrrrrrr")
    cap = md.caption(
        n_table,
        "Cheapest route to a target accuracy on P1. Families are split three ways "
        "because lumping the random-feature arm with the gradient-trained ones hides "
        "the latter entirely — Eig-PIELM is three orders cheaper and would take the "
        "single 'neural' slot. Trial size and iteration budget are separate columns "
        "because they are separate things: a classical size is a number of degrees of "
        "freedom, Eig-PIELM's is a number of random features, and neither iterates at all, "
        "while the gradient arms have a fixed architecture and a 4000-iteration budget. "
        "Two accounting statements, which apply to different quantities: **accuracy "
        "statistics use the 50-seed ensembles**, and **timing uses a single seed, measured "
        "directly as the median of three runs.** The two are separated because the "
        "ensemble-to-per-attempt conversion factor measured 2.53, 3.80 and 7.82 in three "
        "sessions on this shared GPU and is therefore not usable in a result. Classical "
        "solves and Eig-PIELM are single deterministic numpy/CPU solves. The "
        "CPU/GPU hardware difference is not corrected for and is reported as a threat "
        "in §6. Rows marked 'not reached' are kept rather than dropped.",
        ["p1_cheapest_to_epsilon.csv", "p1_classical_vs_neural.csv"],
        sha(data_dir, "manifest_p1_compare.json"))
    return body + "\n" + cap + "\n"


def table7_references(data_dir: str = DATA_DIR, n_table: int = 7) -> str:
    out = []
    for r in _rows(data_dir, "p1_reference"):
        u = r.get("uncertainty_rel")
        out.append(["P1", f"mode {r['mode']}", "(βL)⁴", r["Lambda"], "",
                    r["reference"], u if u not in (None, "") else HIGHPREC_UNC])
    for r in _rows(data_dir, "p2_reference"):
        out.append(["P2", f"m={r['m']}, n={r['radial_order']}", "(kb)⁴",
                    r["Lambda_k4"], "degenerate" if r["degenerate"] else "",
                    r["reference"], HIGHPREC_UNC])
    for r in _rows(data_dir, "p3_reference"):
        out.append(["P3", f"k̂={md.fmt(r['k_hat'])}, mode {r['mode']}", "(βL)⁴",
                    r["Lambda"], "", r["reference"], HIGHPREC_UNC])
    for r in _rows(data_dir, "p4_reference"):
        out.append(["P4", f"mode {r['mode']}", "Λ_el=ω²ρL²/E",
                    r["lambda_extrapolated"],
                    f"rate {md.fmt(r['convergence_rate'])}",
                    r["reference"], r["uncertainty_rel"]])
    body = md.table(["problem", "label", "quantity", "value", "note",
                     "reference method", "relative uncertainty"],
                    out, align="llllllr")
    cap = md.caption(
        n_table,
        "Reference eigenvalues, all dimensionless. References are analytic (P1 "
        "Euler–Bernoulli, P2 exact Bessel for the annular Kirchhoff plate, P3 "
        "transfer-matrix determinant at mpmath dps = 50) or Richardson-extrapolated Q2 "
        "FEM (P4). The P4 uncertainty is the extrapolation residual and sets the floor "
        "below which an error cannot be attributed to the solver. Degenerate P2 pairs "
        "are flagged: they must be scored by subspace MAC, not by individual MAC (§5.2). "
        "Displayed values are rounded; the stored references carry full precision, and the "
        "stated uncertainties refer to the stored values, not the rounded display.",
        ["p1_reference.csv", "p2_reference.csv", "p3_reference.csv",
         "p4_reference.csv"], sha(data_dir, "manifest_p1.json"))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 표 8–10


def table8_conditioning(data_dir: str = DATA_DIR, n_table: int = 8) -> str:
    rows = _rows(data_dir, "p1_conditioning")
    out = [[r["basis"], r["n_dof"], r["kappa_M_raw"], r["kappa_K_raw"],
            r["kappa_M_massnorm"], r["kappa_K_equilibrated"],
            r.get("kappa_A_transformed"),
            "ok" if r["cholesky_ok"] else "**fail**",
            r.get("backward_error")] for r in rows]
    body = md.table(["basis", "n_dof", "κ(M) raw", "κ(K) raw", "κ(M) mass-norm",
                     "κ(K) equilibrated", "κ transformed operator",
                     "Cholesky", "backward error"], out, align="lrrrrrrlr")
    cap = md.caption(
        n_table,
        "Conditioning of the discrete pencil on P1 under four normalizations. The raw "
        "condition numbers are a property of the coordinates, not of the trial space: "
        "three coordinate systems spanning the same space agree only after the "
        "transformed-operator normalization. Factorization failures are shown, not "
        "hidden — raw monomials lose positive-definiteness from N = 12, the smallest size "
        "tested, and high precision does not recover it because the information is already "
        "lost at assembly.",
        "p1_conditioning.csv", sha(data_dir, "manifest_p1.json"))
    return body + "\n" + cap + "\n"


def table9_effective_rank(data_dir: str = DATA_DIR, n_table: int = 9) -> str:
    """세 계열의 명목 / 수치 rank / 사용 가능 차원 — §5.3의 통합 논지."""
    out = []
    cond = _rows(data_dir, "p1_conditioning")
    for r in cond:
        if r["basis"] == "monomial_raw" and int(r["n_dof"]) in (12, 16, 20, 24):
            out.append(["classical monomial (raw)", r["n_dof"],
                        "n/a", "n/a",
                        "ok" if r["cholesky_ok"] else "**fail**",
                        "deterministic coordinate choice; orthonormalizing the same "
                        "span restores factorization"])
    pie = [r for r in _rows(data_dir, "p1_neural_mode_sweep")
           if _base(r["arm"]) == "f_eig_pielm" and int(r["mode"]) == 1]
    for r in pie:
        nf = str(r["arm"]).split("nf=")[-1].rstrip(")")
        out.append(["random features (Eig-PIELM)", nf,
                    r.get("rank_tol_1e-16"), r.get("rank_tol_1e-10"), "ok",
                    "redundancy from the random draw; independent of training"])
    col = _rows(data_dir, "p1_neural_collinearity")
    for arm in sorted({str(r["arm"]) for r in col}):
        g = [r for r in col if str(r["arm"]) == arm and float(r["w_orth"]) == 0.0]
        if not g:
            continue
        nb = g[0]["n_basis"]
        med = lambda k: sorted(float(r[k]) for r in g)[len(g) // 2]      # noqa: E731
        # (d)와 (e)를 한 문구로 묶으면 표가 본문과 어긋난다. (e)는 수치 rank가
        # 6/6으로 온전히 유지되므로 "붕괴"가 아니다 — 붕괴한 것은 (d)뿐이고
        # (e)의 문제는 rank를 잃지 않은 채로 쌍별 상관이 1에 가깝다는 것이다.
        r12 = med("rank_1e12")
        mech = (f"high pairwise collinearity, full numerical rank "
                f"(Σρ²/max = {md.fmt(med('offdiag_ratio'))})"
                if r12 >= float(nb) else
                f"optimization-induced collapse to rank {md.fmt(r12)}; pairwise "
                f"|ρ| ≈ 1 (Σρ²/max = {md.fmt(med('offdiag_ratio'))})")
        out.append([f"trained neural basis — {_base(arm)}", nb,
                    md.fmt(med("corr_rank_1e12")), md.fmt(r12), "ok", mech])
    body = md.table(["family", "nominal dimension", "numerical rank @1e-16",
                     "numerical rank @1e-10/1e-12", "factorization", "mechanism"],
                    out, align="lrrrll")
    cap = md.caption(
        n_table,
        "Three notions of dimension, and three distinct mechanisms by which a nominal "
        "dimension can overstate the usable one: raw monomials by a deterministic choice of "
        "coordinates, random features by the draw, and trained neural bases by the "
        "optimization itself. The mechanism is present in all three representation families "
        "but does not cost every arm its rank. The two trained arms in particular are not "
        "the same failure: the separate-net basis (d) "
        "loses numerical rank outright, whereas the shared-trunk subspace (e) keeps its full "
        "rank of 6 and is redundant only in the weaker sense of high pairwise correlation, "
        "which is what its span-invariant objective permits (§5.3.2). Numerical rank is "
        "threshold-dependent, so it is reported at more than "
        "one tolerance — a single integer would hide the choice. This is a shared "
        "diagnostic requirement, not one mechanism under three names (§5.3).",
        ["p1_conditioning.csv", "p1_neural_mode_sweep.csv",
         "p1_neural_collinearity.csv"],
        sha(data_dir, "manifest_p1_neural_collinearity.json"))
    return body + "\n" + cap + "\n"


def table10_spacetime(data_dir: str = DATA_DIR, n_table: int = 10) -> str:
    """(g) 시공간 PINN — 잔차와 고유값 오차의 역상관."""
    rows = _rows(data_dir, "p1_spacetime")
    g = {}
    for r in rows:
        g.setdefault(float(r["n_periods"]), []).append(r)
    out = []
    for T in sorted(g):
        rs = g[T]
        med = lambda k: sorted(float(r[k]) for r in rs)[len(rs) // 2]    # noqa: E731
        out.append([T, rs[0]["T_nondim"], med("omega_hat"), rs[0]["omega_ref"],
                    med("e_lam"), rs[0]["rel_lambda_bin"], med("e_lam_over_bin"),
                    med("loss_final"), med("seconds")])
    body = md.table(["periods in window", "T (nondim.)", "ω̂ (median)", "ω reference",
                     "e_λ (median)", "first-order relative Λ bin, 2/periods",
                     "e_λ / bin",
                     "final PDE residual", "seconds"], out, align="rrrrrrrrr")
    cap = md.caption(
        n_table,
        "Space–time PINN on P1, 4000 iterations, 3 seeds per window. From 4 to 32 periods the "
        "PDE residual falls by 8.2× between the endpoints while the eigenvalue error rises "
        "monotonically from below the resolution limit to 7 bins. The residual is not "
        "monotone — it rises again in the last interval — so its figure is an endpoint ratio "
        "and not a trend at every step. Minimizing the residual does not minimize the "
        "quantity of interest, because the eigenvalue is read off afterwards rather than "
        "optimized. The recovered frequency drifts monotonically downward by 25 % over the "
        "same sweep. **The bin column sets what this estimator can resolve, and the "
        "4-period row must be read with it.** An FFT over a window of T = periods·2π/ω has "
        "Δω/ω = 1/periods; since Λ = ω², a one-bin change in ω gives an exact relative Λ "
        "change of (1 + 1/periods)² − 1 = 2/periods + 1/periods², of which the column keeps "
        "the leading term (0.500 at 4 periods, against 0.5625 exact; the angular spacing "
        "Δω = 2π/T is 0.879 there). The windows are integer numbers of reference periods and "
        "the record length is exactly T, so the reference frequency falls on bin `periods` "
        "exactly — verified by passing the analytic mode through the same estimator, which "
        "then returns zero error. A peak in that bin therefore means the error is **below "
        "half a bin**, not that it is zero: the 4-period entry of order 1e-16 is an upper "
        "bound of about 0.25, not a measurement of exactness. From 16 periods e_λ/bin exceeds "
        "one, so those errors are resolved and the growth is real. That the windows were "
        "built from integer periods of the reference frequency is prior information, and it "
        "favours this negative control rather than penalizing it.",
        "p1_spacetime.csv", sha(data_dir, "manifest_p1_spacetime.json"))
    return body + "\n" + cap + "\n"


def table_degeneracy(data_dir: str = DATA_DIR, n_table: int = 8) -> str:
    """축퇴쌍을 개별 MAC으로 채점하면 안 된다는 근거 — 본문 산문이 아니라 표로 낸다."""
    rows = _rows(data_dir, "p2_degeneracy")
    out = [[r["m"], r["individual_mac_rotated"], r["subspace_mac_rotated"],
            r["max_principal_angle_rad"], r["note"]] for r in rows]
    body = md.table(["nodal diameters m", "individual MAC (rotated pair)",
                     "subspace MAC", "max principal angle [rad]", "note"],
                    out, align="rrrrl")
    cap = md.caption(
        n_table,
        "Degenerate nodal-diameter pairs on P2. A rotation within the eigenspace leaves "
        "the space unchanged but destroys individual MAC, which falls as low as 0.095, "
        "while the subspace MAC stays at 1.0000 and the largest principal angle is 3e-8. "
        "Scoring degenerate modes one at a time therefore reports failures that are not "
        "failures. This is a scoring result about an intact structure; it is unrelated "
        "to the splitting of a degenerate pair by a defect.",
        "p2_degeneracy.csv", sha(data_dir, "manifest_p2.json"))
    return body + "\n" + cap + "\n"


def table11_objective_weighting(data_dir: str = DATA_DIR, n_table: int = 11) -> str:
    """(e)의 모드1 악화 귀속 — 목적함수의 가중만 바꾼 대조."""
    rows = _rows(data_dir, "p1_e_objective_ablation")
    tr = {int(r["mode"]): r for r in rows if r["objective"] == "trace"}
    lg = {int(r["mode"]): r for r in rows if r["objective"] == "logtrace"}
    out = []
    for m in sorted(tr):
        a, b = tr[m], lg.get(m)
        if b is None:
            continue
        out.append([m, a["p_correct"], a["e_lam_median"], a["h2_over_l2"],
                    b["p_correct"], b["e_lam_median"], b["h2_over_l2"],
                    float(a["e_lam_median"]) / float(b["e_lam_median"])])
    body = md.table(["mode", "p_correct (Σλ)", "e_λ (Σλ)", "H²/L² (Σλ)",
                     "p_correct (Σlog λ)", "e_λ (Σlog λ)", "H²/L² (Σlog λ)",
                     "e_λ improvement"], out, align="rrrrrrrr")
    cap = md.caption(
        n_table,
        "Arm (e) with the objective weighting changed and nothing else — same "
        "architecture, budget, quadrature and seeds; 651 s versus 653 s, since the "
        "log-determinant form needs no eigendecomposition. The absolute trace Σλ_k "
        "weights every mode equally in absolute terms, so on this problem sacrificing "
        "7 % of λ₁ (0.87 in the objective) buys a 0.1 % gain on λ₆ (89) at 100:1; "
        "Σ log λ_k makes relative changes equivalent instead. Three predictions of that "
        "account are met: the inverted accuracy profile disappears, the improvement "
        "decreases monotonically with mode order — 2122×, 516×, 52.6×, 33.3×, 15.7× and "
        "11.3× for modes 1 to 6, i.e. largest exactly where the under-weighting was "
        "worst — and the cost is unchanged. The H²/L² ratio, however, still falls monotonically with "
        "mode number under both objectives — the low modes' error stays concentrated in "
        "curvature, which is a property of the shared representation and not of the "
        "weighting. This is an attribution experiment; the log-determinant objective is "
        "standard in subspace methods and is not proposed here as a contribution.",
        "p1_e_objective_ablation.csv",
        sha(data_dir, "manifest_p1_e_objective.json"))
    return body + "\n" + cap + "\n"


def table_orthogonality(data_dir: str = DATA_DIR, n_table: int = 5) -> str:
    """저장된 원시 출력의 직교성 — (a′)와 (b)가 갈라지는 지점의 정량.

    **두 지표를 구분해 싣는다.** ‖G⁻¹ − D⁻¹‖/‖D⁻¹‖는 계수사상의 거리이지 연산자의
    거리가 아니다. 논문의 규약은 Φ의 열이 기저함수이고 G = ΦᵀWΦ이므로 정확 사영은
    Π_G = ΦG⁻¹ΦᵀW이고 대각 규칙은 A_D = ΦD⁻¹ΦᵀW다. 실제 차이는 Π_G − A_D =
    Φ(G⁻¹ − D⁻¹)ΦᵀW이고, C = D^{-1/2}GD^{-1/2}로 두면 ‖Π_G − A_D‖_W = ‖I − C‖₂로
    정확히 계산된다 — 상한이 아니라 실측값이다.

    **A_D는 사영이 아니다.** A_D² = A_D는 D⁻¹G가 멱등일 때, 즉 C = I일 때만 성립한다.
    저장 모드가 서로 직교하지 않으면 대각 공식은 부정확한 사영이 아니라 **사영이 아닌
    연산자**다 — 이 논문의 핵심 메시지를 더 정확히 만드는 사실이므로 표기를 갈라 쓴다.
    arm 이름 "diagonal-approximation projection"은 실험명으로 유지한다.
    """
    rows = _rows(data_dir, "p1_orthogonality")
    main = [r for r in rows if r.get("gram_dev_fro_median") not in (None, "")]
    leak = {(_base(r["arm"])): r for r in rows
            if r.get("mode3_leak_rel_median") not in (None, "")}
    out = []
    for r in main:
        b = _base(r["arm"])
        # 모드-3 누출은 그 풀이가 쓰는 기저(k = 2)의 행에만 놓는다
        lk = (leak.get(b, {}).get("mode3_leak_rel_median")
              if int(float(r["n_prev_modes"])) == 2 else None)
        out.append([ARM_LABEL.get(b, r["arm"]), r["n_prev_modes"],
                    r["gram_dev_fro_median"], r["max_offdiag_median"],
                    r.get("lam_min_normalized_median"), r.get("kappa_gram_median"),
                    r.get("inv_gram_rel_diff_median"),
                    r.get("proj_op_diff_W_median"), lk])
    body = md.table(["arm", "stored modes k", "‖C − I‖_F", "max |off-diagonal|",
                     "λ_min(C)", "κ₂(G)", "inverse-Gram discrepancy",
                     "‖Π_G − A_D‖_W", "mode-3 residual leakage"],
                    out, align="lrrrrrrrr")
    meth = next((r.get("solve_method") for r in rows if r.get("solve_method")), "")
    # 정확 사영 arm에서 **실제로 풀이에 넘겨지는** 가장 깊은 그람 — 손으로 적으면
    # 재실행마다 어긋난다. k개가 저장된 그람은 모드 k+1을 풀 때 쓰이므로, 모드를 n개까지
    # 계산했으면 쓰이는 최대는 k = n−1이다(k = n 행은 마지막 저장 집합의 성질일 뿐이다).
    ex = [r for r in main if _base(r["arm"]) == "b_projection_exact"]
    kmax = max((int(float(r["n_prev_modes"])) for r in ex), default=0)
    deep = next((r for r in ex
                 if int(float(r["n_prev_modes"])) == max(kmax - 1, 2)), {}) if ex else {}
    kb = int(float(deep.get("n_prev_modes", 0)))
    kap_b = float(deep.get("kappa_gram_median", float("nan")))
    lmin_b = float(deep.get("lam_min_normalized_median", float("nan")))
    # fp64 유효자리 ≈ 16 − log10(κ₂)
    # 자릿수는 κ₂·u 기반 **최악 추정**이다 — 실제 손실은 보통 이보다 적다
    digits_b = (max(int(round(-math.log10(kap_b * 2.22e-16))), 0)
                if kap_b == kap_b and kap_b > 0 else 0)
    # 효과 rank가 모든 행에서 k와 같은지 확인해 그 사실만 서술한다 — 한 행의 값을
    # 대표로 쓰면 "stays full (2)"처럼 읽힌다.
    inv_max = max((float(r["inv_gram_rel_diff_median"]) for r in main
                   if r.get("inv_gram_rel_diff_median") not in (None, "")),
                  default=float("nan"))
    # 유효 rank는 arm마다 다르게 끝난다 — (a′)는 마지막 단계에서 처음 결손된다.
    # "모든 행에서 k와 같다"로 뭉치면 그 사실이 사라지고, 조건 대신 특이성으로 읽힌다.
    er = [r for r in main if r.get("eff_rank_1e12_median") not in (None, "")]
    ex_full = all(float(r["eff_rank_1e12_median"]) == float(r["n_prev_modes"])
                  for r in er if _base(r["arm"]) == "b_projection_exact")
    short = [(int(float(r["n_prev_modes"])), int(float(r["eff_rank_1e12_median"])))
             for r in er if _base(r["arm"]) != "b_projection_exact"
             and float(r["eff_rank_1e12_median"]) < float(r["n_prev_modes"])]
    cap = md.caption(
        n_table,
        "Orthogonality of the stored raw network outputs that each deflation stage is "
        "handed, on P1 with 50 seeds; all entries are medians over seeds. C is their "
        "mass-normalized Gram, `C = D^{-1/2}GD^{-1/2}` with `G = ΦᵀWΦ` the unnormalized "
        "Gram of the stored set and `D = diag(G)`. Two different quantities are "
        "reported and should not be conflated. The **inverse-Gram discrepancy** "
        "‖G⁻¹ − D⁻¹‖_F / ‖D⁻¹‖_F compares the two coefficient maps; it is not "
        "a bound on the error of the map. The **operator difference** is the exact "
        "norm of the two maps themselves. With Φ holding the stored modes "
        "as columns, the exact rule is the W-orthogonal projector `Π_G = ΦG⁻¹ΦᵀW` and the "
        "diagonal rule is the subtraction map `A_D = ΦD⁻¹ΦᵀW`, so "
        "`Π_G − A_D = Φ(G⁻¹ − D⁻¹)ΦᵀW`, and under the mass inner "
        "product that operator has norm ‖Π_G − A_D‖_W = ‖I − C‖₂ = max_i |1 − λ_i(C)| — a "
        "measured value, not a "
        "bound. **A_D is not a projection**: `A_D² = A_D` requires `D⁻¹G` to be idempotent, "
        "i.e. `C = I`. Once the stored set is not mutually orthogonal the diagonal formula "
        "is not an inexact projector but an operator that does not project at all, which is "
        "the sharper form of this paper\u2019s claim. The operator difference is O(1) while "
        f"the inverse-Gram discrepancy reaches {md.fmt(inv_max)}, which is why the "
        "two are kept apart. The last column is defined as "
        "‖D^{-1/2}ΦᵀWφ̃‖₂ / ‖φ̃‖_W with φ̃ the deflated mode-3 function, i.e. the "
        "root-sum-square over stored modes of cos∠(φ_k, φ̃) in the mass inner product; it "
        "is zero if and only if φ̃ is mass-orthogonal to every stored mode, and for a "
        "near-parallel stored pair its ceiling is √2 ≈ 1.414. Four things are visible. The stored outputs are strongly "
        "non-orthogonal in **both** arms — the largest off-diagonal is about 0.99 "
        "throughout — because deflation is applied when a mode is used, not to what is "
        "stored. The k = 2 rows agree to every digit across the two arms, and necessarily "
        "so: that basis was produced by the mode-1 solve, which deflates against nothing, "
        "and the mode-2 solve, whose Gram is 1 × 1, where diag(G) = G and the diagonal "
        "approximation is not an approximation; mode 3 is the first solve handed a Gram "
        "with an off-diagonal. Third, the projections do differ materially there — "
        "‖Π_G − A_D‖_W is comparable to ‖Π_G‖ = 1 — and the consequence is measured "
        "directly by the last column, the residual overlap of the deflated mode-3 function "
        "with the stored set: machine zero for the exact rule against order unity for the "
        "diagonal one, meaning the diagonally \"deflated\" function still lies largely in "
        "the span it was supposed to leave. Fourth, since the exact arm inverts this Gram, "
        "its conditioning is itself a result and is reported rather than assumed away. "
        f"κ₂(G) grows with k and λ_min(C) approaches zero: at k = {kb} — the stored set the "
        f"mode-{kb + 1} solve is handed, i.e. the deepest one behind the no-wall-through-mode-10 "
        f"claim — the exact arm reaches κ₂(G) = {md.fmt(kap_b)} and "
        f"λ_min(C) = {md.fmt(lmin_b)}. With u the fp64 unit roundoff, κ₂(G)·u is then of "
        f"order {md.fmt(kap_b * 2.22e-16, 1)}, i.e. roughly four to five decimal digits in "
        f"a worst-case conditioning estimate. That is thin but not "
        "exhausted, and it is a limit of the exact rule itself rather than an argument for "
        "the diagonal one, whose Gram at the same depth is worse by two further orders. "
        "Arm (a′) reaches an "
        "off-diagonal of exactly 1.0000, i.e. its stored set holds "
        "a duplicate — the collapse of mode 3 onto mode 1 feeding back into the next stage. "
        + (f"At a 1e-12 relative cutoff the exact arm\u2019s median effective rank equals k "
           f"in every row, so its Gram is ill-conditioned rather than numerically singular "
           f"there. " if ex_full else "")
        + (f"Arm (a\u2032) instead loses rank outright at k = {short[0][0]}, where the "
           f"median effective rank is {short[0][1]}. " if short else "")
        + (f"The k = {kmax} row is the set left after all {kmax} modes have been stored; in "
           f"a {kmax}-mode run it is handed to no solve and is reported only to show that "
           f"the trend continues. " if kmax else "")
        + f"The exact arm solves with {meth}.",
        "p1_orthogonality.csv", sha(data_dir, "manifest_p1_orthogonality.json"))
    return body + "\n" + cap + "\n"


def table_headline(data_dir: str = DATA_DIR, n_table: int = 11) -> str:
    """정본 비율 — 초록·본문·캡션·결론이 **모두 이 표를 가리킨다**.

    같은 비율을 다섯 곳에 손으로 적었다가 서로 어긋났다(5.34 / 5.4 / "5.40–5.50" / 5.42가
    한 문서에 공존, narrowing 1.6 대 1.7, 정확도 격차 two 대 three orders). 하나의 계산에서
    렌더링하면 그 종류의 불일치가 구조적으로 불가능해진다."""
    rows = _rows(data_dir, "p1_cost_headline")
    out = []
    for r in rows:
        out.append([r["quantity"], r.get("denominator_solver"),
                    r.get("numerator_solver"),
                    r.get("cost_ratio"), r["cost_orders"],
                    r.get("expected_success_orders"),
                    r.get("accuracy_orders"), r.get("bound")])
    body = md.table(["comparison", "cheaper side", "costlier side", "cost ratio",
                     "per attempt, orders", "expected success, orders",
                     "accuracy advantage of the cheaper side, orders", "bound"],
                    out, align="lllrrrrl")
    cap = md.caption(
        n_table,
        "Canonical cost and accuracy ratios. **The two problems are not on a common "
        "accuracy target.** The P1 rows are taken at the 1e-4 target on mode 1. The P4 "
        "rows cannot be: the neural arm there plateaus at e_λ ≈ 3e-3, so its comparison is "
        "against the cheapest classical mesh that matches or beats that accuracy, which is "
        "a weaker target. The P4 gap and the difference derived from it are therefore "
        "read at each problem\u2019s own achievable accuracy, and the two are compared as "
        "such rather than as a single controlled cross-problem contrast. Every ratio quoted "
        "anywhere in this paper is read from this "
        "table rather than restated, because an earlier draft carried the same number in "
        "five places and they drifted apart. Costs are per attempt: one numpy/CPU run "
        "counted from basis construction onward for the classical and random-feature "
        "families, and a directly measured single-seed run — median of three — for the "
        "gradient-trained arms on both P1 and P4, so the two problems are on the same "
        "footing. The last column states whether a row is a direct measurement or a "
        "bound: a row would be an upper bound if only an ensemble wall-clock were "
        "available for the neural side, since that overstates its per-attempt cost, and "
        "the difference row would then be a lower bound. The last row is the difference "
        "between the two problems — the observed P1-to-P4 gap difference, named as an "
        "observation and not as an effect of the corner, for the reasons given in §5.4. Both "
        "accounting bases are given as columns rather than one being described in prose, "
        "since expected time-to-success is the paper\u2019s declared cost metric: the "
        "per-attempt column divides wall-clock by attempt, the expected-success column "
        "divides it further by p_correct on each side. On P1 mode 1 the two coincide because "
        "p_correct = 1.00 for both sides; on P4 they separate because the neural arm\u2019s "
        "p_correct is 0.76, 0.68 and 0.66. The verbal claim that the observed gap is about two "
        "orders smaller on P4 "
        "holds on either basis. Rounded "
        "verbal characterizations such as \"about two orders\" appear in the text; exact "
        "values appear only here.",
        ["p1_cost_headline.csv", "p1_cheapest_to_epsilon.csv",
         "p4_classical_vs_neural.csv"],
        sha(data_dir, "manifest_cost_headline.json"))
    return body + "\n" + cap + "\n"


def table_p4(data_dir: str = DATA_DIR, n_table: int = 13) -> str:
    """P4 2차원 신경 arm의 사전등록 4분류 — P1과 같은 잣대."""
    rows = _rows(data_dir, "p4_neural")
    out = [[r["mode"], _frac(r, "correct"), _frac(r, "lower_mode_basin"),
            _frac(r, "spurious"), _frac(r, "non_converged"), r["p_correct"],
            _ci(r), r["p_accurate"], r["e_lam_median"], r["mac_median"],
            r["Lam_median"], r["Lam_ref"]] for r in rows]
    body = md.table(["mode", "correct", "lower-mode basin", "spurious",
                     "non-converged", "p_correct", "95 % Wilson", "p_accurate",
                     "e_λ (median)", "MAC (median)", "Λ_el (median)",
                     "Λ_el reference"],
                    out, align="rrrrrrcrrrrr")
    cap = md.caption(
        n_table,
        "Exact-projection arm on P4, in two dimensions, 50 seeds per mode, 4000 "
        "iterations, clamped condition imposed exactly by the distance-function "
        "construction of §3.3. Scored by the same pre-specified rules as P1, with MAC "
        "computed against the FEM reference evaluated at the neural arm's own quadrature "
        "points. Two things carry over and one does not. The deflation result carries "
        "over: there is not a single lower-mode-basin outcome, so an exact projection "
        "shows no mode-order wall in two dimensions either. The accuracy plateau does "
        "not: e_λ settles near 3e-3, one to two orders above the reference uncertainty, "
        "and the remaining failures comprise both non-convergence and spurious solutions — "
        "at mode 3 there are 9 spurious against 8 non-converged, so they cannot be described "
        "as convergence failures alone. What the exact projection removes on P4 is "
        "lower-mode attraction, and that it removes completely. The cost "
        "comparison is in §5.4.",
        "p4_neural.csv", sha(data_dir, "manifest_p4_neural.json"))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 부록


def appendix_a_nondimensional(n_table: int | None = None) -> str:
    """부록 A — 무차원 문제 정의. **절대 치수·물성·Hz는 넣지 않는다**(경계 정본 §1)."""
    from ..problems.p2_annulus import P2_GEOMETRY as G2
    from ..problems.p3_spring import P3_CONFIG as G3
    from ..problems.p4_lshape import P4_GEOMETRY as G4
    rows = [
        ["P1", "clamped–free uniform Euler–Bernoulli beam", "—",
         "Λ = (βL)⁴; independent of material and dimensions"],
        ["P2", "annular Kirchhoff plate, clamped inner / free outer",
         f"a/b = {md.fmt(G2['a'] / G2['b'])}, ν = {md.fmt(G2['nu'])}",
         "Λ = (kb)⁴ with b = 1"],
        ["P3", "beam with an internal rotational spring",
         f"x_c/L = {md.fmt(G3['xc_over_L'])}, "
         f"k̂ ∈ {{{', '.join(md.fmt(k) for k in G3['k_hats'])}}}",
         "Λ = (βL)⁴; k̂ = k_θL/(EI)"],
        ["P4", "L-shaped plane-stress domain",
         f"unit-length arms, ν = {md.fmt(G4['nu'])}",
         "Λ_el = ω²ρL²/E with L = E = ρ = 1 (Ω denotes the domain)"]]
    body = md.table(["problem", "domain", "dimensionless parameters", "eigenvalue"],
                    rows, align="llll")
    note = ("\nEvery test problem is stated dimensionlessly. Absolute dimensions, "
            "material properties and frequencies are deliberately absent: none of the "
            "benchmark's conclusions depend on them, and quoting them would tie the "
            "problems to one particular structure.\n")
    return body + note


# 공시값의 정본은 코드다. 그런데 렌더링은 torch가 없는 호스트에서 돌아야 하므로 여기에
# 사본을 두고, torch가 있는 환경의 테스트
# (`test_render_tables_gpu.py::test_disclosure_defaults_match_the_code`)가 사본과 코드의
# 일치를 강제한다. 사본이 낡으면 그 테스트가 실패한다.
def _spacetime_defaults() -> dict:
    """`solve_spacetime`의 기본 인자를 그대로 읽는다 — 부록이 코드와 어긋나지 않게."""
    import inspect
    try:
        from ..neural import spacetime
    except Exception:                      # torch 없는 환경(문서 렌더)에서는 사본을 쓴다
        return {"n_col": 4096, "n_ic": 256, "n_bc": 256, "n_probe": 512,
                "w_ic": 100.0, "w_bc": 10.0, "iters": 4000, "lr": 2e-3}
    sig = inspect.signature(spacetime.solve_spacetime)
    return {k: sig.parameters[k].default
            for k in ("n_col", "n_ic", "n_bc", "n_probe", "w_ic", "w_bc",
                      "iters", "lr")}


ST = _spacetime_defaults()

DISCLOSED = {"width": 64, "depth": 4, "activation": "tanh",
             "spacetime_params": 12737, "spacetime_width": 64, "spacetime_depth": 4,
             "p4_width": 64, "p4_depth": 4, "p4_params": 12802,
             "p4_n_q": 1200, "p4_n_per_block": 20}


def availability_record(data_dir: str = DATA_DIR) -> str:
    """Data and code availability 절 — **환경 기록에서 렌더링한다.**

    손으로 쓰면 어긋난다. 실제로 이 저장소의 호스트 venv는 작업 중에 numpy를 올렸고,
    그래서 산출물 절반은 numpy 2.1.0으로, 절반은 2.5.0으로 만들어졌다. 그 사실은 각
    manifest에 남아 있으므로 여기서 세어 그대로 적는다 — "라이브러리 버전 하나"로 뭉개면
    거짓이 된다.

    DOI와 최종 SHA는 제출 시점에 정해지므로 자리만 만들고 값은 비운다.
    """
    import glob
    import json

    def env(tag):
        p_ = os.path.join(data_dir, f"environment_{tag}.json")
        if not os.path.exists(p_):
            return {}
        with open(p_, encoding="utf-8") as f:
            return json.load(f)

    h, g = env("host"), env("gpu")

    # 산출물을 실제로 만든 버전 조합을 manifest에서 센다
    combos = {}
    for f_ in sorted(glob.glob(os.path.join(data_dir, "manifest*.json"))):
        with open(f_, encoding="utf-8") as fh:
            d = json.load(fh)
        key = (d.get("python"), d.get("numpy"), d.get("scipy"), d.get("mpmath"))
        combos[key] = combos.get(key, 0) + 1
    combo_txt = "; ".join(
        f"Python {k[0]}, NumPy {k[1]}, SciPy {k[2]}, mpmath {k[3]} ({n} artifacts)"
        for k, n in sorted(combos.items(), key=lambda kv: -kv[1]) if k[0])

    rows = [
        ["Repository DOI", "*minted at submission*"],
        ["Final git SHA", "*recorded at submission; every artifact also carries the "
                          "SHA of the commit that produced it, in its own manifest*"],
        ["Hardware", f"{h.get('cpu') or g.get('cpu', '—')}; "
                     f"{h.get('gpu') or g.get('gpu', '—')}"
                     + (f" (driver {h.get('nvidia_driver') or g.get('nvidia_driver')})"
                        if (h.get("nvidia_driver") or g.get("nvidia_driver")) else "")],
        ["Operating system", f"{h.get('os', '—')}, kernel {h.get('kernel', '—')}, "
                             f"{h.get('platform', '—').split()[-1]}"],
        ["Classical solves, references, rendering",
         f"CPU only. Python {h.get('python', '—')}, "
         f"NumPy {h.get('numpy', '—')}, SciPy {h.get('scipy', '—')}, "
         f"mpmath {h.get('mpmath', '—')}, Matplotlib {h.get('matplotlib', '—')}"],
        ["Neural arms",
         f"GPU. PyTorch {g.get('torch', '—')} on CUDA {g.get('torch_cuda', '—')}"
         + (f", cuDNN {g.get('cudnn')}" if g.get("cudnn") else "")
         + f"; Python {g.get('python', '—')}, NumPy {g.get('numpy', '—')}, "
           f"SciPy {g.get('scipy', '—')}"],
        ["Versions that produced the committed artifacts",
         combo_txt or "—"],
        ["Precision",
         f"{h.get('fp64', '—')}. On the GPU side, {g.get('fp64', '—')}"],
    ]
    body = md.table(["field", "value"], rows, align="ll")
    note = ("\nEvery CSV in the repository is accompanied by a JSON manifest carrying the "
            "git SHA, timestamp, library versions, platform and precision of the run that "
            "wrote it, together with the driver name and its arguments. The two library "
            "combinations above are a fact of the record rather than an inconsistency: the "
            "host environment was upgraded during the work, and each artifact states which "
            "combination produced it. The eigenvalue conclusions do not depend on that "
            "difference — the regression suite runs against the committed data under the "
            "current environment.\n")
    return body + note


def appendix_c_numerics(data_dir: str = DATA_DIR) -> str:
    """부록 C — 재현에 필요한 수치 세부. **값은 코드·데이터에서 읽는다.**

    검토 체크리스트가 요구한 항목들이다. 산문으로 흩어 두면 어긋나므로 한 곳에 모으고,
    격자·자유도·회전각 같은 값은 커밋된 CSV와 드라이버에서 가져온다.
    """
    import inspect

    conv = _rows(data_dir, "p4_convergence")
    best = "3.0"
    lv = sorted({(int(float(r["n"])), int(float(r["n_dof"])))
                 for r in conv if str(float(r["beta"])) == best})
    seq = ", ".join(f"n = {n} ({d} dof)" for n, d in lv)

    alpha = "π/5"
    try:                                     # 회전각의 정본은 드라이버다
        from ..drivers import run_p2
        src = inspect.getsource(run_p2.main)
        if "alpha=np.pi / 5" in src:
            alpha = "π/5 (36°)"
    except Exception:
        pass

    bad = [r for r in _rows(data_dir, "p1_conditioning")
           if str(r["basis"]) == "monomial_raw" and not bool(r["cholesky_ok"])]
    bad_n = min((int(r["n_dof"]) for r in bad), default=None)
    bad_row = next((r for r in bad if int(r["n_dof"]) == bad_n), {})

    items = [
        ("P4 reference by Richardson extrapolation",
         f"Graded Q2 meshes with grading exponent β = {best} on the sequence {seq}, each "
         f"level halving the element size, so the refinement ratio is 2. `h` is the "
         f"parameter of the graded map, i.e. the uniform reference spacing 1/n before "
         f"grading, not a local element diameter — grading redistributes elements but does "
         f"not change the ratio between levels. With the three coarsest-to-finest values "
         f"λ₁, λ₂, λ₃ the model is λ_h = λ_* + C·h^p; the observed order is "
         f"p = log₂((λ₁ − λ₂)/(λ₂ − λ₃)) and the reference is "
         f"λ_* = λ₃ − (λ₂ − λ₃)/(2^p − 1). The quoted uncertainty is the relative "
         f"extrapolation step |λ₃ − λ_*|/|λ_*|, i.e. how far the finest computed level "
         f"still sits from the extrapolant; it is not a statistical interval. A level "
         f"sequence that is not monotone in the required sense (d₁/d₂ ≤ 1) returns "
         f"rate = NaN and uncertainty = ∞ rather than a number. Per-level eigenvalues for "
         f"all three gradings are in `p4_convergence.csv`."),
        ("P2 degenerate-pair rotation",
         f"The rotated pair of Table 8 is generated by rotating the plate through "
         f"α = {alpha} about its axis. For a nodal-diameter number m that rotation acts on "
         f"the (cos mθ, sin mθ) pair as a plane rotation by mα, which is how it is applied "
         f"in the script; the angle is a declared input and is not inferred from the "
         f"resulting MAC values."),
        ("Subspace-trace and log-determinant objectives",
         "Both are evaluated from the reduced pencil (K_Φ, M_Φ) with "
         "M_Φ = ΦᵀWΦ and K_Φ = (Φ″)ᵀWΦ″ formed at the quadrature nodes and symmetrized as "
         "½(X + Xᵀ) before use. The absolute trace is "
         "`trace(torch.linalg.solve(M_Φ, K_Φ))`; the log-determinant form is "
         "`slogdet(K_Φ) − slogdet(M_Φ)`, which is why it needs no eigendecomposition and "
         "costs the same. **No jitter, no regularization and no pseudoinverse are used, and "
         "no rank truncation happens inside either objective** — truncation occurs only at "
         "extraction for arm (d). The one guard is on the log form: if either `slogdet` "
         "returns a non-positive sign, meaning the reduced matrix is numerically indefinite, "
         "the objective returns a fixed penalty of 1e6 for that step instead of a "
         "non-finite value, so the optimizer is pushed away from that region rather than "
         "receiving a NaN. The absolute-trace form has no such guard: a singular M_Φ would "
         "propagate whatever `solve` returns."),
        ("Backward error, including rows where the Cholesky column is `no`",
         f"η = ‖Kx − λMx‖₂ / ((‖K‖₂ + |λ|‖M‖₂)‖x‖₂) for the reported eigenpair. The "
         f"`factorization` column reports an **explicit** Cholesky of M, which is used only "
         f"to build the transformed operator M^(−1/2)KM^(−1/2); the eigenpairs come from a "
         f"separate `scipy.linalg.eigh(K, M)` call. The two can disagree, and at raw "
         f"monomials with N = {bad_n} they do: the explicit factorization fails while `eigh` "
         f"still returns, so η is defined there "
         f"({md.fmt(bad_row.get('backward_error'))}) even though the column reads `fail` and "
         f"κ₂ of the transformed operator is left blank. That disagreement is itself the "
         f"measurement — the raw mass matrix is at the edge of fp64 positive definiteness "
         f"(κ₂(M) = {md.fmt(bad_row.get('kappa_M_raw'))}), and the mpmath cross-check at "
         f"dps = 50 differs from fp64 by "
         f"{md.fmt(bad_row.get('Lam_absdiff_rel'))} relative on the lowest eigenvalue."),
        ("Numerical-rank cutoffs",
         "Ranks are reported at relative cutoffs 1e-10 and 1e-12, which are the two "
         "primary conventions used in the text. The 1e-16 column is retained only as a "
         "sensitivity diagnostic at the edge of fp64 resolution and no claim rests on it; a "
         "single integer would hide the choice, which is why more than one appears."),
        ("Repository metadata",
         "The data and code availability record carries the repository DOI, the git SHA of "
         "the commit that produced each artifact (in every manifest), the CPU and GPU "
         "models, library versions, the operating system, and confirmation that every "
         "reported solve runs in fp64. The DOI is minted at submission; the remaining "
         "fields are written by the manifest builder rather than typed."),
    ]
    body = md.table(["item", "specification"],
                    [[k, v] for k, v in items], align="ll")
    return body + "\n"


def appendix_b_disclosure(data_dir: str = DATA_DIR) -> str:
    """부록 B — arm별 구현 공시. 값의 정본은 코드이고 `DISCLOSED`가 그 사본이다."""
    w, dep = DISCLOSED["width"], DISCLOSED["depth"]
    npar_st = DISCLOSED["spacetime_params"]
    rows = [
        ["(a) penalty deflation", f"MLP {w}×{dep}, tanh", "Adam 2e-3, StepLR",
         "4000", "50", "Gauss 256",
         "φ = x²·N(x); objective R(φ) + w·Σ_k ⟨φ,φ_k⟩²/(‖φ‖²‖φ_k‖²), i.e. a sum of "
         "squared cosines — each term is bounded by 1, so the whole penalty is bounded "
         "by w·k while R grows like Λ_m; the penalty-to-objective ratio therefore "
         "scales as w/Λ_m"],
        ["(a′) diagonal-approx. projection", f"MLP {w}×{dep}, tanh", "Adam 2e-3, StepLR",
         "4000", "50", "Gauss 256", "cₖ = ⟨φ,φₖ⟩/⟨φₖ,φₖ⟩"],
        ["(b) exact M-orthogonal projection", f"MLP {w}×{dep}, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256",
         "φ̃ = φ − Φ(ΦᵀWΦ)⁻¹ΦᵀWφ"],
        ["(c) coarse-Ritz curriculum", f"MLP {w}×{dep}, tanh", "Adam 2e-3, StepLR",
         "4000", "50", "Gauss 256", "stage-1 fraction 0.5, then (b)"],
        ["(d) separate-net neural basis", f"9 × MLP {w}×{dep}, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256",
         "subspace trace; reduced GEP with rank truncation 1e-12"],
        ["(e) shared-trunk subspace", f"MLP {w}×{dep} with 6 outputs, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256", "subspace trace"],
        ["(f) Eig-PIELM", "random tanh features, fixed", "none (no backprop)",
         "—", "50", "Gauss 512",
         "φⱼ = x²·tanh(aⱼx+bⱼ); rank-truncated GEP"],
        ["(g) space–time PINN",
         f"MLP {DISCLOSED['spacetime_width']}×{DISCLOSED['spacetime_depth']} "
         f"on (x,t), tanh", f"Adam {ST['lr']:g}, StepLR",
         f"{int(ST['iters'])}", "3",
         f"{int(ST['n_col'])} random collocation points per iteration, resampled "
         f"every iteration",
         # **자유단 조건이 손실에 실제로 들어간다.** 예전 공시는 `w = x²·N`만 적어
         # "자연경계조건을 강형식에서 부과하지 않았다"로 읽힐 수 있었다. 코드는
         # `spacetime.py`에서 x = 1의 w_xx·w_xxx 잔차를 벌점으로 넣는다.
         f"strong form `w_tt + w_xxxx = 0` on (0,1) × (0,T); trial field "
         f"w = x²·N(x,t), which imposes the clamped end `w(0,t) = w_x(0,t) = 0` "
         f"exactly; the free end `w_xx(1,t) = w_xxx(1,t) = 0` is imposed by penalty "
         f"on {int(ST['n_bc'])} times drawn once from U(0,T) and held fixed; initial "
         f"state is the analytic clamped–free mode 1 at rest, `w(x,0) = φ₁(x)` and "
         f"`w_t(x,0) = 0`, on {int(ST['n_ic'])} uniform x; loss = PDE residual "
         f"(weight 1) + {ST['w_ic']:g} · initial-state residual + {ST['w_bc']:g} · "
         f"free-end residual, weights chosen so the terms are of comparable size and "
         f"not tuned; the eigenvalue is read afterwards from the tip response at "
         f"x = 1 sampled at {int(ST['n_probe'])} uniform times over [0,T] "
         f"(Δt = T/{int(ST['n_probe'])}; the window endpoint is excluded so the record "
         f"length is exactly T and the reference frequency falls on a bin), mean removed, "
         f"**no taper or window**, "
         f"peak taken as the largest |rFFT| bin excluding DC with no interpolation; "
         f"{npar_st} parameters"],
        ["(b) on P4, two dimensions",
         f"MLP {DISCLOSED['p4_width']}×{DISCLOSED['p4_depth']} on (x,y) with two "
         f"outputs, tanh", "Adam 2e-3, StepLR", "4000", "50",
         f"tensor Gauss, {DISCLOSED['p4_n_per_block']}² per unit block "
         f"({DISCLOSED['p4_n_q']} points)",
         f"u = φ_Ω(x,y)·N(x,y) with φ_Ω the R-equivalence approximate distance "
         f"function of Sukumar and Srivastava; {DISCLOSED['p4_params']} parameters; exact "
         f"mass-orthogonal projection applied to the field and its gradients; "
         f"per-mode wall-clock measured directly at a single seed as the median of "
         f"three runs, the same basis as the canonical ratio table"]]
    body = md.table(["arm", "architecture", "optimizer", "iterations", "seeds",
                     "quadrature", "notes"], rows, align="lllrrll")
    note = ("\nArms (a), (a\u2032), (b) and (c) share architecture, optimizer, schedule, "
            "quadrature and seed derivation, so any difference among those four is "
            "attributable to the one component named in the notes column — they are the "
            "paired ablations of \u00a73.3. Arms (d) to (g) deliberately do not: (d) trains "
            "nine separate networks, (e) a single trunk with six outputs, (f) draws random "
            "features and does not backpropagate at all, and (g) trains on a space\u2013time "
            "domain. Those four are read against (b) as separate designs, not as "
            "one-component ablations. Clamped "
            "boundary conditions are satisfied exactly by construction, not by penalty: "
            "an x^2 factor on the one-dimensional problems, and the R-equivalence "
            "distance function of Sukumar and Srivastava (\u00a73.2) on the L-shaped "
            "domain, where the domain is "
            "non-convex and a product of edge functions would vanish inside it.\n")
    return body + note


