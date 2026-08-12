"""P4(L형) 2차원 신경 arm — 논문이 "mesh-free 주장이 진짜 시험되는 무대"라 부른 자리.

P1의 arm (b)(정확 질량-직교 사영)와 같은 deflation을 2차원 재진입 코너 정의역에 올려,
같은 사전등록 판정규칙(MAC ≥ 0.9, e_λ ≤ 0.05, 4분류, 50시드, Wilson)으로 채점한다.
참조는 Richardson 외삽 Q2 등급메시이고, MAC은 FEM 모드를 신경 arm과 **같은 구적점**에서
평가해 잰다(`p4_lshape.mode_at`).

비교 기준을 P1과 맞추기 위해 고전 쪽도 같은 축에 올린다 — 등급 Q2 FEM의 (n_dof, 초, e_λ).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import metrics as mt
from ..cost import expected_time_to_success
from ..problems import p4_lshape as p4
from . import manifest


def _mass_mac(A, B, w) -> float:
    """질량 가중 MAC — 벡터장이므로 두 성분을 함께 내적한다."""
    ab = float((w * (A[0] * B[0] + A[1] * B[1])).sum())
    aa = float((w * (A[0] ** 2 + A[1] ** 2)).sum())
    bb = float((w * (B[0] ** 2 + B[1] ** 2)).sum())
    if aa <= 0 or bb <= 0:
        return float("nan")
    return ab ** 2 / (aa * bb)


def main(outdir=None, n_modes: int = 3, n_seeds: int = 50, iters: int = 4000,
         n_per_block: int = 20, n_ref: int = 8, beta: float = 3.0,
         quick: bool = False, device=None) -> dict:
    from ..neural.p4_neural import solve_p4_neural
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters, n_per_block, n_ref = 1, 3, 20, 8, 4

    out = solve_p4_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                          n_per_block=n_per_block, device=device)
    pts, w = out["pts"], out["w"]

    # 참조: 외삽값이 있으면 그것을, 없으면 세밀 격자 직접해
    ref = p4.solve(n_ref, beta=beta, n_modes=max(n_modes + 3, 6))
    lam_ref = np.sort(ref["Lam"])
    try:
        ext = {int(r["mode"]): float(r["lambda_extrapolated"])
               for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    except Exception:
        ext = {}
    ref_shapes = [np.asarray(p4.mode_at(ref, m, pts)).T for m in
                  range(1, len(lam_ref) + 1)]        # 각 (2, n_q)

    rows, recs = [], []
    for m in range(1, n_modes + 1):
        target = ext.get(m, float(lam_ref[m - 1]))
        cnt = {k: 0 for k in mt.OUTCOMES}
        el, macs, matched = [], [], []
        for s in range(n_seeds):
            lam = float(out["lam"][m - 1][s])
            phi = out["shapes"][m - 1][s]                  # (2, n_q)
            mac_all = [_mass_mac(phi, r, w) for r in ref_shapes]
            j = int(np.nanargmax(mac_all)) + 1
            mac = float(mac_all[j - 1])
            e = abs(lam - target) / target if math.isfinite(lam) else float("nan")
            conv = _converged(out["history"][m - 1], s)
            oc = ("non_converged" if not conv or not math.isfinite(e)
                  else "correct" if (j == m and mac >= mt.MAC_MIN
                                     and e <= mt.ELAM_MAX)
                  else "lower_mode_basin" if (j < m and mac >= mt.MAC_MIN)
                  else "spurious")
            cnt[oc] += 1
            el.append(e); macs.append(mac); matched.append(j)
            recs.append({"arm": out["arm"], "problem": "P4", "mode": m, "seed": s,
                         "ladder": "I0", "nodes": "gauss2d", "outcome": oc,
                         "matched_mode": j, "mac": mac, "e_lam": e,
                         "Lam": lam, "Lam_ref": target,
                         "seconds": out["stage_seconds"][m - 1],
                         "iters": iters, "n_q": out["n_q"]})
        lo, hi = mt.wilson(cnt["correct"], n_seeds)
        acc = sum(1 for j, mm, e in zip(matched, macs, el)
                  if j == m and mm >= mt.MAC_MIN and math.isfinite(e)
                  and e <= mt.ELAM_MAX)
        secs = float(out["stage_seconds"][m - 1])
        rows.append({"problem": "P4", "arm": out["arm"], "mode": m,
                     "ladder": "I0", "nodes": "gauss2d", "n": n_seeds,
                     "p_correct": cnt["correct"] / n_seeds,
                     "wilson_lo": lo, "wilson_hi": hi, **cnt,
                     "p_accurate": acc / n_seeds,
                     "e_lam_median": float(np.nanmedian(el)),
                     "mac_median": float(np.nanmedian(macs)),
                     "Lam_median": float(np.median(out["lam"][m - 1])),
                     "Lam_ref": target, "iters": iters,
                     "seconds": secs, "n_q": out["n_q"],
                     "E_T_success": expected_time_to_success(
                         secs, cnt["correct"] / n_seeds)})
        print(f"  [P4 neural m{m}] p_correct={rows[-1]['p_correct']:.2f} "
              f"e_lam={rows[-1]['e_lam_median']:.3e} "
              f"MAC={rows[-1]['mac_median']:.4f} {secs:.0f}s", flush=True)

    manifest.write_csv(os.path.join(d, "p4_neural.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p4_neural_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p4_neural.json"),
                        manifest.build({
                            "driver": "run_p4_neural.main",
                            "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters,
                            "n_per_block": n_per_block, "n_q": out["n_q"],
                            "n_ref": n_ref, "beta": beta,
                            "disclosure": out["disclosure"],
                            "reference": "Richardson-extrapolated graded Q2 where "
                                         "available, else direct fine-grid solve",
                            "mac": "mass-weighted, FEM reference evaluated at the "
                                   "neural arm's own quadrature points"}))
    print(f"[run_p4_neural] {d} — {len(rows)}행")
    return {"p4_neural": rows, "outdir": d}


def _converged(hist, s: int, tol: float = 1e-3, tail: float = 0.1) -> bool:
    """사전등록 수렴규칙을 시드 s에 적용 — P1과 같은 꼬리 10 % 상대변화 기준."""
    if len(hist) < 3:
        return False
    k = max(int(len(hist) * tail), 2)
    a, b = float(hist[-k][2][s]), float(hist[-1][2][s])
    return math.isfinite(b) and abs(b - a) / max(abs(b), 1e-300) <= tol


if __name__ == "__main__":
    main()


def single_seed_cost(outdir=None, n_modes: int = 3, iters: int = 4000,
                     n_per_block: int = 20, reps: int = 3, quick: bool = False,
                     device=None) -> dict:
    """P4의 1회 시도 비용을 직접 잰다 — P1과 같은 회계로 맞춘다.

    §5.4의 P4 비율은 50시드 앙상블 벽시계를 1회 비용으로 쓰고 있어 격차를 **과대**계상했고,
    그래서 상한으로만 인용할 수 있었다. P1에서 이미 확인했듯 앙상블→1회 환산계수는 이 기계
    (GPU를 상주 추론 서비스와 공유)에서 2.53~7.82로 흔들려 쓸 수 없다. 그러니 P1과 마찬가지로
    **필요한 값을 직접 잰다**: n_seeds = 1, reps회, median과 min/max.

    결과는 `p1_single_seed_cost.csv`에 덧붙인다 — `run_p1_compare._single_seed_cost`가
    (arm, mode) 하나의 사전에서 찾도록 출처를 한 곳으로 유지한다.
    """
    from ..neural.p4_neural import solve_p4_neural
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, iters, n_per_block, reps = 1, 20, 8, 1
    per = {m: [] for m in range(1, n_modes + 1)}
    for _ in range(reps):
        out = solve_p4_neural(n_modes=n_modes, n_seeds=1, iters=iters,
                              n_per_block=n_per_block, device=device)
        for m in range(1, n_modes + 1):
            per[m].append(float(out["stage_seconds"][m - 1]))
    arm = "p4_neural_2d[exact projection]"
    path = os.path.join(d, "p1_single_seed_cost.csv")
    keep = [r for r in manifest.read_csv(path)
            if not str(r["arm"]).startswith("p4_neural_2d")] \
        if os.path.exists(path) else []
    rows = []
    for m in range(1, n_modes + 1):
        ts = sorted(per[m])
        med = ts[len(ts) // 2]
        rows.append({"arm": arm, "mode": m, "iters": iters, "reps": len(ts),
                     "seconds_per_attempt": med, "seconds_min": ts[0],
                     "seconds_max": ts[-1],
                     "spread_rel": (ts[-1] - ts[0]) / max(med, 1e-300),
                     "measured": "directly at n_seeds = 1 (no batching conversion); "
                                 "P4, two dimensions"})
        print(f"  [1seed P4 m{m}] median {med:7.1f}s [{ts[0]:.1f}, {ts[-1]:.1f}]",
              flush=True)
    manifest.write_csv(path, keep + rows)
    manifest.write_json(os.path.join(d, "manifest_p4_single_seed_cost.json"),
                        manifest.build({
                            "driver": "run_p4_neural.single_seed_cost",
                            "n_modes": n_modes, "iters": iters, "reps": reps,
                            "n_per_block": n_per_block,
                            "why": "the P4 cost rows used the 50-seed ensemble wall-clock "
                                   "as if it were a per-attempt cost, so the gap was an "
                                   "upper bound; measuring at one seed puts P4 on the "
                                   "same accounting as P1"}))
    print(f"[p4 single_seed_cost] {len(rows)}행 (p1_single_seed_cost.csv에 병합)")
    return {"single_seed": rows, "outdir": d}
