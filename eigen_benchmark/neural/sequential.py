"""순차 deflation solver — 모드 1..n을 차례로 잡고 사전등록 규칙으로 분류한다.

수렴한 하위모드는 **파라미터로 보관하고 현재 노드에서 재평가**한다. 고정 Gauss면
매번 같은 값이지만 MC 대조군에서는 노드가 바뀌므로 재평가가 필수이고, 그 비용까지가
MC의 실제 비용이다(스파이크: Gauss 402 s vs MC 3279 s).

**평가도 사영된 φ̃로 한다.** raw φ로 재면 하위모드 성분이 남아 λ가 β₂⁴ 아래로
내려가는 물리적으로 불가능한 값이 나온다(스파이크 1차 실행의 실제 오류).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.func import vmap

from .. import metrics as mt
from ..problems import p1_beam as p1
from . import core
from .net import EnsembleMLP


def _converged(hist, tol: float = 1e-3, tail: float = 0.1) -> np.ndarray:
    """마지막 tail 구간의 R 상대변화가 tol 이하면 수렴(사전등록 규칙)."""
    if len(hist) < 3:
        return np.zeros_like(hist[-1][2], dtype=bool)
    k = max(int(len(hist) * tail), 2)
    a, b = hist[-k][2], hist[-1][2]
    rel = np.abs(b - a) / np.maximum(np.abs(b), 1e-300)
    return np.isfinite(b) & (rel <= tol)


def _prev_stack(prevs, base, xs):
    """이전 모드들을 현재 노드에서 재평가 — (n_seeds, k, n_q) 두 개."""
    PH, PH2 = [], []
    for pr in prevs:
        a, b = vmap(lambda q: core.eval_phi_one(q, base, xs))(pr)
        PH.append(a)
        PH2.append(b)
    return torch.stack(PH, 1), torch.stack(PH2, 1)


def solve_sequential(n_modes: int, deflation, n_seeds: int = 50, iters=4000,
                     nodes: str = "gauss", n_q: int = 256, init=None,
                     seed: int = 0, lr: float = 2e-3, snapshot_every: int = 25,
                     n_mc: int = 2000, barrier=None,
                     device: str | None = None) -> dict:
    """`iters`는 정수이거나 **모드별 예산 시퀀스**다.

    `barrier`는 사다리 I2용이다 — 초기화가 아니라 **목적함수에 붙는 항**으로, 목표
    고유값 구간만 아는 상태를 모형화한다(형상정보는 주지 않는다). 그래서 `init`과
    배타적으로 쓰이고, 이것이 없으면 I2는 사다리에서 조용히 빠진다(실제로 빠져 있었다).

    장기변주에서 이미 100 %인 저차모드까지 5배로 돌리는 것은 낭비이므로,
    실패가 관측된 모드에만 큰 예산을 준다. 예: [4000]*8 + [20000]*2."""
    torch.set_default_dtype(torch.float64)
    budget = ([int(iters)] * n_modes if np.isscalar(iters)
              else [int(v) for v in iters])
    if len(budget) != n_modes:
        raise ValueError(f"iters 시퀀스 길이({len(budget)})가 n_modes({n_modes})와 다릅니다")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(20260802 + seed)
    xs, wq = core.gauss_nodes(n_q, dev)

    # `shapes`는 **사영된** 함수이고 `prevs`(→ 다음 단계 deflation의 Φ)는 **원시 망 출력**이다.
    # 이 구분이 (a′)와 (b)를 가르는 지점이므로 원시 출력도 반환한다 — 진단이 사영된 쪽을
    # 재면 그람이 구성상 대각이 되어 "두 arm이 동일하다"는 잘못된 결론이 나온다.
    lam, shapes, raw_shapes, hists, prevs, stage_secs = [], [], [], [], [], []
    init_secs = []
    total = 0.0
    for k in range(n_modes):
        ens = EnsembleMLP(n_seeds, seed=seed * 1000 + k, device=dev)
        # **사전적합 비용을 따로 센다.** I3·I5는 목표에 H² 적합을 먼저 돌리므로 공짜가
        # 아니다. 이걸 청구하지 않으면 "warm-start는 이득이 없다"가 "이득도 없고 비용도
        # 없다"로 읽혀, 실제 결론("이득이 없는데 비용은 든다 = 열등")을 놓친다.
        _t_init = time.perf_counter()
        if init is not None:
            init(ens, k, xs, wq)                     # 사다리 I1–I5
        init_secs.append(time.perf_counter() - _t_init)

        # MC 노드는 **vmap 밖에서** 반복마다 새로 뽑는다(vmap 안 난수는 금지된다).
        node_state = {"x": xs, "w": wq}

        def redraw(_it):
            if nodes == "mc":
                node_state["x"], node_state["w"] = core.mc_nodes(n_mc, gen, dev)

        def loss_one(p, *pv):
            x, w = node_state["x"], node_state["w"]
            ph, d2 = core.eval_phi_one(p, ens.base, x)
            if pv:
                PH = torch.stack([core.eval_phi_one(q, ens.base, x)[0].detach()
                                  for q in pv])
                PH2 = torch.stack([core.eval_phi_one(q, ens.base, x)[1].detach()
                                   for q in pv])
                ph, d2, pen = deflation.apply(ph, d2, PH, PH2, w)
            else:
                pen = ph.new_zeros(())
            R = (w * d2 ** 2).sum() / (w * ph ** 2).sum()
            if barrier is not None:
                pen = pen + barrier(R)       # I2: 고유값 창 배리어
            return R + pen

        out = core.train(ens, loss_one, budget[k], lr, snapshot_every,
                         extra_batched=tuple(prevs), on_iter=redraw)
        total += out["seconds"]
        stage_secs.append(out["seconds"])
        hists.append(out["history"])

        with torch.no_grad():
            ph, d2 = core.eval_phi(ens, xs)
            raw_shapes.append(ph.cpu().numpy())          # 사영 전 원시 출력
            if prevs:
                PH, PH2 = _prev_stack(prevs, ens.base, xs)
                ph, d2, _ = vmap(lambda a, b, c, e:
                                 deflation.apply(a, b, c, e, wq))(ph, d2, PH, PH2)
        r = core.rayleigh(ph, d2, wq).clamp_min(0).cpu().numpy()
        lam.append(r ** 0.25)
        shapes.append(ph.cpu().numpy())
        prevs.append(ens.clone_params())

    return {"lam": np.array(lam), "shapes": np.array(shapes),
            "raw_shapes": np.array(raw_shapes), "history": hists,
            "seconds": total, "xs": xs.cpu().numpy(), "wq": wq.cpu().numpy(),
            "converged": np.array([_converged(h) for h in hists]),
            "deflation": deflation.name, "nodes": nodes, "n_q": n_q,
            "iters": budget, "n_seeds": n_seeds, "stage_seconds": stage_secs,
            "init_seconds": init_secs,
            "arm": f"sequential[{deflation.name}]"}


def classify_stage(shapes, lam, target: int, xs, wq, n_ref: int = 14,
                   converged=None):
    """사전등록 4분기 판정(계획 1 `metrics.classify` 위임). 기준 모드형은 해석해."""
    REF = np.stack([p1.analytic_mode(xs, n) for n in range(1, n_ref + 1)], axis=1)
    ref_lam = p1.beta_roots(n_ref) ** 4
    W = np.diag(wq)
    out = []
    for i in range(shapes.shape[0]):
        ok = True if converged is None else bool(converged[i])
        out.append(mt.classify(shapes[i], float(lam[i]) ** 4, REF, ref_lam,
                               target=target, converged=ok, M=W))
    return out
