"""P4(L형 평면탄성)에 2차원 신경 arm을 올린다 — **mesh-free 주장이 실제로 시험되는 곳**.

여기까지 신경 arm은 1차원 문제(P1)에서만 돌았고, 재진입 코너 정의역에서는 돌지 않았다.
논문이 "mesh-free 주장이 진짜 걸려 있는 무대"라고 부른 자리가 비어 있던 것이다. 이 모듈이
그것을 채운다.

## 하드 경계조건 — 비볼록 정의역에서 곱셈 다항식은 못 쓴다

Ω = [0,2]×[0,1] ∪ [0,1]×[1,2]이고 **∂Ω 전체가 클램프**다. 1차원에서 쓴 `x²·N(x)` 같은
곱셈 인자를 여기서 만들려고 변마다 선형함수를 곱하면, 재진입 변의 **연장선이 정의역
내부를 지난다**(예: x=1은 0<y<1 구간에서 Ω의 내부다). 그런 인자는 내부 직선 위에서 해를
0으로 강제해 버린다.

그래서 Sukumar & Srivastava [3]의 **정규화 근사거리(ADF)와 R-등가**를 쓴다. 변 i마다

    φ_i = √(d_i² + ((√(t_i² + d_i⁴) − t_i)/2)²)

로 선분에서만 0이 되는 매끄러운 함수를 만들고(d_i는 변을 담은 직선까지의 부호거리,
t_i는 선분 밖을 잘라내는 접선항), 이를 R-등가

    φ_Ω = (Σ_i φ_i^{−m})^{−1/m},   m = 2

로 합친다. φ_Ω는 ∂Ω에서 정확히 0이고 내부에서 양수이며 꼭짓점을 뺀 모든 곳에서 매끄럽다.
시행함수는

    u(x,y) = φ_Ω(x,y) · N(x,y),    N: R² → R² (tanh MLP)

이므로 클램프 조건이 **구조적으로** 만족되고 경계 벌점이 없다 — P1에서와 같은 계약이다.

## 약형식과 무차원화

평면응력, L = E = ρ = 1, ν = 0.29. 변형률 ε = (u_x, v_y, u_y+v_x)에 대해

    Ω² = ∫ εᵀCε dΩ / ∫ (u² + v²) dΩ,   C = 1/(1−ν²)·[[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]]

이고 이것이 `p4_lshape.solve`가 내는 것과 같은 무차원 고유값이다. H¹ 문제라 1차 도함수만
필요하고, P1의 H² 문제보다 자동미분이 싸다.

## 구적

정의역을 세 단위블록 A=[0,1]², B=[1,2]×[0,1], C=[0,1]×[1,2]로 쪼개 각 블록에 텐서
Gauss–Legendre를 깐다. 블록 경계가 곧 재진입 코너를 지나는 선이므로 적분이 특이점을
가로지르지 않는다. **코너의 응력특이성(u ~ r^λ, λ<1)은 시행공간이 매끄러워 표현되지
않는다** — 그것이 이 문제에서 재는 e_approx이고, 고전 등급메시와의 대비가 §5.4의 논점이다.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, stack_module_state, vmap

from ..problems.p4_lshape import P4_GEOMETRY
from ..quadrature import gauss_legendre

# ∂Ω의 여섯 변 (반시계). 재진입 코너는 (1,1).
EDGES = (((0.0, 0.0), (2.0, 0.0)),      # 아래
         ((2.0, 0.0), (2.0, 1.0)),      # 오른쪽
         ((2.0, 1.0), (1.0, 1.0)),      # B의 위 = 재진입
         ((1.0, 1.0), (1.0, 2.0)),      # C의 오른쪽 = 재진입
         ((1.0, 2.0), (0.0, 2.0)),      # 위
         ((0.0, 2.0), (0.0, 0.0)))      # 왼쪽
BLOCKS = (((0.0, 1.0), (0.0, 1.0)),
          ((1.0, 2.0), (0.0, 1.0)),
          ((0.0, 1.0), (1.0, 2.0)))


def quadrature(n_per_block: int = 20, device: str = "cpu"):
    """세 단위블록의 텐서 Gauss. 반환 (pts(n,2), w(n,))."""
    P, W = [], []
    for (x0, x1), (y0, y1) in BLOCKS:
        xs, wx = gauss_legendre(n_per_block, x0, x1)
        ys, wy = gauss_legendre(n_per_block, y0, y1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P.append(np.stack([X.ravel(), Y.ravel()], 1))
        W.append(np.outer(wx, wy).ravel())
    p = torch.tensor(np.concatenate(P), dtype=torch.float64, device=device)
    w = torch.tensor(np.concatenate(W), dtype=torch.float64, device=device)
    return p, w


def _edge_adf(xy, a, b):
    """변 하나의 정규화 근사거리 [3]. 선분 위에서 0, 그 밖에서 양수·매끄럽다."""
    ax, ay = a
    bx, by = b
    L = float(np.hypot(bx - ax, by - ay))
    x, y = xy[..., 0], xy[..., 1]
    d = ((x - ax) * (by - ay) - (y - ay) * (bx - ax)) / L        # 직선까지 부호거리
    cx, cy = 0.5 * (ax + bx), 0.5 * (ay + by)
    t = ((0.5 * L) ** 2 - ((x - cx) ** 2 + (y - cy) ** 2)) / L   # 선분 밖 절단항
    inner = torch.sqrt(t ** 2 + d ** 4)
    return torch.sqrt(d ** 2 + (0.5 * (inner - t)) ** 2)


def boundary_adf(xy, m: int = 2, eps: float = 1e-300):
    """φ_Ω = (Σ φ_i^{−m})^{−1/m}. ∂Ω에서 정확히 0, 내부에서 양수."""
    acc = None
    for a, b in EDGES:
        phi = _edge_adf(xy, a, b).clamp_min(eps)
        term = phi ** (-m)
        acc = term if acc is None else acc + term
    return acc ** (-1.0 / m)


class _VecMLP(nn.Module):
    """N: R² → R². 시행함수는 u = φ_Ω·N이므로 여기서 BC를 신경쓰지 않는다."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class EnsembleVecMLP:
    """시드별 파라미터를 배치 텐서로 쌓아 vmap 한 번에 전 시드를 학습한다(P1과 같은 방식)."""

    def __init__(self, n_seeds: int, width: int = 64, depth: int = 4,
                 seed: int = 0, device: str | None = None):
        torch.set_default_dtype(torch.float64)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        models = [_VecMLP(width, depth).to(dev) for _ in range(n_seeds)]
        self.params, _ = stack_module_state(models)
        self.base = _VecMLP(width, depth).to(dev).to("meta")
        self.n_seeds, self.width, self.depth, self.device = n_seeds, width, depth, dev

    def disclosure(self) -> dict:
        n = sum(v[0].numel() for v in self.params.values())
        return {"width": self.width, "depth": self.depth, "activation": "tanh",
                "n_params_per_seed": int(n), "n_seeds": self.n_seeds,
                "ansatz": "u = phi_Omega(x,y) * N(x,y), phi_Omega = R-equivalence ADF"}


def _u_comp(p, base, pt, j: int):
    """한 점에서 u_j — grad로 미분하기 위한 스칼라 진입점."""
    n = functional_call(base, (p, {}), (pt.reshape(1, 2),)).reshape(-1)
    return boundary_adf(pt.reshape(1, 2)).reshape(()) * n[j]


def _state_one(p, base, pts):
    """한 시드의 상태 (6, n_q) = (u, v, u_x, u_y, v_x, v_y).

    정확 질량-직교 사영은 장(場)뿐 아니라 **변형률까지** 같은 계수로 빼야 한다 —
    사영된 시행함수의 Rayleigh 몫을 재려면 그 함수의 도함수가 필요하다. 그래서 여섯
    성분을 한 텐서로 들고 다닌다."""
    g0 = vmap(lambda t: grad(_u_comp, argnums=2)(p, base, t, 0))(pts)   # (n_q,2)
    g1 = vmap(lambda t: grad(_u_comp, argnums=2)(p, base, t, 1))(pts)
    u = vmap(lambda t: _u_comp(p, base, t, 0))(pts)
    v = vmap(lambda t: _u_comp(p, base, t, 1))(pts)
    return torch.stack([u, v, g0[:, 0], g0[:, 1], g1[:, 0], g1[:, 1]], 0)


def _rq_from_state(S, w, nu: float):
    """상태에서 Ω² = 변형에너지/질량."""
    u, v, ux, uy, vx, vy = S
    c = 1.0 / (1.0 - nu ** 2)
    gxy = uy + vx
    num = (w * c * (ux ** 2 + vy ** 2 + 2.0 * nu * ux * vy
                    + 0.5 * (1.0 - nu) * gxy ** 2)).sum()
    den = (w * (u ** 2 + v ** 2)).sum()
    return num / den


def _project_exact(S, P, w):
    """정확 질량-직교 사영. S:(6,n_q), P:(k,6,n_q) → 사영된 상태 (6,n_q).

    질량 내적 ⟨a,b⟩ = ∫(a_u b_u + a_v b_v)는 성분 0·1만 쓰고, 얻은 계수를 **여섯 성분
    전부에** 적용한다 — 선형연산이므로 도함수도 같은 계수로 따라간다."""
    G = torch.einsum("aij,j,bij->ab", P[:, :2], w, P[:, :2])
    r = torch.einsum("aij,j,ij->a", P[:, :2], w, S[:2])
    c = torch.linalg.solve(
        G + 1e-14 * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), r)
    return S - torch.einsum("a,anj->nj", c, P)


def solve_p4_neural(n_modes: int = 3, n_seeds: int = 50, iters: int = 4000,
                    n_per_block: int = 20, seed: int = 0, lr: float = 2e-3,
                    width: int = 64, depth: int = 4, snapshot_every: int = 25,
                    geometry=None, device: str | None = None) -> dict:
    """L형 평면탄성의 최저 n_modes개 Ω²를 순차 Rayleigh 최소화로 푼다.

    deflation은 P1의 arm (b)와 같은 **정확 질량-직교 사영**이다. 그 arm이 P1에서 모드 10까지
    벽 없이 갔으므로 2차원 비교의 기준선으로 쓴다 — 여기서 실패하면 그것은 deflation 기전이
    아니라 정의역·시행공간 쪽의 문제다.
    """
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = P4_GEOMETRY if geometry is None else geometry
    nu = float(g["nu"])
    pts, w = quadrature(n_per_block, dev)

    lam, shapes, prevs, stage_secs, hists = [], [], [], [], []
    for k in range(n_modes):
        ens = EnsembleVecMLP(n_seeds, width, depth, seed=seed * 1000 + k, device=dev)
        opt = torch.optim.Adam(list(ens.params.values()), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(iters // 3, 1), gamma=0.4)

        def loss_one(p, *prev):
            S = _state_one(p, ens.base, pts)
            if prev:
                S = _project_exact(S, torch.stack(prev, 0), w)
            return _rq_from_state(S, w, nu)

        def snap_one(p, *prev):
            S = _state_one(p, ens.base, pts)
            if prev:
                S = _project_exact(S, torch.stack(prev, 0), w)
            return _rq_from_state(S, w, nu)

        t0, hist = time.perf_counter(), []
        prev_stack = tuple(prevs)          # 각 원소가 (n_seeds, 6, n_q) — vmap이 시드축을 맵
        for it in range(iters + 1):
            r = vmap(loss_one)(ens.params, *prev_stack)
            if it % snapshot_every == 0 or it == iters:
                with torch.no_grad():
                    R = vmap(snap_one)(ens.params, *prev_stack)
                hist.append((it, time.perf_counter() - t0, R.cpu().numpy()))
            if it == iters:
                break
            r.sum().backward()
            opt.step()
            sched.step()
            opt.zero_grad()
        stage_secs.append(time.perf_counter() - t0)
        hists.append(hist)

        with torch.no_grad():
            St = vmap(lambda p: _state_one(p, ens.base, pts))(ens.params)  # (S,6,n_q)
            if prev_stack:
                St = vmap(lambda a, *b: _project_exact(a, torch.stack(b, 0), w))(
                    St, *prev_stack)
            R = vmap(lambda a: _rq_from_state(a, w, nu))(St)
        lam.append(R.clamp_min(0).cpu().numpy())
        shapes.append(St[:, :2].cpu().numpy())
        prevs.append(St)

    return {"arm": "p4_neural_2d[exact projection]",
            "lam": np.array(lam), "shapes": np.array(shapes),
            "history": hists, "stage_seconds": stage_secs,
            "seconds": float(sum(stage_secs)),
            "pts": pts.cpu().numpy(), "w": w.cpu().numpy(),
            "n_q": int(pts.shape[0]), "n_seeds": n_seeds, "iters": iters,
            "n_per_block": n_per_block,
            "disclosure": EnsembleVecMLP(1, width, depth,
                                         device="cpu").disclosure()}
