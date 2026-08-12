"""P4 2차원 신경 arm — 구현이 옳은지부터 고정한다.

이 arm은 "mesh-free 주장이 시험되는 무대"의 결과를 내므로, 하드 BC나 Rayleigh 몫이 틀리면
논문의 결론이 무의미해진다. 그래서 성능이 아니라 **정확성**을 회귀로 잡는다.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import p4_neural as pn        # noqa: E402
from eigen_benchmark.problems import p4_lshape as p4      # noqa: E402


def _boundary_points(n: int = 7):
    pts = []
    for a, b in pn.EDGES:
        for s in np.linspace(0.03, 0.97, n):
            pts.append([a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])])
    return torch.tensor(np.array(pts))


def test_adf_vanishes_exactly_on_the_boundary():
    """하드 BC의 근거 — 비볼록 정의역에서 곱셈 다항식으로는 얻을 수 없는 성질."""
    assert float(pn.boundary_adf(_boundary_points()).abs().max()) == 0.0


def test_adf_is_strictly_positive_inside_including_near_the_reentrant_corner():
    inside = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5],
                           [0.98, 0.98], [1.02, 0.5], [0.5, 1.02]])
    assert float(pn.boundary_adf(inside).min()) > 0.0


def test_trial_field_satisfies_the_clamped_condition_bitwise():
    ens = pn.EnsembleVecMLP(1, device="cpu")
    p0 = {k: v[0] for k, v in ens.params.items()}
    b = _boundary_points()
    u = torch.stack([torch.stack([pn._u_comp(p0, ens.base, t, j) for j in (0, 1)])
                     for t in b])
    assert float(u.abs().max()) == 0.0


def test_quadrature_integrates_the_L_shaped_area_exactly():
    """세 단위블록 텐서 Gauss — 면적 3을 정확히 내야 구적이 맞다."""
    _, w = pn.quadrature(12)
    assert float(w.sum()) == pytest.approx(3.0, abs=1e-12)


def test_quadrature_points_are_all_inside_the_domain():
    pts, _ = pn.quadrature(10)
    x, y = pts[:, 0].numpy(), pts[:, 1].numpy()
    assert np.all((x > 0) & (x < 2) & (y > 0) & (y < 2))
    assert not np.any((x > 1) & (y > 1)), "제외 사분면에 구적점이 있다"


def test_rayleigh_quotient_respects_the_upper_bound_property():
    """임의 시행함수의 Ω²는 최저 고유값 이상이어야 한다 — 약형식이 맞다는 필요조건."""
    pts, w = pn.quadrature(16)
    nu = p4.P4_GEOMETRY["nu"]
    lam1 = float(np.sort(p4.solve(5, beta=3.0, n_modes=3)["Lam"])[0])
    for sd in range(4):
        ens = pn.EnsembleVecMLP(1, seed=sd, device="cpu")
        p0 = {k: v[0] for k, v in ens.params.items()}
        with torch.no_grad():
            R = float(pn._rq_from_state(pn._state_one(p0, ens.base, pts), w, nu))
        assert R >= lam1, (sd, R, lam1)


def test_exact_projection_removes_the_previous_mode_in_the_mass_inner_product():
    """정확 질량-직교 사영 — 사영 후 앞 모드와의 질량 내적이 0이어야 한다."""
    pts, w = pn.quadrature(10)
    g = torch.Generator().manual_seed(0)
    P = torch.randn(2, 6, pts.shape[0], generator=g, dtype=torch.float64)
    S = torch.randn(6, pts.shape[0], generator=g, dtype=torch.float64)
    Sp = pn._project_exact(S, P, w)
    for a in range(P.shape[0]):
        ip = float((w * (P[a, 0] * Sp[0] + P[a, 1] * Sp[1])).sum())
        assert abs(ip) < 1e-10, (a, ip)


def test_projection_applies_the_same_coefficients_to_the_derivatives():
    """장만 빼고 도함수를 안 빼면 사영된 함수의 Rayleigh 몫이 그 함수의 것이 아니다."""
    pts, w = pn.quadrature(8)
    g = torch.Generator().manual_seed(1)
    P = torch.randn(1, 6, pts.shape[0], generator=g, dtype=torch.float64)
    S = torch.randn(6, pts.shape[0], generator=g, dtype=torch.float64)
    Sp = pn._project_exact(S, P, w)
    c = (S - Sp)[0] / P[0, 0]                     # 성분 0에서 역산한 계수
    for n in range(1, 6):                          # 나머지 다섯 성분도 같은 계수여야
        assert torch.allclose((S - Sp)[n], c * P[0, n], atol=1e-10), n


def test_fem_reference_can_be_evaluated_at_the_neural_quadrature_points():
    """MAC을 재려면 두 해를 같은 격자에서 봐야 한다 — 평가기가 빈틈없이 덮는지."""
    pts, w = pn.quadrature(12)
    r = p4.solve(5, beta=3.0, n_modes=2)
    U = p4.mode_at(r, 1, pts.numpy())
    assert not np.isnan(U).any(), int(np.isnan(U[:, 0]).sum())
    m = float((w.numpy() * (U[:, 0] ** 2 + U[:, 1] ** 2)).sum())
    assert m > 0 and math.isfinite(m)


def test_fem_reference_vanishes_on_the_clamped_boundary():
    r = p4.solve(5, beta=3.0, n_modes=1)
    b = _boundary_points(5).numpy()
    assert float(np.nanmax(np.abs(p4.mode_at(r, 1, b)))) == 0.0


def test_q2_assembly_still_works_after_adding_the_evaluator():
    """평가기를 붙이며 `_q2_shape`를 덮어써 조립이 조용히 깨진 적이 있다 — 회귀로 막는다."""
    mesh = p4.build_mesh(3, beta=2.0)
    K, M = p4.assemble(mesh)
    assert K.shape == M.shape and K.shape[0] == 2 * len(mesh["nodes"])
    assert float(abs(M).sum()) > 0


def test_p4_neural_and_fem_use_the_same_plane_stress_form():
    """FEM과 신경 arm이 같은 구성행렬·전단규약·질량밀도·단위두께를 쓰는가.

    FEM은 εᵀDε(공학전단 γ = u_y + v_x, D = E/(1−ν²)[[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]])를,
    신경 arm은 전개식을 쓴다. 둘이 다르면 같은 물리문제를 푸는 것이 아니므로 P4 비교가
    무의미해진다. 두 형식의 적분항을 무작위 변형률에서 직접 대조한다."""
    import numpy as np

    from eigen_benchmark.neural.p4_neural import _rq_from_state
    from eigen_benchmark.problems.p4_lshape import P4_GEOMETRY, _plane_stress_D

    nu, E, rho = P4_GEOMETRY["nu"], P4_GEOMETRY["E"], P4_GEOMETRY["rho"]
    assert (E, rho) == (1.0, 1.0), "무차원 규약이 E = ρ = 1이다"
    D = _plane_stress_D(E, nu)
    rng = np.random.default_rng(0)
    for _ in range(500):
        ux, uy, vx, vy, u, v = rng.normal(size=6)
        eps = np.array([ux, vy, uy + vx])
        r_fem = float(eps @ D @ eps) / float(rho * (u ** 2 + v ** 2))
        S = torch.tensor([[u], [v], [ux], [uy], [vx], [vy]], dtype=torch.float64)
        r_nn = float(_rq_from_state(S, torch.ones(1, dtype=torch.float64), nu))
        assert abs(r_fem - r_nn) <= 1e-12 * abs(r_fem), (r_fem, r_nn)


def test_exact_projection_coefficients_are_constant_in_space():
    """사영 계수 c는 **구적 적분**이므로 공간미분에 대해 상수여야 한다.

    c가 실수로 공간에 의존하면 (b)의 φ″ 처리가 틀리고 중앙 결과가 무너진다. 확인 세 가지:
    c에 구적축이 없다, 같은 c가 장과 도함수에 함께 적용된다, 그리고 파라미터에 대해서는
    여전히 미분 가능하다."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural.net import EnsembleMLP
    from eigen_benchmark.neural import core

    torch.set_default_dtype(torch.float64)
    n_q = 200
    xs = torch.linspace(1e-3, 1.0, n_q)
    wq = torch.full((n_q,), 1.0 / n_q)
    ens = EnsembleMLP(n_seeds=3, width=8, depth=2, device="cpu")
    phi_all, d2_all = core.eval_phi(ens, xs)
    Phi, Phi2 = phi_all[:2], d2_all[:2]
    phi = phi_all[2].clone().requires_grad_(True)
    d2 = d2_all[2]

    c = df.project_exact(phi, Phi, wq)
    assert c.shape == (2,), f"c에 구적축이 있다: {tuple(c.shape)}"
    ph_t, d2_t, _ = df.ProjectionExact().apply(phi, d2, Phi, Phi2, wq)
    # φ̃″ = φ″ − cᵀΦ″ 가 정확히 성립해야 한다(c가 x에 무관하므로)
    assert torch.equal(d2_t, d2 - c @ Phi2)
    # 저장 span과 질량직교
    leak = torch.einsum("ki,i,i->k", Phi, wq, ph_t).abs().max()
    assert float(leak) < 1e-14, float(leak)
    # 파라미터(장)에 대해서는 미분 가능해야 한다 — detach가 없어야 한다
    g = torch.autograd.grad(c.sum(), phi, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0
