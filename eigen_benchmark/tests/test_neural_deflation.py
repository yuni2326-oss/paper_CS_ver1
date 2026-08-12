import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import deflation as df    # noqa: E402


def _w(n=200):
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor(0.5 * x + 0.5), torch.tensor(0.5 * w))


def test_exact_projection_makes_the_result_orthogonal():
    xs, wq = _w()
    Phi = torch.stack([torch.sin(np.pi * xs), torch.sin(2 * np.pi * xs)])
    phi = 3.0 * Phi[0] - 2.0 * Phi[1] + torch.sin(5 * np.pi * xs)
    c = df.project_exact(phi, Phi, wq)
    resid = phi - c @ Phi
    assert torch.allclose((Phi * wq) @ resid, torch.zeros(2, dtype=torch.float64),
                          atol=1e-12)


def test_diagonal_projection_is_exact_only_for_orthogonal_previous_modes():
    """대각근사는 하위모드가 서로 직교할 때만 정확하다. 비직교면 잔차가 남는다 —
    파일럿 구현이 모드3에서 실패한 유력 원인."""
    xs, wq = _w()
    ortho = torch.stack([torch.sin(np.pi * xs), torch.sin(2 * np.pi * xs)])
    phi = 3.0 * ortho[0] - 2.0 * ortho[1] + torch.sin(5 * np.pi * xs)
    c_d = df.project_diagonal(phi, ortho, wq)
    c_e = df.project_exact(phi, ortho, wq)
    assert torch.allclose(c_d, c_e, atol=1e-10)

    skew = torch.stack([ortho[0], 0.7 * ortho[0] + 0.3 * ortho[1]])
    r_d = phi - df.project_diagonal(phi, skew, wq) @ skew
    r_e = phi - df.project_exact(phi, skew, wq) @ skew
    leak_d = float(torch.abs((skew * wq) @ r_d).max())
    leak_e = float(torch.abs((skew * wq) @ r_e).max())
    assert leak_e < 1e-12
    assert leak_d > 1e-3


def test_penalty_returns_unmodified_functions_and_a_positive_term():
    xs, wq = _w()
    Phi = torch.stack([torch.sin(np.pi * xs)])
    phi = Phi[0] + 0.1 * torch.sin(4 * np.pi * xs)
    d2 = torch.zeros_like(phi)
    p, q, pen = df.Penalty(10.0).apply(phi, d2, Phi, Phi, wq)
    assert torch.allclose(p, phi)          # 페널티는 함수를 바꾸지 않는다
    assert float(pen) > 0.0


def test_penalty_vanishes_for_an_orthogonal_candidate():
    xs, wq = _w()
    Phi = torch.stack([torch.sin(np.pi * xs)])
    phi = torch.sin(2 * np.pi * xs)
    _, _, pen = df.Penalty(10.0).apply(phi, torch.zeros_like(phi), Phi, Phi, wq)
    assert float(pen) < 1e-20


def test_no_deflation_is_identity():
    xs, wq = _w()
    phi = torch.sin(np.pi * xs)
    d2 = torch.cos(np.pi * xs)
    p, q, pen = df.NoDeflation().apply(phi, d2, None, None, wq)
    assert torch.equal(p, phi) and torch.equal(q, d2) and float(pen) == 0.0


def test_exact_projection_also_deflates_the_second_derivative():
    """평가·RQ 모두 사영된 φ̃와 φ̃″를 써야 한다. φ″를 사영하지 않으면 RQ 분자가
    하위모드 곡률을 그대로 품어 값이 무의미해진다."""
    xs, wq = _w()
    Phi = torch.stack([torch.sin(np.pi * xs)])
    Phi2 = torch.stack([-(np.pi ** 2) * torch.sin(np.pi * xs)])
    phi = Phi[0] + torch.sin(3 * np.pi * xs)
    d2 = Phi2[0] - 9 * (np.pi ** 2) * torch.sin(3 * np.pi * xs)
    p, q, _ = df.ProjectionExact().apply(phi, d2, Phi, Phi2, wq)
    assert torch.allclose((Phi * wq) @ p, torch.zeros(1, dtype=torch.float64),
                          atol=1e-12)
    assert not torch.allclose(q, d2)


def test_every_strategy_exposes_a_name_for_the_manifest():
    for s in (df.NoDeflation(), df.Penalty(1.0), df.ProjectionExact(),
              df.ProjectionDiagonal()):
        assert isinstance(s.name, str) and s.name


def test_one_stored_mode_makes_the_two_projections_algebraically_identical():
    """Table 2의 "k = 2에서 두 arm이 구성상 동일하다"를 고정한다.

    Φ가 한 열이면 그람은 1×1이고 대각 = 전체이므로 (a′)의 근사가 근사가 아니게 된다.
    그래서 모드 2(=1개 저장) 풀이는 두 arm에서 같은 궤적을 밟고, k = 2에 저장된
    기저도 같은 행렬이 된다 — 진단값이 자릿수까지 일치하는 이유. 두 formula가
    실제로 갈리려면 비대각이 있어야 하고, 그건 모드 3(=2×2 그람)부터다.
    """
    torch.manual_seed(0)
    nq = 64
    wq = torch.rand(nq, dtype=torch.float64) + 0.5
    phi = torch.randn(nq, dtype=torch.float64)
    Phi1 = torch.randn(1, nq, dtype=torch.float64)          # 저장 모드 1개
    c_e = df.project_exact(phi, Phi1, wq)
    c_d = df.project_diagonal(phi, Phi1, wq)
    assert torch.allclose(c_e, c_d, rtol=0, atol=1e-14), (c_e, c_d)

    Phi2 = torch.randn(2, nq, dtype=torch.float64)           # 2개면 갈린다
    g = torch.einsum("ki,i,li->kl", Phi2, wq, Phi2)
    assert abs(g[0, 1]) > 1e-3, "비대각이 0이면 이 검사가 무의미하다"
    assert not torch.allclose(df.project_exact(phi, Phi2, wq),
                              df.project_diagonal(phi, Phi2, wq),
                              rtol=1e-6, atol=1e-9)


def test_spacetime_loss_actually_contains_the_free_end_residual():
    """arm (g)가 P1과 **같은** 물리문제를 푸는지가 negative result의 전제다.

    시행함수 w = x²·N(x,t)는 clamped 단만 하드로 만족하고, 자유단
    w_xx(1,t) = w_xxx(1,t) = 0은 강형식에서 자동으로 성립하지 않는다. 그 잔차가 손실에
    없으면 이 arm은 다른 경계조건 문제를 푸는 것이 되고 §5.5의 해석이 무너진다. 그래서
    손실이 실제로 자유단 항에 반응하는지를 **기울기로** 확인한다 — 소스 문자열 검사가
    아니라 동작 검사다.
    """
    from eigen_benchmark.neural import spacetime as st

    torch.manual_seed(0)
    net = st.SpaceTimeNet(width=8, depth=2).double()
    xb = torch.ones(16, 1, dtype=torch.float64, requires_grad=True)
    tb = torch.rand(16, 1, dtype=torch.float64, requires_grad=True)
    wb = net(xb, tb)
    free = (st._d(wb, xb, 2) ** 2).mean() + (st._d(wb, xb, 3) ** 2).mean()
    assert float(free) > 0, "초기 망에서 자유단 잔차가 0이면 검사가 무의미하다"
    gr = torch.autograd.grad(free, list(net.parameters()), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in gr), \
        "자유단 항이 파라미터에 기울기를 주지 않는다"

    # clamped 단은 하드로 정확히 만족해야 한다
    x0 = torch.zeros(8, 1, dtype=torch.float64, requires_grad=True)
    t0 = torch.rand(8, 1, dtype=torch.float64)
    w0 = net(x0, t0)
    assert float(w0.abs().max()) == 0.0
    assert float(st._d(w0, x0, 1).abs().max()) == 0.0


def test_spacetime_disclosure_matches_the_code_defaults():
    """부록 B의 공시값이 `solve_spacetime`의 기본 인자와 같아야 한다."""
    import inspect

    from eigen_benchmark.neural import spacetime as st
    from eigen_benchmark.render import tables

    sig = inspect.signature(st.solve_spacetime)
    row = [l for l in tables.appendix_b_disclosure().splitlines()
           if "space–time PINN" in l][0].replace("\\|", "|")
    for k, txt in (("n_bc", "{} times drawn once from U(0,T)"),
                   ("n_ic", "{} uniform x"),
                   ("n_col", "{} random collocation points per iteration"),
                   ("n_probe", "{} uniform times over [0,T]")):
        v = int(sig.parameters[k].default)
        assert txt.format(v) in row, (k, v)
    assert f"+ {sig.parameters['w_ic'].default:g} · initial-state residual" in row
    assert f"+ {sig.parameters['w_bc'].default:g} · free-end residual" in row
    assert f"Δt = T/{int(sig.parameters['n_probe'].default)};" in row
    assert "the window endpoint is excluded so the record length is exactly T" in row
    assert "no taper or window" in row
    assert "excluding DC with no interpolation" in row


def test_spacetime_estimator_is_unbiased_on_the_exact_mode():
    """추정기를 **해석해**에 넣으면 오차가 0이어야 한다 — 아니면 그 칸은 솔버가 아니라
    추정기를 재는 것이다.

    끝점을 포함해 표본을 뽑으면 dt = T/(n−1)이므로 FFT가 보는 기록길이가
    n·dt = T·n/(n−1)이 되어 주파수 격자가 어긋나고, 정수 주기 창이라도 참조 주파수가 빈에
    떨어지지 않는다(4주기·512표본에서 빈 4.0078). 그 결과 창 길이와 무관한 고정 편향
    3.902e-03이 생겼고, 그것이 4주기에서 보고된 e_λ와 자릿수까지 같았다. 끝점을 빼면
    기록길이가 정확히 T가 되어 편향이 사라진다.
    """
    import numpy as np

    from eigen_benchmark.neural import spacetime as st
    from eigen_benchmark.problems.p1_beam import beta_roots

    w1 = float(beta_roots(1)[0]) ** 2
    n = 512
    for periods in (4.0, 8.0, 16.0, 32.0):
        T = st.bin_spacing(periods)["T_nondim"]
        t = np.arange(n) * (T / n)                    # 끝점 제외 = 기록길이 T
        om, dom = st._peak_frequency(np.cos(w1 * t), T / n)
        assert abs(om ** 2 - w1 ** 2) / w1 ** 2 == 0.0, (periods, om)
        # 참조 주파수가 빈 `periods`에 정확히 떨어져야 한다
        assert abs(w1 / dom - periods) < 1e-9, (periods, w1 / dom)
    # 옛 방식(끝점 포함)은 창과 무관한 고정 편향을 낸다 — 회귀 방지
    T = st.bin_spacing(4.0)["T_nondim"]
    t_bad = np.linspace(0.0, T, n)
    om_bad, _ = st._peak_frequency(np.cos(w1 * t_bad), T / (n - 1))
    assert abs(om_bad ** 2 - w1 ** 2) / w1 ** 2 > 1e-3
