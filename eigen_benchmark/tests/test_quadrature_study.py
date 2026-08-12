import numpy as np

from eigen_benchmark.bases.monomial import MonomialBasis
from eigen_benchmark.problems import p1_beam as p1
from eigen_benchmark.quadrature_study import doubling_table


def test_doubling_table_shows_quadrature_convergence():
    # n_q는 기저 자유도 이상이어야 질량행렬이 full-rank다(아래 테스트 참조).
    rows = doubling_table(lambda: MonomialBasis(8),
                          lambda b, nq: p1.solve(b, n_q=nq, n_modes=1)["Lam"][0],
                          [9, 10, 20, 40])
    assert [r["n_q"] for r in rows] == [9, 10, 20, 40]
    assert rows[0]["abs_change"] is None
    assert all(np.isfinite(r["value"]) for r in rows)
    # 구적차수를 올리면 값의 변화가 줄어든다(수렴)
    deltas = [r["abs_change"] for r in rows[1:]]
    assert deltas[-1] <= deltas[0]


def test_doubling_table_separates_quadrature_from_basis_error():
    # 같은 기저(8항)를 두 구적차수로 풀면 기저오차는 같고 구적오차만 다르다.
    rows = doubling_table(lambda: MonomialBasis(8),
                          lambda b, nq: p1.solve(b, n_q=nq, n_modes=1)["Lam"][0],
                          [30, 60])
    assert abs(rows[1]["value"] - rows[0]["value"]) < 1e-8


def test_under_integration_below_n_dof_makes_mass_matrix_rank_deficient():
    """Gauss 점수가 기저 자유도보다 적으면 M = Ψ diag(w) Ψᵀ의 rank가 n_q로 제한되어
    구성상 특이해진다 — 구적차수를 기저차수와 독립으로 두되 **하한은 n_dof**라는 조건.
    §5.3에서 구적 부족을 기저 부족으로 오독하지 않도록 이 경계를 명시한다."""
    r = p1.solve(MonomialBasis(8), n_q=5, n_modes=1)
    assert r["cholesky_ok"] is False
    assert np.linalg.matrix_rank(r["M"]) <= 5
