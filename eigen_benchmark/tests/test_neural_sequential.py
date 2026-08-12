import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="GPU 없으면 시간이 과다")

from eigen_benchmark.neural import deflation as df      # noqa: E402
from eigen_benchmark.neural import sequential as sq     # noqa: E402
from eigen_benchmark.problems import p1_beam as p1      # noqa: E402


def test_first_mode_is_recovered_from_cold_start():
    out = sq.solve_sequential(1, df.NoDeflation(), n_seeds=8, iters=1500)
    lam = out["lam"][0]
    b1 = p1.beta_roots(1)[0]
    assert np.median(np.abs(lam - b1) / b1) < 1e-3


def test_exact_projection_reaches_mode_three_from_cold_start():
    """스파이크의 핵심 결과를 회귀로 고정 — 고차모드 실패는 spectral bias가 아니다."""
    out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=8, iters=4000)
    b3 = p1.beta_roots(3)[2]
    lam3 = out["lam"][2]
    assert np.median(np.abs(lam3 - b3) / b3) < 1e-3
    assert int((np.abs(lam3 - b3) / b3 < 0.02).sum()) >= 7


def test_pure_penalty_collapses_to_the_lowest_mode():
    out = sq.solve_sequential(3, df.Penalty(10.0), n_seeds=8, iters=4000)
    b1, b3 = p1.beta_roots(3)[0], p1.beta_roots(3)[2]
    lam3 = out["lam"][2]
    assert int((np.abs(lam3 - b1) / b1 < 0.02).sum()) >= 6      # 모드1로 붕괴
    assert int((np.abs(lam3 - b3) / b3 < 0.02).sum()) == 0


def test_classification_matches_the_preregistered_rule():
    out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=8, iters=4000)
    recs = sq.classify_stage(out["shapes"][2], out["lam"][2], target=3,
                             xs=out["xs"], wq=out["wq"],
                             converged=out["converged"][2])
    assert len(recs) == 8
    assert sum(r["outcome"] == "correct" for r in recs) >= 7
    assert all(set(r) >= {"outcome", "matched_mode", "mac", "e_lam"} for r in recs)


def test_history_supports_time_to_accuracy():
    out = sq.solve_sequential(1, df.NoDeflation(), n_seeds=4, iters=500,
                              snapshot_every=50)
    hist = out["history"][0]
    assert len(hist) >= 5
    assert hist[-1][1] > hist[0][1]        # 시간 단조 증가


def test_returned_shapes_are_the_projected_functions():
    """사영 arm의 평가는 사영된 φ̃로 해야 한다 — raw φ로 재면 하위모드 성분이 남아
    λ가 β₂⁴ 아래로 내려가는 물리적으로 불가능한 값이 나온다(스파이크에서 실제 발생)."""
    out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=6, iters=3000)
    b2 = p1.beta_roots(3)[1]
    assert np.median(out["lam"][2]) > b2      # 3번째 스테이지는 β₂를 넘어야 한다
