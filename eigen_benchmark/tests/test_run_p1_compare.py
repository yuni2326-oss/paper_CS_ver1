"""고전 vs 신경 통합 대조 — 범위문서 §7이 논문 2로 지정한 항목."""
import math
import os
import shutil

import pytest

from eigen_benchmark.drivers import manifest
from eigen_benchmark.drivers import run_p1_compare as rc

DATA = "docs/_generated/data/paper2"


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    """**출력은 임시 디렉터리로 보낸다.** 이전에는 `outdir=DATA`여서 테스트가 커밋된
    데이터를 덮어썼다 — manifest의 git_sha가 테스트 실행 시점의 HEAD로 바뀌면서 캡션이
    조용히 갱신되고, 커밋마다 `render_paper --check`가 드리프트를 보고했다. 입력은
    같은 실데이터를 읽으므로 검증력은 그대로다."""
    out = str(tmp_path_factory.mktemp("compare"))
    for f in os.listdir(DATA):          # 드라이버가 읽는 입력만 복사한다
        if f.endswith((".csv", ".jsonl", ".json")):
            shutil.copy(os.path.join(DATA, f), out)
    return rc.main(outdir=out)


def test_all_three_families_are_present(res):
    """gradient arm과 랜덤특징을 하나로 묶으면 논문의 핵심 대조가 가려진다 —
    PIELM이 3자리 싸서 'neural' 슬롯을 독차지하고 gradient arm이 표에서 사라진다."""
    assert {r["family"] for r in res["compare"]} == set(rc.FAMILIES)


def test_gradient_arms_carry_an_accuracy_so_they_are_not_silently_dropped(res):
    """기본격자 CSV에 gradient arm의 e_λ가 없어 ε-표에서 조용히 빠지던 결함의 회귀."""
    import math as _m
    grad = [r for r in res["compare"] if r["family"] == "neural_gradient"
            and r["p_correct"] > 0]
    assert grad
    have = [r for r in grad if _m.isfinite(r["e_lam"])]
    assert len(have) == len(grad), [r["solver"] for r in grad
                                   if not _m.isfinite(r["e_lam"])]
    assert any(r["e_lam_source"].startswith("records.jsonl") for r in have)


def test_every_row_declares_its_accounting_basis(res):
    """회계 기준을 행마다 밝히지 않으면 앙상블 시간과 1회 시간이 섞인다 — 감사 지적."""
    for r in res["compare"]:
        assert r["accounting"]
        assert math.isfinite(r["seconds_per_attempt"]) or r["seconds_per_attempt"] is None


def test_neural_per_attempt_is_measured_directly_not_converted(res):
    """환산계수는 같은 기계에서 2.53과 7.82로 나왔다 — 결론에 실을 수 없으므로
    n_seeds=1로 직접 잰 값을 쓴다. 계수가 코드에 남아 있으면 안 된다."""
    assert not hasattr(rc, "BATCH_FACTOR")
    for r in res["compare"]:
        if r["family"] == "neural_gradient":
            assert "n_seeds = 1" in r["accounting"]
            assert r["seconds_per_attempt"] != r["ensemble_seconds"]


def test_shared_gpu_spread_is_reported_for_directly_measured_rows(res):
    """공유 GPU라 벽시계가 흔들린다 — 산포를 숨기지 않는다."""
    grad = [r for r in res["compare"] if r["family"] == "neural_gradient"]
    assert grad
    assert any(r["seconds_per_attempt_min"] not in ("", None) for r in grad)


def test_pielm_is_not_divided_because_it_already_loops_seeds(res):
    for r in res["compare"]:
        if r["solver"].startswith("f_eig_pielm"):
            assert r["seconds_per_attempt"] == pytest.approx(r["ensemble_seconds"])


def test_failures_are_charged_infinite_expected_time(res):
    for r in res["compare"]:
        if r["p_correct"] == 0.0:
            assert r["E_T_success"] == math.inf


def test_pareto_front_is_actually_nondominated(res):
    """Pareto 정의를 코드가 지키는지 — 전선 안에서 서로 지배하는 점이 없어야 한다."""
    for m in (1, 2, 3):
        f = [p for p in res["pareto"] if p["pareto_mode"] == m]
        for a in f:
            for b in f:
                if a is b:
                    continue
                assert not (b["e_lam"] <= a["e_lam"]
                            and b["seconds_per_attempt"] <= a["seconds_per_attempt"]
                            and (b["e_lam"] < a["e_lam"]
                                 or b["seconds_per_attempt"] < a["seconds_per_attempt"]))


def test_cheapest_table_covers_every_level_and_family(res):
    got = {(c["mode"], c["eps"], c["family"]) for c in res["cheapest"]}
    assert got == {(m, e, f) for m in (1, 2, 3) for e in rc.EPS_LEVELS
                   for f in rc.FAMILIES}


def test_unreached_levels_are_marked_not_silently_dropped(res):
    """도달 못한 수준을 행에서 빼면 '전부 도달했다'로 읽힌다 — 명시적으로 남긴다."""
    for c in res["cheapest"]:
        if not c["reached"]:
            assert c["E_T_success"] == math.inf and c["solver"] == ""


def test_classical_rows_charge_basis_construction():
    """비용표의 분모 — 구성이 빠지면 자릿수가 최대 400배 틀린다."""
    rows = manifest.read_csv(f"{DATA}/p1_basis_study.csv")
    assert rows and "seconds_construct" in rows[0] and "seconds_solve" in rows[0]
    for r in rows:
        assert r["seconds"] == pytest.approx(r["seconds_construct"]
                                             + r["seconds_solve"], rel=1e-9)
    orth = [r for r in rows if r["basis"] == "monomial_orthonormalized"]
    raw = [r for r in rows if r["basis"] == "monomial_raw"]
    # 이 수정의 근거: 정규직교화는 구성이 지배하고 원시는 무료에 가깝다
    assert max(r["seconds_construct"] for r in orth) > 10 * max(
        r["seconds_construct"] for r in raw)


def test_warm_cold_train_time_is_linear_in_the_budget():
    """예산과 무관한 상수 훈련시간(200/400/4000이 모두 127 s)이 나오던 파이프라인
    버그의 회귀. scale을 예산마다 계산해 tr = ensemble*scale = single_seed가 됐다."""
    rows = [r for r in manifest.read_csv(f"{DATA}/p1_warm_cold_summary.csv")
            if str(r["arm"]).startswith("b_projection_exact") and r["ladder"] == "I0"]
    by = {int(r["budget_iters"]): float(r["train_seconds_per_attempt"]) for r in rows}
    bs = sorted(by)
    assert len(bs) >= 3
    for a, b in zip(bs, bs[1:]):
        assert by[b] > 1.5 * by[a], (a, by[a], b, by[b])
    # 예산 비와 시간 비가 대략 비례해야 한다
    assert by[bs[-1]] / by[bs[0]] == pytest.approx(bs[-1] / bs[0], rel=0.3)


def test_warm_cold_prefit_grows_with_the_budget_by_design():
    """사전적합은 max(iters//2, 200)이라 예산에 묶여 있다 — 감소하면 버그다."""
    rows = [r for r in manifest.read_csv(f"{DATA}/p1_warm_cold_summary.csv")
            if str(r["arm"]).startswith("b_projection_exact") and r["ladder"] == "I5"]
    by = {int(r["budget_iters"]): float(r["init_seconds_per_attempt"]) for r in rows}
    bs = sorted(by)
    assert by[bs[-1]] > by[bs[0]], by


def test_cheapest_budget_is_the_smallest_one_where_cold_still_fails():
    """이 절의 실험 논리 — 예산을 줄여 축을 분리한다. E[T] 최소가 최대 예산으로
    옮겨가면 논리가 무너진다(버그 시 실제로 그랬다)."""
    rows = [r for r in manifest.read_csv(f"{DATA}/p1_warm_cold_summary.csv")
            if str(r["arm"]).startswith("b_projection_exact")]
    best = min(rows, key=lambda r: float(r["E_T_success"]))
    assert int(best["budget_iters"]) == min(int(r["budget_iters"]) for r in rows)


def test_p4_neural_cost_uses_the_direct_single_seed_measurement():
    """앙상블 벽시계를 1회 비용처럼 쓰면 격차가 과대계상된다 — P1과 같은 회계여야 한다."""
    rows = [r for r in manifest.read_csv(f"{DATA}/p4_classical_vs_neural.csv")
            if r["family"] == "neural_gradient"]
    assert rows
    for r in rows:
        assert not bool(r["per_attempt_is_upper_bound"]), r["mode"]
        assert "n_seeds = 1" in str(r["accounting"])
        # 1회 비용은 앙상블보다 반드시 작다
        assert float(r["seconds_per_attempt"]) < float(r["ensemble_seconds"])


def test_every_headline_row_declares_whether_it_is_measured_or_bounded():
    rows = manifest.read_csv(f"{DATA}/p1_cost_headline.csv")
    assert rows
    for r in rows:
        assert str(r["bound"]) in ("measured", "upper", "lower"), r["bound"]
    # 직접 측정이 들어온 뒤로는 경계 행이 없어야 한다
    assert all(str(r["bound"]) == "measured" for r in rows)
