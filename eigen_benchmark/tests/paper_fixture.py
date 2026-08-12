"""원고 파일을 읽는 테스트의 공용 진입점.

이 리포는 두 형태로 배포된다 — 원고를 포함한 사내 형태와, **계산만** 담은 공개 형태다
(코드·데이터·테스트는 같고 `docs/paper2-cs/`가 없다). 원고-데이터 일관성 검사는 원고가
있을 때만 의미가 있으므로, 없으면 실패가 아니라 **건너뛴다.** 조용히 통과시키지는 않는다 —
skip 이유에 어느 배포인지 적어서 "검사를 지웠다"와 구분된다.
"""
from __future__ import annotations

import os

import pytest

PAPER = os.path.join("docs", "paper2-cs", "paperA_CS_benchmark.md")

SKIP_REASON = ("manuscript not present in this checkout — compute-only "
               "distribution; manuscript-data consistency is checked where the "
               "manuscript lives")


def has_paper() -> bool:
    return os.path.exists(PAPER)


def read() -> str:
    """원고 전문. 없으면 그 테스트를 건너뛴다."""
    if not has_paper():
        pytest.skip(SKIP_REASON)
    with open(PAPER, encoding="utf-8") as f:
        return f.read()


def flat() -> str:
    """산문 검사용 — 줄바꿈이 문장을 쪼개므로 공백을 정규화한다."""
    return " ".join(read().split())
