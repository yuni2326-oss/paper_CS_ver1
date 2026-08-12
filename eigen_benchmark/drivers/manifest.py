"""출처(provenance) 기록 — 표 하나하나가 어떤 코드·환경에서 나왔는지 남긴다.

§3.7의 "문제당 드라이버 하나가 모든 표를 재생성한다"를 지키려면, 재생성 시점의
git SHA·라이브러리 버전·플랫폼이 결과와 같은 폴더에 있어야 한다.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import subprocess
import sys

DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "docs", "_generated", "data", "paper2")


def _root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", _root(), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _head_sha() -> str:
    """커밋 SHA. **git 실행파일이 없어도** `.git`을 직접 읽어 알아낸다.

    GPU 컨테이너에는 git이 설치돼 있지 않아 `git rev-parse`가 실패했고, 그 결과 컨테이너
    에서 만든 산출물 두 건의 manifest에 `git_sha: unknown`이 남았다. 논문 표의 캡션이
    그 sha를 인용하므로 추적선이 거기서 끊긴다 — 파일을 직접 읽어 메운다."""
    sha = _git("rev-parse", "HEAD")
    if sha != "unknown":
        return sha
    try:
        g = os.path.join(_root(), ".git")
        # **워크트리에서는 `.git`이 디렉터리가 아니라 파일이다** — `gitdir: <경로>` 한 줄이
        # 실제 git 디렉터리를 가리킨다. 그것을 따라가지 않으면 open()이 IsADirectoryError가
        # 아니라 NotADirectoryError로 죽고 sha가 unknown이 된다(캡션의 추적선이 끊긴다).
        if os.path.isfile(g):
            with open(g, encoding="utf-8") as f:
                line = f.read().strip()
            if line.startswith("gitdir:"):
                g = line.split(":", 1)[1].strip()
        with open(os.path.join(g, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            cands = [os.path.join(g, ref)]
            cd = os.path.join(g, "commondir")      # 워크트리 → 공용 git 디렉터리
            if os.path.exists(cd):
                with open(cd, encoding="utf-8") as f:
                    common = os.path.normpath(os.path.join(g, f.read().strip()))
                cands.append(os.path.join(common, ref))
            for fp in cands:
                if os.path.exists(fp):
                    with open(fp, encoding="utf-8") as f:
                        return f.read().strip()
            pr = next((q for q in (os.path.join(os.path.dirname(c), "..",
                                                "packed-refs") for c in cands)
                       if os.path.exists(q)), os.path.join(g, "packed-refs"))
            with open(pr, encoding="utf-8") as f:
                for line in f:
                    if line.rstrip().endswith(" " + ref):
                        return line.split()[0]
            return "unknown"
        return head
    except Exception:
        return "unknown"


def build(extra=None) -> dict:
    import mpmath
    import numpy
    import scipy
    m = {
        "git_sha": _head_sha(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "precision": "fp64 + mpmath dps=50",
    }
    if extra:
        m.update(extra)
    return m


def ensure_outdir(outdir=None) -> str:
    d = os.path.abspath(DEFAULT_OUTDIR if outdir is None else outdir)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def write_jsonl(path: str, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")


def write_csv(path: str, rows, fieldnames=None) -> None:
    if not rows:
        raise ValueError("빈 행 목록은 쓸 수 없습니다")
    if fieldnames is None:
        seen = []
        for r in rows:                       # 행마다 키가 달라도 합집합을 헤더로
            for k in r:
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    else:
        fieldnames = list(fieldnames)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in fieldnames})


def coerce_row(row: dict) -> dict:
    """CSV에서 읽은 행의 값 타입을 복원한다(int → float → bool → str 순 시도).

    `write_csv`로 쓴 결과를 다시 읽어 이어붙일 때 모든 값이 문자열이 되고, 그 상태로
    float와 비교하면 TypeError로 죽는다 — 실제로 장기변주에서 2시간 11분치 계산을
    마지막 기록 단계에서 잃었다. 그래서 읽기 쪽 helper를 write_csv 옆에 둔다."""
    out = {}
    for k, v in row.items():
        if not isinstance(v, str):
            out[k] = v
            continue
        if v in ("True", "False"):
            out[k] = (v == "True")
            continue
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def read_csv(path: str) -> list:
    """write_csv의 역연산 — 타입 복원까지 포함."""
    with open(path, encoding="utf-8") as f:
        return [coerce_row(r) for r in csv.DictReader(f)]
