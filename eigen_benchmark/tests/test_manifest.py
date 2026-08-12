

def test_head_sha_is_found_even_without_the_git_executable(monkeypatch):
    """GPU 컨테이너에는 git이 없어 manifest에 'unknown'이 남았다 — 표 캡션이 그 sha를
    인용하므로 추적선이 끊긴다. .git을 직접 읽는 폴백을 회귀로 지킨다."""
    from eigen_benchmark.drivers import manifest as mf
    monkeypatch.setattr(mf, "_git", lambda *a: "unknown")
    sha = mf._head_sha()
    assert sha != "unknown" and len(sha) == 40 and all(
        c in "0123456789abcdef" for c in sha), sha
