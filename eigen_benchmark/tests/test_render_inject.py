import pytest

from eigen_benchmark.render import inject

DOC = "before\n<!-- AUTO:t1 -->\nold\n<!-- /AUTO -->\nafter\n"


def test_inject_replaces_block_content():
    out = inject.inject(DOC, {"t1": "NEW"})
    assert "NEW" in out and "old" not in out
    assert out.startswith("before") and out.rstrip().endswith("after")


def test_inject_is_idempotent():
    once = inject.inject(DOC, {"t1": "NEW"})
    assert inject.inject(once, {"t1": "NEW"}) == once


def test_inject_keeps_the_markers_so_it_can_run_again():
    out = inject.inject(DOC, {"t1": "NEW"})
    assert "<!-- AUTO:t1 -->" in out and "<!-- /AUTO -->" in out


def test_unplaced_block_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="t2"):
        inject.inject(DOC, {"t1": "A", "t2": "B"})


def test_empty_marker_in_document_is_an_error():
    with pytest.raises(ValueError, match="t1"):
        inject.inject(DOC, {})


def test_render_all_produces_every_declared_table_and_figure(tmp_path):
    b = inject.render_all(fig_dir=str(tmp_path))
    for k in inject.TABLES:
        assert k in b and b[k].strip()
    for i in range(1, 7):
        assert f"figure_{i}" in b
    assert "appendix_a" in b and "appendix_b" in b


def test_figure_blocks_link_a_relative_path_that_resolves(tmp_path):
    """논문에서 본 상대경로가 실제 파일을 가리켜야 한다 — docx 변환이 여기서 깨진다."""
    import os
    import re
    fig = tmp_path / "fig"
    b = inject.render_all(fig_dir=str(fig), paper_dir=str(tmp_path))
    src = re.search(r"!\[Figure 1\]\(([^)]+)\)", b["figure_1"]).group(1)
    assert os.path.exists(os.path.join(str(tmp_path), src)), src


def test_every_figure_block_carries_a_caption(tmp_path):
    b = inject.render_all(fig_dir=str(tmp_path))
    for i in range(1, 7):
        assert f"**Figure {i}.**" in b[f"figure_{i}"]
