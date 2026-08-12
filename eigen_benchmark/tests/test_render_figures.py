import os

import pytest

pytest.importorskip("matplotlib")
from eigen_benchmark.render import figures      # noqa: E402

DATA = "docs/_generated/data/paper2"


@pytest.mark.parametrize("name", sorted(figures.ALL))
def test_every_figure_writes_a_nonempty_png(tmp_path, name):
    p = figures.ALL[name](DATA, str(tmp_path))
    assert os.path.exists(p) and os.path.getsize(p) > 5000


def test_figures_do_not_depend_on_a_display():
    """헤드리스에서 돌아야 한다 — Agg 백엔드를 import 시점에 강제하는지."""
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def test_figure_labels_are_ascii_so_no_font_is_missing(tmp_path):
    """한글 라벨은 폰트 누락 시 두부글자가 된다 — 소스에 비ASCII 축라벨이 없는지."""
    import re
    src = open(figures.__file__, encoding="utf-8").read()
    for call in re.findall(r'set_(?:xlabel|ylabel|title)\(\s*(r?["\'][^"\']*["\'])',
                           src):
        assert call.isascii(), call
