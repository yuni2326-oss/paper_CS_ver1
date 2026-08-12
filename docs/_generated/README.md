# docs/_generated — generated artifacts and their generators

Committed here: the benchmark data (`data/paper2/`, one JSON manifest per CSV) and the
scripts that turn it into figures and documents.

Not committed: PNG, PDF and DOCX. They are regenerable binaries, so a fresh clone must run
the generators once — see the repository README. The generators fail loudly and name the
command to run rather than skipping silently.

| script | output |
|---|---|
| `make_fig1_benchmark_problems.py` | Figure 1, the four test problems with a computed mode shape on each. Solves the governing equations directly; the figure is illustrative and no quantitative claim is read from it |
| `coverage_map_paper2.py` | coverage map: which solver arm ran on which problem, at how many seeds, and which axes were swept. Read from the committed CSVs |
| `md2docx.py` | markdown → DOCX, with repeated table header rows and rows that do not split across pages |
| `docx2md.py` | the reverse direction, used when a revision arrives as DOCX |

Paths are hardcoded relative to the repository root, so the generators must be run from
there with `PYTHONPATH=$PWD`.
