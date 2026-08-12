"""docx → markdown 정본화 (paper3 첨부 원본용).

md2docx.py의 역방향. 스타일 매핑: Heading N → '#'*N, 그 외 문단은 평문.
런 단위 bold/italic → **/_ 보존, 표 → 파이프 표. 본문 순서(문단·표 혼재)를 body 순회로 유지.
"""
import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def run_md(run):
    t = run.text
    if not t.strip():
        return t
    lead = t[:len(t) - len(t.lstrip())]
    trail = t[len(t.rstrip()):]
    core = t.strip()
    if run.bold and '**' not in core:
        core = f'**{core}**'
    if run.italic and '*' not in core:
        core = f'*{core}*'
    return lead + core + trail


def para_md(p):
    text = ''.join(run_md(r) for r in p.runs).strip()
    if not text:
        return None
    style = (p.style.name or '').strip()
    if style.startswith('Heading'):
        try:
            lvl = int(style.split()[-1])
        except ValueError:
            lvl = 2
        return '#' * lvl + ' ' + text
    if style in ('Title',):
        return '# ' + text
    return text


def table_md(tb):
    rows = [[c.text.strip().replace('\n', ' ') for c in r.cells] for r in tb.rows]
    if not rows:
        return None
    out = ['| ' + ' | '.join(rows[0]) + ' |',
           '|' + '|'.join('---' for _ in rows[0]) + '|']
    for r in rows[1:]:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def convert(src, dst):
    doc = Document(src)
    chunks = []
    for item in iter_block_items(doc):
        md = para_md(item) if isinstance(item, Paragraph) else table_md(item)
        if md:
            chunks.append(md)
    with open(dst, 'w') as f:
        f.write('\n\n'.join(chunks) + '\n')
    print(f'{dst}: {len(chunks)} blocks')


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
