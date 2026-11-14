# Path: projects/document-parsing/evals/make_fixtures.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Generate deterministic synthetic PDF pages for the evaluation suite.

The generator is the fixture: synthetic PDFs are never committed. Every
scenario accepts a ``variant`` so the calibration and test pages exercise the
same layout pattern without being byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pymupdf

A4 = pymupdf.paper_rect("a4")
MARGIN = 56.0
TEXT_SIZE = 10.0

_SENTENCES: tuple[str, ...] = (
    "A page classifier separates textual content from visual content before extraction.",
    "Retrieval quality depends on how cleanly prose is separated from figures and tables.",
    "Layout features describe where text sits on the page rather than what it means.",
    "Occupancy measures how much of a fixed grid is covered by text blocks.",
    "Edge density records how much high-frequency structure a rendered page contains.",
    "The same geometry can describe a dense paragraph or a table of short cells.",
    "Calibration pages must stay separate from the pages used to report metrics.",
    "A confidence value is only meaningful if it was measured on held-out input.",
    "Synthetic fixtures make the evaluation deterministic and cheap to reproduce.",
    "Curated fixtures show how the method behaves on intentionally designed pages.",
    "Misclassified pages are more informative than a single aggregate score.",
    "A binary label becomes ambiguous when prose and a schematic share the page.",
    "The rubric assigns the label of the artifact that carries the page's content.",
    "Small test sets require intervals rather than bare point estimates.",
    "Every decision threshold should be traceable to a stated assumption.",
    "A pipeline that cannot explain a decision is difficult to debug in production.",
)

Scenario = Callable[[pymupdf.Page, int], None]


def _prose(variant: int, sentence_count: int) -> str:
    """Return a deterministic paragraph whose content depends on the variant."""
    start = variant % len(_SENTENCES)
    return " ".join(
        _SENTENCES[(start + index) % len(_SENTENCES)] for index in range(sentence_count)
    )


def _dense_prose(page: pymupdf.Page, variant: int) -> None:
    rect = pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, A4.height - MARGIN)
    page.insert_textbox(
        rect,
        _prose(variant, sentence_count=24),
        fontsize=TEXT_SIZE,
        align=pymupdf.TEXT_ALIGN_JUSTIFY,
    )


def _two_columns(page: pymupdf.Page, variant: int) -> None:
    gutter = 28.0
    middle = A4.width / 2.0
    left = pymupdf.Rect(MARGIN, MARGIN, middle - gutter / 2.0, A4.height - MARGIN)
    right = pymupdf.Rect(middle + gutter / 2.0, MARGIN, A4.width - MARGIN, A4.height - MARGIN)
    page.insert_textbox(
        left,
        _prose(variant, sentence_count=13),
        fontsize=TEXT_SIZE,
        align=pymupdf.TEXT_ALIGN_JUSTIFY,
    )
    page.insert_textbox(
        right,
        _prose(variant + 3, sentence_count=13),
        fontsize=TEXT_SIZE,
        align=pymupdf.TEXT_ALIGN_JUSTIFY,
    )


def _text_table(page: pymupdf.Page, variant: int) -> None:
    columns = 4
    rows = 6
    top = MARGIN + 24.0
    table = pymupdf.Rect(MARGIN, top, A4.width - MARGIN, top + 210.0)
    page.insert_text(
        pymupdf.Point(MARGIN, MARGIN + 8.0),
        f"Synthetic comparison matrix {variant}",
        fontsize=12.0,
    )
    page.draw_rect(table, color=(0, 0, 0), width=0.8)
    column_width = table.width / columns
    row_height = table.height / rows
    for row in range(rows + 1):
        y = table.y0 + row * row_height
        page.draw_line(
            pymupdf.Point(table.x0, y), pymupdf.Point(table.x1, y), color=(0, 0, 0), width=0.6
        )
    for column in range(columns + 1):
        x = table.x0 + column * column_width
        page.draw_line(
            pymupdf.Point(x, table.y0), pymupdf.Point(x, table.y1), color=(0, 0, 0), width=0.6
        )
    headers = ("Component", "Strength", "Weakness", "Metric")
    for column, header in enumerate(headers):
        page.insert_text(
            pymupdf.Point(table.x0 + column * column_width + 4.0, table.y0 + 12.0),
            header,
            fontsize=7.5,
        )
    for row in range(1, rows):
        for column, cell in enumerate(
            ("Retrieval", "Recall at k", "Cost per query", "0.71"),
            start=0,
        ):
            page.insert_text(
                pymupdf.Point(
                    table.x0 + column * column_width + 4.0,
                    table.y0 + row * row_height + 12.0,
                ),
                f"{cell} {row}",
                fontsize=7.5,
            )


def _notes(page: pymupdf.Page, variant: int) -> None:
    lines = [
        f"Note {index + 1}: keep calibration pages separate from evaluation pages."
        if index % 2 == 0
        else f"Note {index + 1}: record every threshold and the reason behind it."
        for index in range(16)
    ]
    body = "\n".join(f"- {line}" for line in lines)
    rect = pymupdf.Rect(MARGIN, MARGIN + 20.0, A4.width - MARGIN, A4.height - MARGIN)
    page.insert_text(pymupdf.Point(MARGIN, MARGIN + 8.0), f"Checklist {variant}", fontsize=12.0)
    page.insert_textbox(rect, body, fontsize=9.5, lineheight=1.4)


def _figure_grid(page: pymupdf.Page, variant: int) -> None:
    offset = 8.0 * variant
    for row in range(3):
        for column in range(4):
            x0 = MARGIN + column * 120.0 + offset
            y0 = MARGIN + row * 150.0 + offset
            rect = pymupdf.Rect(x0, y0, x0 + 90.0, y0 + 110.0)
            page.draw_rect(rect, color=(0, 0, 0), width=1.2)
            page.draw_line(rect.tl, rect.br, color=(0, 0, 0), width=0.8)
            page.draw_line(rect.bl, rect.tr, color=(0, 0, 0), width=0.8)
            page.draw_circle(pymupdf.Point(rect.x0 + 45.0, rect.y0 + 55.0), 18.0, color=(0, 0, 0))


def _flow_diagram(page: pymupdf.Page, variant: int) -> None:
    labels = ("Documents", "Chunk", "Index", "Retrieve")
    y = MARGIN + 90.0 + 10.0 * variant
    width = 110.0
    gap = 28.0
    for index, label in enumerate(labels):
        x0 = MARGIN + index * (width + gap)
        box = pymupdf.Rect(x0, y, x0 + width, y + 60.0)
        page.draw_rect(box, color=(0, 0, 0), width=1.2)
        page.insert_textbox(box, label, fontsize=9.0, align=pymupdf.TEXT_ALIGN_CENTER)
        if index < len(labels) - 1:
            page.draw_line(
                pymupdf.Point(box.x1, y + 30.0),
                pymupdf.Point(box.x1 + gap, y + 30.0),
                color=(0, 0, 0),
                width=1.2,
            )


def _line_art(page: pymupdf.Page, variant: int) -> None:
    shift = 6.0 * variant
    for index in range(14):
        y = MARGIN + index * 48.0 + shift
        page.draw_line(
            pymupdf.Point(MARGIN, y),
            pymupdf.Point(A4.width - MARGIN, y + 24.0),
            color=(0, 0, 0),
            width=0.8,
        )
    for index in range(5):
        page.draw_circle(
            pymupdf.Point(MARGIN + 70.0 + index * 95.0, A4.height / 2.0),
            34.0 + 2.0 * variant,
            color=(0, 0, 0),
        )


def _caption_page(page: pymupdf.Page, variant: int) -> None:
    figure = pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, A4.height - 180.0)
    page.draw_rect(figure, color=(0, 0, 0), width=1.2)
    for index in range(6):
        y = figure.y0 + 60.0 + index * 55.0
        page.draw_line(
            pymupdf.Point(figure.x0 + 40.0, y),
            pymupdf.Point(figure.x1 - 40.0, y - 18.0),
            color=(0, 0, 0),
            width=1.0,
        )
    page.insert_text(
        pymupdf.Point(MARGIN, A4.height - 120.0 + variant),
        "Figure 1. Synthetic schematic used to test diagram detection.",
        fontsize=9.0,
    )


def _labeled_figure(page: pymupdf.Page, variant: int) -> None:
    labels = ("Input", "Parse", "Embed", "Index", "Search", "Rerank", "Answer")
    for index, label in enumerate(labels):
        x0 = MARGIN + (index % 4) * 125.0
        y0 = MARGIN + (index // 4) * 130.0 + 12.0 * variant
        box = pymupdf.Rect(x0, y0, x0 + 100.0, y0 + 70.0)
        page.draw_rect(box, color=(0, 0, 0), width=1.0)
        page.insert_textbox(box, label, fontsize=8.0, align=pymupdf.TEXT_ALIGN_CENTER)


def _blank_page(page: pymupdf.Page, variant: int) -> None:
    del page, variant


def _title_page(page: pymupdf.Page, variant: int) -> None:
    page.insert_text(
        pymupdf.Point(MARGIN, 220.0 + variant),
        "Evaluation Fixture",
        fontsize=24.0,
    )
    page.insert_text(
        pymupdf.Point(MARGIN, 260.0 + variant),
        "Minimal title page",
        fontsize=12.0,
    )


def _scanned_prose(page: pymupdf.Page, variant: int) -> None:
    """Render a text page to an image so the PDF has no text layer."""
    source = pymupdf.open()
    try:
        source_page = source.new_page(width=A4.width, height=A4.height)
        _dense_prose(source_page, variant)
        pixmap = source_page.get_pixmap(matrix=pymupdf.Matrix(150 / 72.0, 150 / 72.0))
        png = pixmap.tobytes("png")
    finally:
        source.close()
    page.insert_image(page.rect, stream=png)


_SCENARIOS: dict[str, Scenario] = {
    "dense_prose": _dense_prose,
    "two_columns": _two_columns,
    "text_table": _text_table,
    "notes": _notes,
    "figure_grid": _figure_grid,
    "flow_diagram": _flow_diagram,
    "line_art": _line_art,
    "caption_page": _caption_page,
    "labeled_figure": _labeled_figure,
    "blank_page": _blank_page,
    "title_page": _title_page,
    "scanned_prose": _scanned_prose,
}


def build_fixtures(
    output_dir: Path,
    scenarios: Iterable[tuple[str, int]],
) -> dict[tuple[str, int], Path]:
    """Generate the requested ``(scenario, variant)`` pages and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    built: dict[tuple[str, int], Path] = {}
    for name, variant in sorted(set(scenarios)):
        if name not in _SCENARIOS:
            raise KeyError(f"unknown synthetic scenario: {name}")
        path = output_dir / f"{name}-{variant}.pdf"
        document = pymupdf.open()
        try:
            page = document.new_page(width=A4.width, height=A4.height)
            _SCENARIOS[name](page, variant)
            document.save(path)
        finally:
            document.close()
        built[(name, variant)] = path
    return built
