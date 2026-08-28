"""
Detects whether a page has a real plotted graph (coordinate axes,
gridlines, plotted lines) by counting vector line segments drawn on the
page - NOT by looking for the word "graph" in extracted text.

Why this works when keyword matching doesn't: a question can have a full
xy-plane graph with two plotted lines and never use the word "graph"
anywhere in its text (e.g. "What system of linear equations is
represented by the lines shown?"). But a real graph always draws a
distinctive number of straight vector line segments for its axes and
gridlines - something keyword matching can't see and this can.

Calibrated against all 1,925 real questions in the question bank:
- 1,712 questions (89%) have exactly 0 vector lines - no graph.
- A small handful (47 questions) have 1-7 lines - noise (e.g. a single
  underline, a fraction bar).
- Everything from ~10 lines up to 120 is a real coordinate-plane graph.

LINE_THRESHOLD=8 sits cleanly in the gap between those two groups.

IMPORTANT for performance: opening a PDF with pdfplumber parses the whole
document. If you're checking many pages from the same file, open it ONCE
with open_pdf() and reuse it - do not call has_plotted_graph() in a loop
with a fresh path each time, or you'll re-parse a 600-page PDF hundreds
of times (this is exactly what made the first version of this script hang).
"""

import pdfplumber

LINE_THRESHOLD = 8


def open_pdf(pdf_path: str):
    """Open once per file, then reuse across multiple page checks."""
    return pdfplumber.open(pdf_path)


def page_has_plotted_graph(pdf, page_number: int) -> bool:
    """pdf is an already-open pdfplumber.PDF object from open_pdf()."""
    page = pdf.pages[page_number - 1]
    return len(page.lines) >= LINE_THRESHOLD


def has_plotted_graph(pdf_path: str, page_number: int) -> bool:
    """Convenience one-off check for a single page. Slow if called
    repeatedly on the same file - use open_pdf() + page_has_plotted_graph()
    instead when checking many pages."""
    with pdfplumber.open(pdf_path) as pdf:
        return page_has_plotted_graph(pdf, page_number)

