"""
Extracts a single page from the original source PDF - either as a
rendered image (for on-screen preview) or as a standalone one-page PDF
(for downloading/printing to give directly to a student).

Deliberately does NOT touch the extracted-text pipeline. The whole point
is that the original page is the ground truth: it always has the exact
equations, figures, and formatting, regardless of what did or didn't
extract as searchable text.
"""

from __future__ import annotations

import io

import pymupdf
from pypdf import PdfReader, PdfWriter


def render_page_image(pdf_path: str, page_number: int, dpi: int = 150) -> bytes:
    """Returns PNG bytes for the given 1-indexed page number."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_number - 1]  # pymupdf is 0-indexed
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


def extract_single_page_pdf(pdf_path: str, page_number: int) -> bytes:
    """Returns bytes of a standalone one-page PDF, suitable for downloading."""
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])  # pypdf is also 0-indexed
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def extract_multiple_pages_pdf(pages: list[tuple[str, int]]) -> bytes:
    """
    Combines multiple (pdf_path, page_number) pairs into a single
    downloadable PDF - e.g. a worksheet built from several matched questions
    across different source files.
    """
    writer = PdfWriter()
    readers: dict[str, PdfReader] = {}
    for pdf_path, page_number in pages:
        if pdf_path not in readers:
            readers[pdf_path] = PdfReader(pdf_path)
        writer.add_page(readers[pdf_path].pages[page_number - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
