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
        return render_page_image_from_doc(doc, page_number, dpi)
    finally:
        doc.close()


def render_page_image_from_doc(doc, page_number: int, dpi: int = 150) -> bytes:
    """Same as render_page_image, but reuses an already-open pymupdf document -
    use this when rendering several pages from the same file (open once with
    open_pymupdf(), reuse across calls) instead of reopening per page."""
    page = doc[page_number - 1]  # pymupdf is 0-indexed
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return pix.tobytes("png")


def open_pymupdf(pdf_path: str):
    return pymupdf.open(pdf_path)


def extract_single_page_pdf(pdf_path: str, page_number: int) -> bytes:
    """Returns bytes of a standalone one-page PDF, suitable for downloading."""
    reader = PdfReader(pdf_path)
    return extract_page_from_reader(reader, page_number)


def extract_page_from_reader(reader: PdfReader, page_number: int) -> bytes:
    """Same as extract_single_page_pdf, but reuses an already-open PdfReader -
    use this when extracting several pages from the same file."""
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])  # pypdf is also 0-indexed
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def open_pypdf_reader(pdf_path: str) -> PdfReader:
    return PdfReader(pdf_path)


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
