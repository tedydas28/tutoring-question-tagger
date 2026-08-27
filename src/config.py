"""
Where your original source PDFs live locally. The rest of the project only
stores each question's FILENAME (not a full path), so this is the one place
that needs to match your machine.

Put your four PDFs in this folder (same filenames as when you uploaded
them), or change SOURCE_PDF_DIR to wherever you keep them.
"""

from pathlib import Path

SOURCE_PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "source_pdfs"


def resolve_pdf_path(filename: str) -> str:
    return str(SOURCE_PDF_DIR / filename)
