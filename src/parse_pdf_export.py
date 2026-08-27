"""
Parses College Board Question Bank PDF exports (one question per page).

IMPORTANT DATA LIMITATION: College Board renders the actual mathematical
content (equations, variables, numeric values) using vector-typeset glyphs
rather than extractable Unicode text. Plain text extraction reliably
recovers:
  - Question ID
  - Domain / Skill / Difficulty (College Board's own official metadata)
  - English "wrapper" text around a problem (word problem framing, answer
    choice labels where they're not pure math)

It does NOT reliably recover the math itself. A question like "the value
of s?" may have had "s" and an entire equation stripped from what looks
like a complete sentence. Downstream skill-tagging accuracy on
math-heavy stems will be limited until those pages are also processed via
vision-based transcription (see vision_transcribe.py).

This parser still gets you the official Domain/Skill/Difficulty tags for
every question for free, which is a real, usable coarse layer on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import pdfplumber


@dataclass
class RawQuestion:
    id: str
    source_file: str
    source_pdf_filename: str  # just the filename - resolved locally via src/config.py
    page_number: int  # 1-indexed - needed to pull the exact original page later
    domain: str
    skill: str
    difficulty: str
    stem_text_partial: str  # may be missing embedded math - see module docstring
    answer_text_partial: str


def parse_pdf(pdf_path: str, source_file: str) -> list[RawQuestion]:
    """
    Uses pdfplumber for the metadata table (it handles wrapped table cells
    correctly, unlike raw pdftotext which reorders wrapped lines relative
    to sibling columns) and plain text extraction for the stem/answer body.
    """
    import os
    pdf_filename = os.path.basename(pdf_path)

    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            full_text = page.extract_text() or ""
            id_match = re.search(r"Question ID:\s*(\S+)", full_text)
            if not id_match:
                continue
            qid = id_match.group(1)

            domain, skill, difficulty = "", "", ""
            tables = page.extract_tables()
            if tables and len(tables[0]) > 1:
                header, data_row = tables[0][0], tables[0][1]
                row = dict(zip([h.strip() if h else "" for h in header], data_row))
                domain = (row.get("Domain") or "").replace("\n", " ").strip()
                skill = (row.get("Skill") or "").replace("\n", " ").strip()
                difficulty = (row.get("Difficulty") or "").replace("\n", " ").strip()

            stem_match = re.search(r"\bQuestion\s*\n(.*?)(?:\n\s*Answer\b|$)", full_text, re.DOTALL)
            stem_text = re.sub(r"\n{2,}", " ", stem_match.group(1)).strip() if stem_match else ""

            answer_match = re.search(r"\bAnswer\s*\n(.*)", full_text, re.DOTALL)
            answer_text = re.sub(r"\n{2,}", " ", answer_match.group(1)).strip() if answer_match else ""

            if not domain:
                # Rare table-parse miss (~1 in 1000 pages) - fall back to the
                # known domain for this source file rather than dropping it.
                domain = {"algebra": "Algebra", "advmath": "Advanced Math"}.get(source_file, "")

            records.append(RawQuestion(
                id=qid,
                source_file=source_file,
                source_pdf_filename=pdf_filename,
                page_number=page_idx,
                domain=domain,
                skill=skill,
                difficulty=difficulty,
                stem_text_partial=stem_text,
                answer_text_partial=answer_text,
            ))
    return records


def main():
    from pathlib import Path

    sources = [
        ("/mnt/user-data/uploads/sat_algebra_all.pdf", "algebra"),
        ("/mnt/user-data/uploads/sat_advmath_all.pdf", "advmath"),
        ("/mnt/user-data/uploads/sat_geotrig_all.pdf", "geotrig"),
        ("/mnt/user-data/uploads/sat_ps_da_all.pdf", "ps_da"),
    ]

    all_records = []
    for pdf_path, name in sources:
        records = parse_pdf(pdf_path, name)
        print(f"{name}: parsed {len(records)} questions")
        all_records.extend(records)

    out_path = Path("data/raw/parsed_questions.json")
    out_path.write_text(json.dumps([asdict(r) for r in all_records], indent=2))
    print(f"\nTotal: {len(all_records)} questions written to {out_path}")

    # Quick data-quality summary
    by_domain = {}
    for r in all_records:
        by_domain.setdefault(r.domain, 0)
        by_domain[r.domain] += 1
    print("\nBy domain:")
    for d, n in sorted(by_domain.items()):
        print(f"  {d}: {n}")

    short_stems = [r for r in all_records if len(r.stem_text_partial) < 40]
    print(f"\nQuestions with <40 chars of extracted stem text (likely near-total math loss): {len(short_stems)} / {len(all_records)}")


if __name__ == "__main__":
    main()
