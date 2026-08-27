"""
Runs the keyword baseline classifier over all 1,925 real parsed questions
and saves the result to data/tagged_real_questions.json.

Every question keeps its official College Board Domain/Skill/Difficulty
(100% reliable) plus whatever fine-grained skill/format tags the keyword
baseline could detect from the available text (partial - see README for
the math-extraction limitation).

Run this once after re-parsing PDFs or editing the taxonomy, then use
search_cli.py against its output.
"""

import json
from pathlib import Path

from src.classifier import KeywordBaselineClassifier
from src.schema import Question, Taxonomy


def main():
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    raw = json.load(open("data/raw/parsed_questions.json"))

    questions = [
        Question(
            id=r["id"],
            text=f"{r['stem_text_partial']} {r['answer_text_partial']}".strip(),
            difficulty=(r["difficulty"] or "").lower(),
        )
        for r in raw
    ]

    model = KeywordBaselineClassifier(taxonomy)
    tagged = []
    for r, q in zip(raw, questions):
        pred = model.predict(q)
        tagged.append({
            "id": r["id"],
            "source_file": r["source_file"],
            "source_pdf_filename": r["source_pdf_filename"],
            "page_number": r["page_number"],
            "domain": r["domain"],
            "cb_skill": r["skill"],
            "difficulty": r["difficulty"],
            "stem_text_partial": r["stem_text_partial"],
            "answer_text_partial": r["answer_text_partial"],
            "auto_tags_skills": pred.skills,
            "auto_tags_formats": pred.formats,
        })

    out_path = Path("data/tagged_real_questions.json")
    out_path.write_text(json.dumps(tagged, indent=2))
    print(f"Tagged {len(tagged)} questions -> {out_path}")

    n_with_fine_tags = sum(1 for t in tagged if t["auto_tags_skills"])
    print(f"{n_with_fine_tags} / {len(tagged)} got at least one fine-grained skill tag "
          f"from keyword matching ({n_with_fine_tags/len(tagged):.1%})")
    print("(Everything has the official CB domain/skill/difficulty regardless - "
          "the fine-grained tag rate is lower because it depends on text that's "
          "often missing the actual math. See README.)")


if __name__ == "__main__":
    main()
