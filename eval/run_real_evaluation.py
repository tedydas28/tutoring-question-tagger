"""Evaluates the keyword baseline against the REAL hand-transcribed gold sample
(data/real_gold_sample.json) rather than the earlier synthetic set."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import KeywordBaselineClassifier
from src.evaluate import evaluate_model
from src.schema import Question, Taxonomy


def main():
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    raw = json.load(open("data/real_gold_sample.json"))
    questions = [
        Question(id=r["id"], text=r["text"], difficulty=r["difficulty"],
                  gold_skills=r["gold_skills"], gold_formats=r["gold_formats"])
        for r in raw
    ]

    model = KeywordBaselineClassifier(taxonomy)
    skill_result, format_result = evaluate_model(model, questions)

    lines = ["# Evaluation Report: Keyword Baseline on REAL Data\n"]
    lines.append(f"Evaluated on {len(questions)} hand-transcribed real questions "
                 f"(spanning all four uploaded files: Algebra, Advanced Math, "
                 f"Geometry and Trigonometry, Problem-Solving and Data Analysis).\n")

    lines.append("## Skill tagging\n")
    lines.append(f"- Macro F1: {skill_result.macro_f1():.3f}")
    lines.append(f"- Macro Precision: {skill_result.macro_precision():.3f}")
    lines.append(f"- Macro Recall: {skill_result.macro_recall():.3f}\n")
    lines.append("| Skill | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for row in skill_result.summary_table():
        lines.append(f"| {row['label']} | {row['precision']} | {row['recall']} | {row['f1']} | {row['support']} |")
    lines.append("\n### Errors\n")
    for e in skill_result.errors:
        lines.append(f"- **{e['question_id']}**: gold={e['gold']}, predicted={e['predicted']}")

    lines.append("\n## Format tagging\n")
    lines.append(f"- Macro F1: {format_result.macro_f1():.3f}\n")
    lines.append("### Errors\n")
    for e in format_result.errors:
        lines.append(f"- **{e['question_id']}**: gold={e['gold']}, predicted={e['predicted']}")

    report = "\n".join(lines)
    Path("eval/real_data_evaluation_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
