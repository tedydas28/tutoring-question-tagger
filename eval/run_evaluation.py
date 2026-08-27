"""
Runs the keyword baseline over the full sample question set and writes a
real evaluation report (not a mock) - this is what you'd re-run after
labeling the real question bank and swapping in the LLM classifier for
comparison via compare_models().
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import KeywordBaselineClassifier
from src.evaluate import evaluate_model
from src.schema import Question, Taxonomy


def load_questions(path: str) -> list[Question]:
    with open(path) as f:
        raw = json.load(f)
    return [
        Question(
            id=r["id"], text=r["text"], choices=r["choices"], answer=r["answer"],
            difficulty=r["difficulty"], gold_skills=r["gold_skills"], gold_formats=r["gold_formats"],
        )
        for r in raw
    ]


def main():
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    questions = load_questions("data/sample_questions.json")
    model = KeywordBaselineClassifier(taxonomy)

    skill_result, format_result = evaluate_model(model, questions)

    lines = []
    lines.append("# Evaluation Report: Keyword Baseline Classifier\n")
    lines.append(f"Evaluated on {len(questions)} hand-labeled sample questions.\n")

    lines.append("## Skill tagging performance\n")
    lines.append(f"- Macro F1: {skill_result.macro_f1():.3f}")
    lines.append(f"- Macro Precision: {skill_result.macro_precision():.3f}")
    lines.append(f"- Macro Recall: {skill_result.macro_recall():.3f}\n")

    lines.append("| Skill | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for row in skill_result.summary_table():
        lines.append(f"| {row['label']} | {row['precision']} | {row['recall']} | {row['f1']} | {row['support']} |")

    lines.append("\n### Skill tagging errors\n")
    for err in skill_result.errors:
        lines.append(f"- **{err['question_id']}**: \"{err['text']}\"")
        lines.append(f"  - gold: {err['gold']}, predicted: {err['predicted']}")

    lines.append("\n## Format tagging performance\n")
    lines.append(f"- Macro F1: {format_result.macro_f1():.3f}")
    lines.append(f"- Macro Precision: {format_result.macro_precision():.3f}")
    lines.append(f"- Macro Recall: {format_result.macro_recall():.3f}\n")

    lines.append("| Format | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for row in format_result.summary_table():
        lines.append(f"| {row['label']} | {row['precision']} | {row['recall']} | {row['f1']} | {row['support']} |")

    lines.append("\n### Format tagging errors\n")
    for err in format_result.errors:
        lines.append(f"- **{err['question_id']}**: \"{err['text']}\"")
        lines.append(f"  - gold: {err['gold']}, predicted: {err['predicted']}")

    lines.append("\n## Reading this report\n")
    lines.append(
        "The baseline is deliberately naive (literal keyword matching). Low recall on "
        "paraphrased or word-problem-wrapped questions is the expected failure mode - "
        "it's the reason to add an LLM-based classifier and/or semantic search, not "
        "a sign something is broken. Once real gold-labeled data exists, run "
        "`compare_models()` with both the baseline and an `LLMClassifier` instance "
        "to see whether the added cost of an API call is actually earning better recall."
    )

    report = "\n".join(lines)
    Path("eval/evaluation_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
