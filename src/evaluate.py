"""
Evaluation harness for comparing tagging models against hand-labeled gold data.

This is a multi-label classification problem (a question can have more than
one skill and more than one format tag), so we score it the way multi-label
problems are actually scored: per-label precision/recall/F1, then macro-
averaged across labels, plus a small error-analysis report so mislabeling
patterns are visible rather than hidden inside a single aggregate number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .classifier import TaggingModel
from .schema import Question


@dataclass
class LabelScore:
    label: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvaluationResult:
    label_scores: dict[str, LabelScore] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)  # per-question mismatches, for error analysis

    def macro_f1(self) -> float:
        scores = [s.f1 for s in self.label_scores.values() if (s.true_positives + s.false_positives + s.false_negatives) > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def macro_precision(self) -> float:
        scores = [s.precision for s in self.label_scores.values() if (s.true_positives + s.false_positives) > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def macro_recall(self) -> float:
        scores = [s.recall for s in self.label_scores.values() if (s.true_positives + s.false_negatives) > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def summary_table(self) -> list[dict]:
        rows = []
        for label, s in sorted(self.label_scores.items()):
            if s.true_positives + s.false_positives + s.false_negatives == 0:
                continue
            rows.append({
                "label": label,
                "precision": round(s.precision, 2),
                "recall": round(s.recall, 2),
                "f1": round(s.f1, 2),
                "support": s.true_positives + s.false_negatives,
            })
        return rows


def _score_label_set(gold: list[str], predicted: list[str], label_scores: dict[str, LabelScore]):
    gold_set, pred_set = set(gold), set(predicted)
    for label in gold_set | pred_set:
        score = label_scores.setdefault(label, LabelScore(label=label))
        if label in gold_set and label in pred_set:
            score.true_positives += 1
        elif label in pred_set and label not in gold_set:
            score.false_positives += 1
        elif label in gold_set and label not in pred_set:
            score.false_negatives += 1


def evaluate_model(model: TaggingModel, gold_questions: list[Question]) -> tuple[EvaluationResult, EvaluationResult]:
    """
    Runs `model` over every gold question and scores skill tags and format
    tags separately (they're different label spaces). Returns
    (skill_result, format_result).
    """
    skill_result = EvaluationResult()
    format_result = EvaluationResult()

    for q in gold_questions:
        pred = model.predict(q)

        _score_label_set(q.gold_skills, pred.skills, skill_result.label_scores)
        _score_label_set(q.gold_formats, pred.formats, format_result.label_scores)

        if set(pred.skills) != set(q.gold_skills):
            skill_result.errors.append({
                "question_id": q.id,
                "text": q.text[:80] + ("..." if len(q.text) > 80 else ""),
                "gold": q.gold_skills,
                "predicted": pred.skills,
            })
        if set(pred.formats) != set(q.gold_formats):
            format_result.errors.append({
                "question_id": q.id,
                "text": q.text[:80] + ("..." if len(q.text) > 80 else ""),
                "gold": q.gold_formats,
                "predicted": pred.formats,
            })

    return skill_result, format_result


def compare_models(models: dict[str, TaggingModel], gold_questions: list[Question]) -> list[dict]:
    """Runs multiple models over the same gold set and returns a comparison table."""
    rows = []
    for name, model in models.items():
        skill_result, format_result = evaluate_model(model, gold_questions)
        rows.append({
            "model": name,
            "skill_macro_f1": round(skill_result.macro_f1(), 3),
            "skill_macro_precision": round(skill_result.macro_precision(), 3),
            "skill_macro_recall": round(skill_result.macro_recall(), 3),
            "format_macro_f1": round(format_result.macro_f1(), 3),
            "n_skill_errors": len(skill_result.errors),
            "n_format_errors": len(format_result.errors),
        })
    return rows
