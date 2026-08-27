from src.classifier import TaggingModel
from src.evaluate import LabelScore, evaluate_model
from src.schema import Question, TagPrediction


class FixedModel(TaggingModel):
    """A fake model that returns pre-set predictions, so we can hand-verify metrics."""

    def __init__(self, canned: dict[str, TagPrediction]):
        self.canned = canned

    def predict(self, question: Question) -> TagPrediction:
        return self.canned[question.id]


def test_label_score_precision_recall_f1_hand_computed():
    score = LabelScore(label="x", true_positives=3, false_positives=1, false_negatives=2)
    assert score.precision == 3 / 4
    assert score.recall == 3 / 5
    expected_f1 = 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5))
    assert abs(score.f1 - expected_f1) < 1e-9


def test_label_score_handles_zero_denominators():
    empty = LabelScore(label="x")
    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0


def test_evaluate_model_exact_match_gives_perfect_scores():
    questions = [
        Question(id="a", text="...", gold_skills=["similar_triangles"], gold_formats=["word_problem"]),
        Question(id="b", text="...", gold_skills=["linear_eq_one_var"], gold_formats=["straightforward"]),
    ]
    model = FixedModel({
        "a": TagPrediction(question_id="a", skills=["similar_triangles"], formats=["word_problem"]),
        "b": TagPrediction(question_id="b", skills=["linear_eq_one_var"], formats=["straightforward"]),
    })

    skill_result, format_result = evaluate_model(model, questions)

    assert skill_result.macro_f1() == 1.0
    assert format_result.macro_f1() == 1.0
    assert skill_result.errors == []
    assert format_result.errors == []


def test_evaluate_model_known_error_counts():
    """
    One question: gold skill is A, model predicts B (wrong).
    This should register exactly 1 false negative on A and 1 false positive on B.
    """
    questions = [Question(id="a", text="...", gold_skills=["skill_A"], gold_formats=[])]
    model = FixedModel({
        "a": TagPrediction(question_id="a", skills=["skill_B"], formats=[]),
    })

    skill_result, _ = evaluate_model(model, questions)

    assert skill_result.label_scores["skill_A"].false_negatives == 1
    assert skill_result.label_scores["skill_A"].true_positives == 0
    assert skill_result.label_scores["skill_B"].false_positives == 1
    assert len(skill_result.errors) == 1
    assert skill_result.errors[0]["question_id"] == "a"


def test_evaluate_model_partial_multilabel_match():
    """
    Gold has two skills, model gets one right and misses one - precision
    should be perfect (no wrong guesses) but recall should be 0.5 overall
    across the two labels.
    """
    questions = [Question(id="a", text="...", gold_skills=["skill_A", "skill_B"], gold_formats=[])]
    model = FixedModel({
        "a": TagPrediction(question_id="a", skills=["skill_A"], formats=[]),
    })

    skill_result, _ = evaluate_model(model, questions)

    assert skill_result.label_scores["skill_A"].precision == 1.0
    assert skill_result.label_scores["skill_A"].recall == 1.0
    assert skill_result.label_scores["skill_B"].recall == 0.0
    assert skill_result.label_scores["skill_B"].true_positives == 0
    assert skill_result.label_scores["skill_B"].false_negatives == 1
