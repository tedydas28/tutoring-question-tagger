import json

import pytest

from src.classifier import KeywordBaselineClassifier
from src.schema import Question, Taxonomy
from src.search import QuestionBank


@pytest.fixture
def taxonomy():
    return Taxonomy.from_yaml("data/taxonomy.yaml")


@pytest.fixture
def sample_questions():
    with open("data/sample_questions.json") as f:
        raw = json.load(f)
    return [
        Question(
            id=r["id"], text=r["text"], choices=r["choices"], answer=r["answer"],
            difficulty=r["difficulty"], gold_skills=r["gold_skills"], gold_formats=r["gold_formats"],
        )
        for r in raw
    ]


@pytest.fixture
def question_bank(taxonomy, sample_questions):
    clf = KeywordBaselineClassifier(taxonomy)
    predictions = {q.id: clf.predict(q) for q in sample_questions}
    return QuestionBank(questions=sample_questions, predictions=predictions)


def test_filter_by_skill_returns_only_matching_questions(question_bank):
    results = question_bank.filter(skills=["similar_triangles"])
    assert len(results) >= 1
    assert all("similar" in q.text.lower() or "triangle" in q.text.lower() for q in results)


def test_filter_by_difficulty(question_bank):
    results = question_bank.filter(difficulty="easy")
    assert all(q.difficulty == "easy" for q in results)
    assert len(results) > 0


def test_filter_by_skill_and_format_combined(question_bank):
    results = question_bank.filter(skills=["similar_triangles"], formats=["word_problem"])
    # q002 (flagpole) is a gold word-problem similar-triangles case, but the
    # keyword baseline can't detect its skill (see test_classifier.py) - so
    # depending on tagging quality this may legitimately return zero results.
    # This test just checks the filter logic doesn't crash and stays consistent.
    for q in results:
        pred = question_bank.predictions[q.id]
        assert "similar_triangles" in pred.skills
        assert "word_problem" in pred.formats


def test_semantic_search_finds_relevant_question_even_without_exact_tag(question_bank):
    """
    q002 (the flagpole/shadow problem) is missed by the keyword tagger's
    skill detection, but semantic search should still surface it for a
    query about proportional shadows - this is the whole point of having
    a semantic backstop layer.
    """
    results = question_bank.semantic_search("shadow height proportion", top_k=5)
    result_ids = [q.id for q in results]
    assert "q002" in result_ids


def test_search_combines_filter_and_ranking(question_bank):
    results = question_bank.search(query_text="triangle", skills=["special_right_triangles"], top_k=5)
    for q in results:
        assert "special_right_triangles" in question_bank.predictions[q.id].skills
