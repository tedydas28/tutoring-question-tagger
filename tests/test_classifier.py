from unittest.mock import patch

import pytest

from src.classifier import KeywordBaselineClassifier, LLMClassifier
from src.schema import Question, Taxonomy


@pytest.fixture
def taxonomy():
    return Taxonomy.from_yaml("data/taxonomy.yaml")


def test_keyword_classifier_catches_explicit_similar_triangles(taxonomy):
    q = Question(id="t1", text="Triangle ABC is similar to triangle DEF. Find EF.")
    pred = KeywordBaselineClassifier(taxonomy).predict(q)
    assert "similar_triangles" in pred.skills


def test_keyword_classifier_misses_paraphrased_similar_triangles(taxonomy):
    """
    This is the expected FAILURE mode of a keyword approach: the shadow/
    flagpole problem is a similar-triangles question but never uses the
    words "similar" or "triangle". This test documents the limitation
    that motivates using an LLM/semantic layer at all - it should stay
    red for the baseline and is the exact case the LLM path is meant to fix.
    """
    q = Question(
        id="t2",
        text="A flagpole casts a shadow 20 feet long. A 5-foot-tall person "
             "casts a shadow 4 feet long. How tall is the flagpole?",
    )
    pred = KeywordBaselineClassifier(taxonomy).predict(q)
    assert "similar_triangles" not in pred.skills  # documents the gap, not a bug


def test_keyword_classifier_defaults_to_straightforward_format(taxonomy):
    q = Question(id="t3", text="Solve for x: 2x + 4 = 10")
    pred = KeywordBaselineClassifier(taxonomy).predict(q)
    assert "straightforward" in pred.formats
    assert "word_problem" not in pred.formats


def test_keyword_classifier_tags_word_problem_format(taxonomy):
    q = Question(id="t4", text="A store sells notebooks for $2 each. How many did John buy?")
    pred = KeywordBaselineClassifier(taxonomy).predict(q)
    assert "word_problem" in pred.formats


def test_llm_classifier_requires_api_key(taxonomy):
    q = Question(id="t5", text="Solve for x: x + 1 = 2")
    model = LLMClassifier(taxonomy, provider="gemini", api_key=None)
    with pytest.raises(RuntimeError, match="No API key configured"):
        model.predict(q)


def test_llm_classifier_parses_mocked_response(taxonomy):
    """
    Tests prompt construction and response parsing WITHOUT hitting a real
    API - the network call itself is mocked so this test is fast, free,
    and doesn't depend on network access or a live key.
    """
    q = Question(id="t6", text="Triangle ABC is similar to triangle DEF.")
    model = LLMClassifier(taxonomy, provider="gemini", api_key="fake-key-for-test")

    fake_response = '{"skills": ["similar_triangles"], "formats": ["diagram_based"], "confidence": 0.92}'
    with patch.object(model, "_call_provider", return_value=fake_response) as mock_call:
        pred = model.predict(q)

    mock_call.assert_called_once()
    assert pred.skills == ["similar_triangles"]
    assert pred.formats == ["diagram_based"]
    assert pred.confidence == 0.92


def test_llm_classifier_drops_hallucinated_labels(taxonomy):
    """A model could invent a skill ID that isn't in our taxonomy - we must not trust it blindly."""
    q = Question(id="t7", text="Solve for x: x + 1 = 2")
    model = LLMClassifier(taxonomy, provider="gemini", api_key="fake-key-for-test")

    fake_response = '{"skills": ["linear_eq_one_var", "made_up_skill_xyz"], "formats": ["straightforward"]}'
    with patch.object(model, "_call_provider", return_value=fake_response):
        pred = model.predict(q)

    assert pred.skills == ["linear_eq_one_var"]
    assert "made_up_skill_xyz" not in pred.skills


def test_llm_classifier_handles_markdown_fenced_json(taxonomy):
    """Models frequently wrap JSON in ```json ... ``` even when told not to - must handle it."""
    q = Question(id="t8", text="Solve for x: x + 1 = 2")
    model = LLMClassifier(taxonomy, provider="gemini", api_key="fake-key-for-test")

    fenced_response = '```json\n{"skills": ["linear_eq_one_var"], "formats": ["straightforward"]}\n```'
    with patch.object(model, "_call_provider", return_value=fenced_response):
        pred = model.predict(q)

    assert pred.skills == ["linear_eq_one_var"]
