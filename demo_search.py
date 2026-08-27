"""
Quick end-to-end demo: tag the sample questions, then search them two ways -
by exact skill tag, and by free-text semantic query.

Run: python3 demo_search.py
"""

import json

from src.classifier import KeywordBaselineClassifier
from src.schema import Question, Taxonomy
from src.search import QuestionBank


def load_questions(path: str) -> list[Question]:
    with open(path) as f:
        raw = json.load(f)
    return [Question(id=r["id"], text=r["text"], choices=r["choices"], answer=r["answer"], difficulty=r["difficulty"]) for r in raw]


def main():
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    questions = load_questions("data/sample_questions.json")

    model = KeywordBaselineClassifier(taxonomy)
    predictions = {q.id: model.predict(q) for q in questions}
    bank = QuestionBank(questions=questions, predictions=predictions)

    print("=== Tag filter: similar_triangles ===")
    for q in bank.filter(skills=["similar_triangles"]):
        print(f"  [{q.id}] {q.text[:70]}...")

    print("\n=== Semantic search: 'shadow height proportion' ===")
    print("(catches the flagpole word problem even though the keyword tagger missed its skill tag)")
    for q in bank.semantic_search("shadow height proportion", top_k=3):
        print(f"  [{q.id}] {q.text[:70]}...")

    print("\n=== Combined: skill=special_right_triangles + query='hypotenuse' ===")
    for q in bank.search(query_text="hypotenuse", skills=["special_right_triangles"]):
        print(f"  [{q.id}] {q.text[:70]}...")


if __name__ == "__main__":
    main()
