"""
The user-facing retrieval layer: filter by tag, search by free text, or both.
This is what "find me similar-triangles problems that are word problems"
actually resolves to.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import SemanticSearchIndex
from .schema import Question, TagPrediction


@dataclass
class QuestionBank:
    questions: list[Question]
    predictions: dict[str, TagPrediction]  # keyed by question_id

    def __post_init__(self):
        self._semantic_index = SemanticSearchIndex(self.questions)

    def filter(
        self,
        skills: list[str] | None = None,
        formats: list[str] | None = None,
        difficulty: str | None = None,
    ) -> list[Question]:
        results = []
        for q in self.questions:
            pred = self.predictions.get(q.id)
            if pred is None:
                continue
            if skills and not any(s in pred.skills for s in skills):
                continue
            if formats and not any(f in pred.formats for f in formats):
                continue
            if difficulty and q.difficulty != difficulty:
                continue
            results.append(q)
        return results

    def semantic_search(self, query_text: str, top_k: int = 5) -> list[Question]:
        return [sq.question for sq in self._semantic_index.query(query_text, top_k=top_k)]

    def search(
        self,
        query_text: str | None = None,
        skills: list[str] | None = None,
        formats: list[str] | None = None,
        difficulty: str | None = None,
        top_k: int = 10,
    ) -> list[Question]:
        """
        Combines exact tag filtering with optional free-text ranking.
        If a free-text query is given, results are restricted to the tag
        filter first, then ranked by semantic similarity to the query.
        """
        candidates = self.filter(skills=skills, formats=formats, difficulty=difficulty)

        if not query_text:
            return candidates[:top_k]

        candidate_ids = {q.id for q in candidates}
        ranked = self._semantic_index.query(query_text, top_k=len(self.questions))
        filtered_ranked = [sq.question for sq in ranked if sq.question.id in candidate_ids]
        return filtered_ranked[:top_k]
