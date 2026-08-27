"""
Semantic search backstop for when tags miss a question (e.g. a paraphrase
that doesn't share vocabulary with the taxonomy keywords).

This uses TF-IDF + cosine similarity rather than a neural embedding model.
That's a deliberate choice for this stage: it needs no external API call
and no downloaded model weights, so it runs anywhere and is fast to test.
The interface (`SemanticSearchIndex.query`) is the same shape you'd want
for a real embedding model (e.g. OpenAI/Gemini embeddings, or a local
sentence-transformers model) - swapping the backend later doesn't change
anything that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import Question


@dataclass
class ScoredQuestion:
    question: Question
    score: float


class SemanticSearchIndex:
    def __init__(self, questions: list[Question]):
        self.questions = questions
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [q.text for q in questions]
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def query(self, text: str, top_k: int = 5) -> list[ScoredQuestion]:
        if self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [
            ScoredQuestion(question=self.questions[i], score=float(sims[i]))
            for i in ranked_idx
            if sims[i] > 0
        ]
