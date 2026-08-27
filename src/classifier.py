"""
Tagging models, following a common interface so different approaches can be
swapped in and benchmarked against each other (see evaluate.py).

Two implementations:

1. KeywordBaselineClassifier - a simple, fully offline rule-based tagger.
   This exists for two reasons: (a) it's a real baseline to compare an LLM
   against - if the LLM doesn't clearly beat this, it's not earning its
   cost - and (b) it demonstrates a testable failure mode (paraphrased
   questions with no keyword overlap get missed), which motivates why we
   want a learned/semantic approach at all.

2. LLMClassifier - a provider-agnostic wrapper (OpenAI / Gemini / Anthropic)
   that sends the question + taxonomy and asks for structured JSON back.
   The HTTP call is isolated in a small `_call_*` method per provider so it
   can be mocked in tests without hitting the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re

from .schema import Question, TagPrediction, Taxonomy


class TaggingModel(ABC):
    @abstractmethod
    def predict(self, question: Question) -> TagPrediction:
        ...

    def predict_batch(self, questions: list[Question]) -> list[TagPrediction]:
        return [self.predict(q) for q in questions]


class KeywordBaselineClassifier(TaggingModel):
    """Tags a question by literal keyword/phrase matching against the taxonomy."""

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy

    def predict(self, question: Question) -> TagPrediction:
        text = question.text.lower()

        matched_skills = [
            skill.id
            for skill in self.taxonomy.all_skills()
            if any(kw.lower() in text for kw in skill.keywords)
        ]

        matched_formats = [
            fmt.id
            for fmt in self.taxonomy.formats
            if fmt.keywords and any(kw.lower() in text for kw in fmt.keywords)
        ]
        # "straightforward" is the implicit default when nothing else about
        # phrasing/wrapper matched.
        if not any(f in matched_formats for f in ("word_problem", "graph_based", "table_based", "diagram_based", "conceptual")):
            matched_formats.append("straightforward")

        return TagPrediction(
            question_id=question.id,
            skills=matched_skills,
            formats=matched_formats,
            confidence=None,  # rule-based match has no calibrated confidence
        )


LLM_SYSTEM_PROMPT = """You are tagging SAT Math questions with a fixed taxonomy.
Given a question and the list of valid skill IDs and format IDs below, return
ONLY a JSON object of the form:
{{"skills": ["skill_id", ...], "formats": ["format_id", ...], "confidence": 0.0-1.0}}

Valid skill IDs: {skill_ids}
Valid format IDs: {format_ids}

Only use IDs from these lists. A question may have more than one skill or format tag.
Do not include any text outside the JSON object."""


class LLMClassifier(TaggingModel):
    """
    Provider-agnostic LLM-based tagger. Supports "openai", "gemini", "anthropic".

    The actual HTTP call is isolated per provider so tests can mock
    `_call_provider` directly instead of hitting the network - this lets us
    unit test prompt construction and response parsing without an API key.
    """

    def __init__(self, taxonomy: Taxonomy, provider: str = "gemini", api_key: str | None = None, model: str | None = None):
        self.taxonomy = taxonomy
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model(provider)

    @staticmethod
    def _default_model(provider: str) -> str:
        return {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-haiku-4-5",
        }.get(provider, "gpt-4o-mini")

    def _build_prompt(self, question: Question) -> str:
        skill_ids = ", ".join(sorted(self.taxonomy.skill_ids()))
        format_ids = ", ".join(sorted(self.taxonomy.format_ids()))
        system = LLM_SYSTEM_PROMPT.format(skill_ids=skill_ids, format_ids=format_ids)
        user = f"Question: {question.text}\nChoices: {question.choices}"
        return f"{system}\n\n{user}"

    def predict(self, question: Question) -> TagPrediction:
        prompt = self._build_prompt(question)
        raw_response = self._call_provider(prompt)
        return self._parse_response(question.id, raw_response)

    def _call_provider(self, prompt: str) -> str:
        """
        Makes the actual API call. Isolated as its own method so tests can
        monkeypatch/mock it without needing network access or a real key.
        Raises if no api_key was configured - this is intentional: we never
        want to silently skip tagging.
        """
        if not self.api_key:
            raise RuntimeError(
                f"No API key configured for provider '{self.provider}'. "
                "Set one via LLMClassifier(..., api_key=...) before calling predict()."
            )

        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _call_anthropic(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _parse_response(self, question_id: str, raw_response: str) -> TagPrediction:
        # Strip markdown code fences if the model wrapped its JSON in them.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse model response as JSON: {raw_response!r}") from e

        valid_skills = self.taxonomy.skill_ids()
        valid_formats = self.taxonomy.format_ids()

        # Defensive filtering: drop any hallucinated IDs outside the taxonomy
        # rather than silently trusting the model's output.
        skills = [s for s in data.get("skills", []) if s in valid_skills]
        formats = [f for f in data.get("formats", []) if f in valid_formats]

        return TagPrediction(
            question_id=question_id,
            skills=skills,
            formats=formats,
            confidence=data.get("confidence"),
        )
