"""
Core data structures for the SAT question tagging project.

Design note: Question carries an OPTIONAL set of gold labels (human-verified
tags). This lets the same object flow through the untagged pipeline (real
usage) and the evaluation pipeline (where gold labels exist for scoring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import yaml


@dataclass
class Skill:
    id: str
    name: str
    keywords: list[str] = field(default_factory=list)
    cb_skill: str | None = None  # official College Board skill name, when known


@dataclass
class Domain:
    id: str
    name: str
    skills: list[Skill]


@dataclass
class Format:
    id: str
    name: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    domains: list[Domain]
    formats: list[Format]
    difficulty_levels: list[str]

    @classmethod
    def from_yaml(cls, path: str) -> "Taxonomy":
        with open(path) as f:
            raw = yaml.safe_load(f)
        domains = [
            Domain(
                id=d["id"],
                name=d["name"],
                skills=[Skill(**s) for s in d["skills"]],
            )
            for d in raw["domains"]
        ]
        formats = [Format(**fmt) for fmt in raw["formats"]]
        return cls(domains=domains, formats=formats, difficulty_levels=raw["difficulty_levels"])

    def all_skills(self) -> list[Skill]:
        return [s for d in self.domains for s in d.skills]

    def skill_ids(self) -> set[str]:
        return {s.id for s in self.all_skills()}

    def format_ids(self) -> set[str]:
        return {f.id for f in self.formats}

    def skill_by_id(self, skill_id: str) -> Skill | None:
        for s in self.all_skills():
            if s.id == skill_id:
                return s
        return None

    def domain_of_skill(self, skill_id: str) -> Domain | None:
        for d in self.domains:
            for s in d.skills:
                if s.id == skill_id:
                    return d
        return None


@dataclass
class Question:
    id: str
    text: str
    choices: list[str] = field(default_factory=list)
    answer: str | None = None
    difficulty: str | None = None  # from College Board metadata, if available

    # Gold labels - only populated for hand-labeled evaluation questions.
    gold_skills: list[str] = field(default_factory=list)
    gold_formats: list[str] = field(default_factory=list)


@dataclass
class TagPrediction:
    question_id: str
    skills: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    confidence: float | None = None  # 0-1, only meaningful for model-based predictions
