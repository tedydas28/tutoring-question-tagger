"""
Runs the LLM-based classifier (Claude Haiku, via your own Anthropic API key)
against the 12 real gold-labeled questions and compares it head-to-head
with the free keyword baseline.

Setup (pick one):
    Option A - .env file (recommended, only needs doing once):
        1. Copy .env.example to a new file named .env
        2. Open .env and replace "your-key-here" with your real key
        (.env is gitignored, so it will never get committed)

    Option B - environment variable, one terminal session at a time:
        Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-key-here"
        Mac/Linux:            export ANTHROPIC_API_KEY=your-key-here

Run:
    python3 run_llm_comparison.py

This makes exactly 12 API calls total (one per gold question) - at Claude
Haiku 4.5 pricing that's a fraction of a cent. Predictions are computed
once and reused for both the aggregate scores and the per-question
printout below, so nothing gets double-charged.
"""

import json
import os
import sys

from dotenv import load_dotenv

from src.classifier import KeywordBaselineClassifier, LLMClassifier, TaggingModel
from src.evaluate import compare_models
from src.schema import Question, TagPrediction, Taxonomy

load_dotenv()  # reads .env into os.environ if the file exists - no-op if it doesn't


class PrecomputedModel(TaggingModel):
    """Wraps predictions that were already computed, so compare_models()
    can score them without triggering a second round of API calls."""

    def __init__(self, predictions: dict[str, TagPrediction]):
        self.predictions = predictions

    def predict(self, question: Question) -> TagPrediction:
        return self.predictions[question.id]


def load_gold_questions() -> tuple[Taxonomy, list[Question]]:
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    raw = json.load(open("data/real_gold_sample.json"))
    questions = [
        Question(
            id=r["id"], text=r["text"], difficulty=r["difficulty"],
            gold_skills=r["gold_skills"], gold_formats=r["gold_formats"],
        )
        for r in raw
    ]
    return taxonomy, questions


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY isn't set.\n")
        print("Either:")
        print("  1. Copy .env.example to .env and put your real key in it, or")
        print("  2. Set it for this terminal session:")
        print('     Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-key-here"')
        print("     Mac/Linux:            export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    taxonomy, gold_questions = load_gold_questions()
    baseline = KeywordBaselineClassifier(taxonomy)
    llm = LLMClassifier(taxonomy, provider="anthropic", api_key=api_key)

    print(f"Calling Claude Haiku on {len(gold_questions)} gold questions "
          f"(one API call each, roughly a fraction of a cent total)...\n")

    llm_predictions = {}
    for q in gold_questions:
        try:
            llm_predictions[q.id] = llm.predict(q)
            print(f"  [{q.id}] done")
        except RuntimeError as e:
            print(f"\nStopped - API call failed on question {q.id}: {e}")
            sys.exit(1)

    llm_wrapped = PrecomputedModel(llm_predictions)

    rows = compare_models({"keyword_baseline": baseline, "claude_haiku": llm_wrapped}, gold_questions)

    print("\n" + "=" * 70)
    print(f"{'model':<20} {'skill F1':<10} {'skill P':<10} {'skill R':<10} {'format F1':<10}")
    for row in rows:
        print(f"{row['model']:<20} {row['skill_macro_f1']:<10} {row['skill_macro_precision']:<10} "
              f"{row['skill_macro_recall']:<10} {row['format_macro_f1']:<10}")

    print("\nPer-question skill tags (gold vs. each model):")
    for q in gold_questions:
        baseline_pred = baseline.predict(q)
        llm_pred = llm_predictions[q.id]
        print(f"\n  [{q.id}]")
        print(f"    gold:    {q.gold_skills}")
        print(f"    keyword: {baseline_pred.skills}")
        print(f"    claude:  {llm_pred.skills}")


if __name__ == "__main__":
    main()
