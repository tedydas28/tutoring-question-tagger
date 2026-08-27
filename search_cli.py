"""
Search your tagged SAT question bank from the command line.

Examples:
  python3 search_cli.py --cb-skill "Linear equations in one variable"
  python3 search_cli.py --skill similar_triangles
  python3 search_cli.py --domain "Advanced Math" --difficulty Hard
  python3 search_cli.py --query "margin of error"
  python3 search_cli.py --skill ratios_rates --format word_problem --limit 5

Run with --list-skills or --list-cb-skills to see valid values for --skill
and --cb-skill.
"""

import argparse
import json

from src.embeddings import SemanticSearchIndex
from src.schema import Question, Taxonomy


def load_tagged_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def print_result(r: dict):
    text = (r["stem_text_partial"] + " " + r["answer_text_partial"]).strip()
    snippet = (text[:140] + "...") if len(text) > 140 else text
    print(f"\n[{r['id']}]  {r['domain']} > {r['cb_skill']}  ({r['difficulty']})")
    if r["auto_tags_skills"]:
        print(f"  fine-grained skill tags: {', '.join(r['auto_tags_skills'])}")
    if r["auto_tags_formats"]:
        print(f"  format tags: {', '.join(r['auto_tags_formats'])}")
    print(f"  {snippet}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cb-skill", help="Filter by official College Board skill name (exact match)")
    parser.add_argument("--skill", help="Filter by fine-grained skill tag id (e.g. similar_triangles)")
    parser.add_argument("--format", dest="fmt", help="Filter by format tag id (e.g. word_problem)")
    parser.add_argument("--domain", help="Filter by domain (e.g. Algebra, 'Advanced Math')")
    parser.add_argument("--difficulty", help="Filter by difficulty (Easy/Medium/Hard)")
    parser.add_argument("--query", help="Free-text semantic search over question text")
    parser.add_argument("--limit", type=int, default=10, help="Max results to show (default 10)")
    parser.add_argument("--list-skills", action="store_true", help="List all fine-grained skill tag ids and exit")
    parser.add_argument("--list-cb-skills", action="store_true", help="List all official CB skill names and exit")
    args = parser.parse_args()

    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")

    if args.list_skills:
        for s in taxonomy.all_skills():
            print(f"  {s.id:35s} {s.name}")
        return

    tagged = load_tagged_questions("data/tagged_real_questions.json")

    if args.list_cb_skills:
        for name in sorted({t["cb_skill"] for t in tagged}):
            print(f"  {name}")
        return

    results = tagged
    if args.domain:
        results = [r for r in results if args.domain.lower() in r["domain"].lower()]
    if args.difficulty:
        results = [r for r in results if r["difficulty"].lower() == args.difficulty.lower()]
    if args.cb_skill:
        results = [r for r in results if args.cb_skill.lower() in r["cb_skill"].lower()]
    if args.skill:
        results = [r for r in results if args.skill in r["auto_tags_skills"]]
    if args.fmt:
        results = [r for r in results if args.fmt in r["auto_tags_formats"]]

    if args.query:
        questions = [Question(id=r["id"], text=(r["stem_text_partial"] + " " + r["answer_text_partial"])) for r in results]
        index = SemanticSearchIndex(questions)
        ranked = index.query(args.query, top_k=args.limit)
        ranked_ids = [sq.question.id for sq in ranked]
        by_id = {r["id"]: r for r in results}
        results = [by_id[qid] for qid in ranked_ids if qid in by_id]
    else:
        results = results[:args.limit]

    print(f"\n{len(results)} result(s) shown (of {len(tagged)} total questions in the bank)")
    for r in results:
        print_result(r)


if __name__ == "__main__":
    main()
