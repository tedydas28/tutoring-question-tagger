# SAT Question Tagger

A tagging + search system for SAT Math questions that goes finer-grained than
College Board's own Domain/Skill categories (e.g. "similar triangles" and
"special right triangles" as distinct searchable tags, not lumped into
"Right Triangles and Trigonometry").

## Why this is structured the way it is

This isn't "call an LLM and hope." It's built so tagging quality is
**measured**, not assumed:

- A **rule-based baseline** (`KeywordBaselineClassifier`) that needs no API
  and no cost, used as the floor any smarter approach has to beat.
- A **provider-agnostic LLM classifier** (`LLMClassifier`) that can hit
  OpenAI, Gemini, or Anthropic — same interface, swap the provider string.
- A **semantic search layer** (TF-IDF now, swappable for real embeddings
  later) that catches questions tags miss entirely.
- An **evaluation harness** that scores per-tag precision/recall/F1 against
  hand-labeled gold questions, with an error log so failure patterns are
  visible, not hidden behind one aggregate number.

## Current state (real data, all four Question Bank exports)

All four of your PDFs are College Board Question Bank exports, one question
per page. Parsed with `src/parse_pdf_export.py`:

| File | Domain | Questions |
|---|---|---|
| sat_algebra_all.pdf | Algebra | 616 |
| sat_advmath_all.pdf | Advanced Math | 540 |
| sat_geotrig_all.pdf | Geometry and Trigonometry | 348 |
| sat_ps_da_all.pdf | Problem-Solving and Data Analysis | 421 |
| **Total** | | **1,925** |

- **100% coverage** on official College Board Domain/Skill/Difficulty for
  all 1,925 questions — extracts cleanly as plain text. See
  `data/raw/parsed_questions.json`. (Known edge case: 2 of 1,927 pages are
  a continuation of a scatterplot question that spans two pages, so they
  don't carry their own Question ID — negligible, ~0.1%.)
- **Same data limitation as before applies to all four files:** College
  Board renders math as vector-typeset glyphs, not selectable text, so
  plain extraction gets the English wrapper but drops the actual math.
  Full-content tagging needs vision-based transcription (rasterize +
  read), not text extraction — see the Algebra example in the original
  writeup below for a side-by-side of what's lost.

### Every domain's draft taxonomy needed correction against real data

This turned out to be the actual pattern, not a one-off:

- **Algebra/Advanced Math** (first pass): found a missing skill
  ("Equivalent expressions") and a wrongly-merged pair ("Linear equations
  in two variables" vs. "Systems of two linear equations" are separate CB
  skills).
- **Geometry and Trigonometry**: official CB skill list is exactly 4
  categories. The draft had split "Lines, angles, and triangles" into
  several separate top-level skills (triangle congruence, similar
  triangles) that are actually one official skill, and had misplaced
  special right triangles outside "Right triangles and trigonometry"
  where it officially belongs.
- **Problem-Solving and Data Analysis**: official CB skill list is exactly
  7 categories. Two were entirely missing from the draft — "Inference
  from sample statistics and margin of error" and "Evaluating statistical
  claims: Observational studies and experiments."

`data/taxonomy.yaml` now has every skill's `cb_skill` field mapped to the
real, confirmed official name — this is worth knowing if you add more
Question Bank files later for other tests (PSAT, etc.): expect to re-verify
skill names against real data rather than trusting a draft.

### Real gold-labeled sample (12 questions, hand-transcribed via vision)

`data/real_gold_sample.json` — now spans all four domains, 12 questions
total, full text hand-transcribed by rasterizing real pages and reading
them (not synthetic, not text-extracted).

Baseline performance on this real 12-question sample:

| | Macro F1 | Macro Precision | Macro Recall |
|---|---|---|---|
| Skill tags | 0.542 | 0.607 | 0.750 |
| Format tags | 0.267 | — | — |

New real error cases from the expanded set: the sampling-inference question
about margin of error gets correctly tagged but also picks up a spurious
`percentages` tag (the question mentions "30%" and "3%" incidentally, even
though its actual skill is statistical inference) — a real example of why
literal keyword matching struggles with questions that use vocabulary from
one skill in service of a different one. The two Geometry questions were
missed entirely (`lines_angles`, `special_right_triangles`) because their
skill lives entirely in the diagram and equation, with almost no
distinguishing keyword in the surrounding text.

Full report: `eval/real_data_evaluation_report.md` (regenerate with
`python3 eval/run_real_evaluation.py`).

## Project layout

```
data/
  taxonomy.yaml           - the tag schema: domains, skills, formats, difficulty
  sample_questions.json   - 20 synthetic questions with hand-verified gold tags
src/
  schema.py               - Question, Taxonomy, TagPrediction data classes
  classifier.py           - KeywordBaselineClassifier + LLMClassifier
  embeddings.py            - TF-IDF semantic search index
  search.py               - QuestionBank: tag filter + semantic search combined
  evaluate.py             - precision/recall/F1 scoring + model comparison
tests/                    - 18 tests, all passing (run: pytest tests/ -v)
eval/
  run_evaluation.py       - regenerates the evaluation report
  evaluation_report.md    - the actual numbers from the last run
demo_search.py            - runnable end-to-end demo
```

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v                  # 18 tests
python3 demo_search.py            # tag + search demo
python3 eval/run_evaluation.py    # regenerate the evaluation report
```

## What's next (needs you)

1. **Scale up transcription.** Only 12 of 1,925 questions have full
   vision-transcribed text so far (done manually, in-chat). To tag the rest
   with fine-grained skills, you need one of:
   - A batch vision pipeline: rasterize each page (`pdftoppm`, already
     proven to work cleanly) and send it to a vision-capable LLM
     (GPT-4o, Gemini, or Claude) in one call that both transcribes the
     math AND assigns tags. This needs your own API key and runs outside
     this chat — I can write the batch script, but can't execute ~1,900
     external API calls from here.
   - Or: keep using College Board's own Domain/Skill/Difficulty (already
     100% extracted, zero cost) as your only tags for now, and add
     fine-grained tags incrementally as you tutor and notice patterns.
2. **A larger gold set.** 12 real hand-labeled questions across all four
   domains proved the pipeline and taxonomy hold up against real, messy
   data; it's not enough to trust the F1 numbers themselves. Aim for
   80-100+ before trusting the scores to mean anything.
3. **An API key** for whichever provider you pick, to run `LLMClassifier`
   for real and then `compare_models()` against the keyword baseline on the
   real gold set — the actual point of having both.

## How to actually use it

Two steps. First, tag all 1,925 real questions (run once, or again after
editing the taxonomy):

```bash
python3 tag_all_questions.py
```

This writes `data/tagged_real_questions.json` and prints coverage stats.

Then search from the command line with `search_cli.py`:

```bash
# By official College Board skill (100% reliable, works for everything)
python3 search_cli.py --cb-skill "Linear equations in one variable"

# By your fine-grained tag (works for the ~63% of questions with matchable text)
python3 search_cli.py --skill similar_triangles

# Combine filters
python3 search_cli.py --domain "Advanced Math" --difficulty Hard --format word_problem

# Free-text semantic search (finds relevant questions even without an exact tag match)
python3 search_cli.py --query "margin of error sample"

# See valid values for --skill and --cb-skill
python3 search_cli.py --list-skills
python3 search_cli.py --list-cb-skills

# Control how many results print (default 10)
python3 search_cli.py --skill similar_triangles --limit 20
```

Each result shows the question ID, official domain/skill/difficulty, any
fine-grained tags found, and a text snippet (partial — see the math
extraction limitation above) so you can identify the question and go find
it on the actual page in the PDF by its ID.

**Known noise to expect:** tagging still runs on keyword matching, so some
results pick up unrelated tags from incidental word overlap — e.g. a
statistics question about margin of error can pick up a spurious `circles`
tag from an unrelated substring match. `--cb-skill` filtering is unaffected
by this since it comes from the metadata table, not keyword matching — use
that when you need reliability over granularity.

## The UI (search bar + real PDF page delivery)

This is the actual point of the project: type what you need, get back real
questions with an exact preview of the original PDF page, and download them
(individually or bundled as a worksheet) to hand to a student.

**Why it pulls the original PDF page instead of showing extracted text:**
extracted text is missing the math (see the limitation above) - the source
PDF page is always complete and exact, so that's what gets delivered,
regardless of how good or bad the tagging under the hood is.

### Setup (one-time)

```bash
pip install -r requirements.txt   # now includes streamlit, pypdf, pymupdf
```

**Put your four original PDFs in `data/source_pdfs/`**, with these exact
filenames (there's a placeholder file there with the same instructions):

```
data/source_pdfs/sat_algebra_all.pdf
data/source_pdfs/sat_advmath_all.pdf
data/source_pdfs/sat_geotrig_all.pdf
data/source_pdfs/sat_ps_da_all.pdf
```

The project only stores each question's *filename*, not a full path (this
was a bug in an earlier version - it stored the path from the machine
that first parsed the PDFs, which broke on any other computer). `src/config.py`
resolves the filename against `data/source_pdfs/` at runtime - if you'd
rather keep your PDFs somewhere else, edit `SOURCE_PDF_DIR` there instead
of moving files around.

Then tag everything:

```bash
python3 tag_all_questions.py
```

### Run it

```bash
streamlit run app.py
```

This opens a browser tab automatically (usually `http://localhost:8501`).
Type a query like "similar triangles word problems" or "margin of error" in
the search bar, optionally narrow with the sidebar filters (Domain,
Difficulty, official CB skill), then:

- Click **"Preview original page"** on any result to see the exact page
  rendered as an image.
- Click **"Download this question (PDF)"** to save just that one page.
- Check **"Add to worksheet"** on several results, then use the **"Download
  worksheet (combined PDF)"** button at the bottom to get them all as one
  PDF - even when they come from different source files (e.g. mixing an
  Algebra and a Geometry question in one worksheet).

Close it with `Ctrl+C` in the terminal when you're done.

### How search works under the hood

The search bar runs semantic (TF-IDF) matching over each question's
available text PLUS its official CB skill name and any fine-grained tags -
so a query like "similar triangles" matches on the skill name even for
questions where the actual math didn't extract as text. This is why typing
skill-like phrases works better than exact quotes from a question.
