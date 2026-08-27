"""
Search UI for the tagged SAT question bank.

Run with:
    streamlit run app.py

Type what you're looking for in the search bar (e.g. "similar triangles
word problems", "margin of error", "quadratic factoring easy") and get
back real questions from your question bank - each with a preview of the
EXACT original PDF page (not a reconstruction) and a one-click download of
that page as a standalone PDF you can hand to a student. Select several
and download them together as one worksheet.
"""

import json

import streamlit as st

from src.embeddings import SemanticSearchIndex
from src.pdf_extract import extract_multiple_pages_pdf, extract_single_page_pdf, render_page_image
from src.schema import Question, Taxonomy
from src.config import resolve_pdf_path

st.set_page_config(page_title="SAT Question Bank Search", layout="wide")


@st.cache_data
def load_data():
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    with open("data/tagged_real_questions.json") as f:
        tagged = json.load(f)
    return taxonomy, tagged


@st.cache_data
def build_search_index(tagged: list[dict]):
    """
    The search haystack per question combines whatever question text
    extracted PLUS the official CB skill name and fine-grained tag names.
    This matters a lot here: a typed query like "similar triangles" should
    match on the SKILL NAME even for the ~37% of questions whose actual
    text didn't extract well enough for keyword tagging to fire.
    """
    docs = []
    for r in tagged:
        haystack_parts = [
            r["stem_text_partial"],
            r["answer_text_partial"],
            r["cb_skill"],
            r["domain"],
            " ".join(r["auto_tags_skills"]),
            " ".join(r["auto_tags_formats"]),
        ]
        docs.append(Question(id=r["id"], text=" ".join(p for p in haystack_parts if p)))
    return SemanticSearchIndex(docs)


@st.cache_data
def get_page_image(pdf_path: str, page_number: int) -> bytes:
    return render_page_image(pdf_path, page_number)


def main():
    st.title("SAT Question Bank Search")
    st.caption("1,925 real questions across Algebra, Advanced Math, Geometry & Trig, "
               "and Problem-Solving & Data Analysis.")

    taxonomy, tagged = load_data()
    index = build_search_index(tagged)
    by_id = {r["id"]: r for r in tagged}

    # Worksheet summary + download - shown at the TOP, before search/results,
    # so it's visible without scrolling. Reads current checkbox state from
    # session_state (set by the checkboxes further down on the previous run).
    selected_ids = [
        key[len("chk_"):] for key, checked in st.session_state.items()
        if key.startswith("chk_") and checked
    ]
    selected_pages = [
        (resolve_pdf_path(by_id[qid]["source_pdf_filename"]), by_id[qid]["page_number"])
        for qid in selected_ids if qid in by_id
    ]
    worksheet_box = st.container(border=True)
    with worksheet_box:
        if selected_pages:
            st.subheader(f"📋 {len(selected_pages)} question(s) selected for worksheet")
            worksheet_bytes = extract_multiple_pages_pdf(selected_pages)
            st.download_button(
                "Download worksheet (combined PDF)",
                data=worksheet_bytes,
                file_name="worksheet.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("📋 No questions selected yet - check \"Add to worksheet\" on any result below.")

    st.divider()

    with st.sidebar:
        st.header("Filters")
        domains = sorted({r["domain"] for r in tagged})
        selected_domains = st.multiselect("Domain", domains, default=[])

        difficulties = ["Easy", "Medium", "Hard"]
        selected_difficulties = st.multiselect("Difficulty", difficulties, default=[])

        cb_skills = sorted({r["cb_skill"] for r in tagged})
        selected_cb_skills = st.multiselect("Official College Board skill", cb_skills, default=[])

        st.divider()
        st.caption(
            "Search matches question text, official skill names, and "
            "fine-grained tags. Filters narrow further. Not every question "
            "has fine-grained tags (~63% do) - the official CB skill filter "
            "above always works."
        )

    query = st.text_input(
        "What kind of questions do you need?",
        placeholder="e.g. similar triangles word problems, margin of error, easy quadratic factoring",
    )

    results = tagged
    if selected_domains:
        results = [r for r in results if r["domain"] in selected_domains]
    if selected_difficulties:
        results = [r for r in results if r["difficulty"] in selected_difficulties]
    if selected_cb_skills:
        results = [r for r in results if r["cb_skill"] in selected_cb_skills]

    if query:
        candidate_ids = {r["id"] for r in results}
        ranked = index.query(query, top_k=len(tagged))
        by_id = {r["id"]: r for r in results}
        results = [by_id[sq.question.id] for sq in ranked if sq.question.id in candidate_ids]
    else:
        results = results[:50]  # avoid dumping all 1,925 with no query

    st.write(f"**{len(results)} result(s)**" + (" (showing first 50 - type a query or add filters to narrow)" if not query and len(results) == 50 else ""))

    for r in results[:30]:
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{r['domain']} > {r['cb_skill']}** &nbsp;·&nbsp; {r['difficulty']} &nbsp;·&nbsp; `{r['id']}`")
                if r["auto_tags_skills"]:
                    st.caption("Tags: " + ", ".join(r["auto_tags_skills"]))
                snippet = (r["stem_text_partial"] + " " + r["answer_text_partial"]).strip()
                st.write(snippet[:220] + ("..." if len(snippet) > 220 else ""))
            with col2:
                st.checkbox("Add to worksheet", key=f"chk_{r['id']}")

            with st.expander("Preview original page"):
                img_bytes = get_page_image(resolve_pdf_path(r["source_pdf_filename"]), r["page_number"])
                st.image(img_bytes, use_container_width=True)
                pdf_bytes = extract_single_page_pdf(resolve_pdf_path(r["source_pdf_filename"]), r["page_number"])
                st.download_button(
                    "Download this question (PDF)",
                    data=pdf_bytes,
                    file_name=f"{r['id']}.pdf",
                    mime="application/pdf",
                    key=f"dl_{r['id']}",
                )


if __name__ == "__main__":
    main()
