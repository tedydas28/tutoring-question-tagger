"""
Search UI for the tagged SAT question bank.

Run with:
    streamlit run app.py

Type what you're looking for (e.g. "similar triangles word problems",
"margin of error", "graph") and get back real questions - each with a
preview of the EXACT original PDF page and a one-click download of that
page as a standalone PDF. Select several (or a whole page of results at
once) and download them together as one worksheet.
"""

import json

import streamlit as st

from src.embeddings import SemanticSearchIndex
from src.pdf_extract import (
    extract_multiple_pages_pdf,
    extract_page_from_reader,
    open_pymupdf,
    open_pypdf_reader,
    render_page_image_from_doc,
)
from src.schema import Question, Taxonomy
from src.config import resolve_pdf_path

st.set_page_config(page_title="SAT Question Bank Search", layout="wide")

PAGE_SIZE = 30

CUSTOM_CSS = """
<style>
html, body, [class*="css"]  { font-size: 19px; }
h1 { font-size: 2.1rem !important; }
.stCheckbox label p { font-size: 19px !important; }
</style>
"""


@st.cache_data
def load_data():
    taxonomy = Taxonomy.from_yaml("data/taxonomy.yaml")
    with open("data/tagged_real_questions.json") as f:
        tagged = json.load(f)
    return taxonomy, tagged


FORMAT_SEARCH_LABELS = {
    "graph_based": "graph plotted coordinate plane line",
    "word_problem": "word problem",
    "diagram_based": "diagram figure",
    "table_based": "table data",
    "multi_step": "multi step",
    "conceptual": "conceptual",
    "straightforward": "straightforward direct",
}


@st.cache_data
def build_search_index(tagged: list[dict]):
    """
    Haystack per question = extracted text + official CB skill name +
    fine-grained tags + plain-English format labels (e.g. "graph" instead
    of the token "graph_based") so free-text search actually matches them.
    """
    docs = []
    for r in tagged:
        format_labels = " ".join(FORMAT_SEARCH_LABELS.get(f, f) for f in r["auto_tags_formats"])
        haystack_parts = [
            r["stem_text_partial"],
            r["answer_text_partial"],
            r["cb_skill"],
            r["domain"],
            " ".join(r["auto_tags_skills"]),
            format_labels,
        ]
        docs.append(Question(id=r["id"], text=" ".join(p for p in haystack_parts if p)))
    return SemanticSearchIndex(docs)


@st.cache_data
def get_page_image(pdf_path: str, page_number: int) -> bytes:
    return render_page_image(pdf_path, page_number)


@st.cache_data
def render_page_batch(item_keys: tuple[tuple[str, str, int], ...]) -> dict:
    """
    item_keys: tuple of (question_id, source_filename, page_number) for
    every question on the current results page.

    This is the actual fix for slow page loads: instead of opening a
    600+ page, 30MB source PDF from scratch for every single question
    (up to 30 full-file opens per page), each distinct source file gets
    opened exactly ONCE here, and every page needed from it is pulled
    from that one open handle. Cached by Streamlit, so revisiting the
    same page of results (e.g. after typing more into the search box,
    which triggers a full script rerun) doesn't redo any of this work.
    """
    from collections import defaultdict

    by_file = defaultdict(list)
    for qid, filename, page_number in item_keys:
        by_file[filename].append((qid, page_number))

    results = {}
    for filename, items in by_file.items():
        pdf_path = resolve_pdf_path(filename)
        doc = open_pymupdf(pdf_path)
        reader = open_pypdf_reader(pdf_path)
        try:
            for qid, page_number in items:
                img_bytes = render_page_image_from_doc(doc, page_number)
                pdf_bytes = extract_page_from_reader(reader, page_number)
                results[qid] = (img_bytes, pdf_bytes)
        finally:
            doc.close()
    return results


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("SAT Question Bank Search")

    taxonomy, tagged = load_data()
    index = build_search_index(tagged)
    by_id = {r["id"]: r for r in tagged}

    # Worksheet summary + download, shown at the top.
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
        selected_cb_skills = st.multiselect("College Board skill", cb_skills, default=[])

        only_graphs = st.checkbox("Only show questions with a real graph")

    query = st.text_input(
        "What kind of questions do you need?",
        placeholder="e.g. similar triangles word problems, margin of error, graph",
    )

    results = tagged
    if selected_domains:
        results = [r for r in results if r["domain"] in selected_domains]
    if selected_difficulties:
        results = [r for r in results if r["difficulty"] in selected_difficulties]
    if selected_cb_skills:
        results = [r for r in results if r["cb_skill"] in selected_cb_skills]
    if only_graphs:
        results = [r for r in results if "graph_based" in r["auto_tags_formats"]]

    if query:
        candidate_ids = {r["id"] for r in results}
        ranked = index.query(query, top_k=len(tagged))
        results_by_id = {r["id"]: r for r in results}
        results = [results_by_id[sq.question.id] for sq in ranked if sq.question.id in candidate_ids]

    # Reset to page 1 whenever the search/filters actually change.
    search_signature = (query, tuple(selected_domains), tuple(selected_difficulties), tuple(selected_cb_skills), only_graphs)
    if st.session_state.get("_search_signature") != search_signature:
        st.session_state["_search_signature"] = search_signature
        st.session_state["_page_num"] = 0

    total_matches = len(results)
    total_pages = max(1, (total_matches + PAGE_SIZE - 1) // PAGE_SIZE)
    page_num = st.session_state.get("_page_num", 0)
    page_num = max(0, min(page_num, total_pages - 1))
    st.session_state["_page_num"] = page_num

    page_results = results[page_num * PAGE_SIZE: (page_num + 1) * PAGE_SIZE]

    st.write(f"**{total_matches} result(s)** — page {page_num + 1} of {total_pages}")

    if page_results:
        select_all = st.checkbox(f"Select all {len(page_results)} on this page", key=f"select_all_{page_num}_{search_signature}")
        if select_all:
            for r in page_results:
                st.session_state[f"chk_{r['id']}"] = True

    item_keys = tuple((r["id"], r["source_pdf_filename"], r["page_number"]) for r in page_results)
    page_batch = render_page_batch(item_keys)

    for r in page_results:
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
                img_bytes, pdf_bytes = page_batch[r["id"]]
                st.image(img_bytes, use_container_width=True)
                st.download_button(
                    "Download this question (PDF)",
                    data=pdf_bytes,
                    file_name=f"{r['id']}.pdf",
                    mime="application/pdf",
                    key=f"dl_{r['id']}",
                )

    # Bottom pagination controls.
    st.divider()
    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Previous page", disabled=(page_num == 0)):
            st.session_state["_page_num"] = page_num - 1
            st.rerun()
    with col_mid:
        st.markdown(f"<p style='text-align:center'>Page {page_num + 1} of {total_pages}</p>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next page →", disabled=(page_num >= total_pages - 1)):
            st.session_state["_page_num"] = page_num + 1
            st.rerun()


if __name__ == "__main__":
    main()
