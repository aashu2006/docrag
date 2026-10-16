"""DocRAG - Streamlit UI. Run: streamlit run app.py"""

import os
import re
import tempfile
import streamlit as st
from rag import parse_pdf, chunk_pages, Index, answer

st.set_page_config(page_title="DocRAG")
st.title("DocRAG")
st.caption("Hybrid-search RAG over technical PDFs: BM25 + vector retrieval, Gemini answers")

if "index" not in st.session_state:
    st.session_state.index = None


# Streamlit reruns the whole script on every click, so without this the same
# question gets sent to Gemini again each time.
@st.cache_data(max_entries=200, show_spinner=False)
def cached_answer(query: str, hits: list[dict]) -> str:
    return answer(query, hits)

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Index document"):
        with st.spinner("Parsing and indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                path = f.name
            try:
                pages = parse_pdf(path)
                chunks = chunk_pages(pages)
                idx = Index()
                idx.build(chunks)
            finally:
                os.unlink(path)
            st.session_state.index = idx
        st.success(f"Indexed {len(pages)} pages → {len(chunks)} chunks")

    st.divider()
    st.markdown("**How it works**")
    st.caption(
        "1. Reads the PDF page by page\n"
        "2. Splits each page into overlapping chunks\n"
        "3. Scores chunks by keywords (BM25) and by meaning (MiniLM embeddings)\n"
        "4. Mixes the two: 60% meaning, 40% keywords\n"
        "5. Gemini answers from the top 5 chunks and cites pages"
    )

query = st.text_input("Ask a question about the document")

if query:
    if st.session_state.index is None:
        st.warning("Upload and index a PDF first.")
    else:
        with st.spinner("Retrieving..."):
            hits = st.session_state.index.search(query)
        with st.spinner("Generating answer..."):
            try:
                ans = cached_answer(query, hits)
                st.markdown("### Answer")
                st.write(ans)
            except KeyError:
                st.error("Set GEMINI_API_KEY env var to enable answer generation.")
            except Exception as exc:
                # error text can include the request URL, and the key is in it
                msg = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "<key>", str(exc))
                st.error(f"Couldn't get an answer from Gemini: {msg}")

        st.markdown("### Retrieved chunks")
        for h in hits:
            with st.expander(f"page {h['page']} · score {h['score']}"):
                st.write(h["text"])
