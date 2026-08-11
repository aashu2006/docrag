"""DocRAG - Streamlit UI. Run: streamlit run app.py"""

import os
import tempfile
import streamlit as st
from rag import parse_pdf, chunk_pages, Index, answer

st.set_page_config(page_title="DocRAG", page_icon="📄")
st.title("📄 DocRAG")
st.caption("Hybrid-search RAG over technical PDFs — BM25 + vector retrieval, Gemini answers")

if "index" not in st.session_state:
    st.session_state.index = None

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Index document"):
        with st.spinner("Parsing and indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                path = f.name
            pages = parse_pdf(path)
            chunks = chunk_pages(pages)
            idx = Index()
            idx.build(chunks)
            st.session_state.index = idx
            os.unlink(path)
        st.success(f"Indexed {len(pages)} pages → {len(chunks)} chunks")

    st.divider()
    st.markdown(
        "**Pipeline:** PyMuPDF parsing → sliding-window chunking → "
        "MiniLM embeddings → ChromaDB → hybrid retrieval "
        "(0.6·vector + 0.4·BM25) → Gemini answer with page citations"
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
                ans = answer(query, hits)
                st.markdown("### Answer")
                st.write(ans)
            except KeyError:
                st.error("Set GEMINI_API_KEY env var to enable answer generation.")
                ans = None

        st.markdown("### Retrieved chunks")
        for h in hits:
            with st.expander(f"page {h['page']} · score {h['score']}"):
                st.write(h["text"])
