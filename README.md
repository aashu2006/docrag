# DocRAG
<img width="1470" height="831" alt="Screenshot 2026-08-11 at 9 22 34 PM" src="https://github.com/user-attachments/assets/3bc25998-a751-4b76-99b5-ebf31ca28424" />

RAG-based question answering over technical PDFs, with **hybrid retrieval (BM25 + vector search)**.

Built to explore how retrieval quality changes when lexical and semantic search are fused, motivated by earlier work using LLMs to extract implementation-defined parameters from the RISC-V Privileged Spec, where pure semantic retrieval often missed exact-term matches (register names, CSR fields).

## Pipeline

```
PDF → PyMuPDF parsing → sliding-window chunking (900 chars, 150 overlap)
    → MiniLM embeddings → ChromaDB (vector index)
    → BM25 index (rank_bm25)
    → hybrid retrieval: score = 0.6·vector + 0.4·BM25 (min-max normalized)
    → Gemini 3.5 Flash answer, grounded in retrieved chunks, with page citations
```

## Why hybrid search

Technical documents are full of exact identifiers (`stvec`, `MTVEC_MODE`, part numbers) that embedding models treat as noise. BM25 catches exact-term matches; vector search catches paraphrases and conceptual questions. Fusing both gives better recall than either alone.

## Run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key   # answers need this; retrieval works without it
streamlit run app.py
```

Upload a PDF, index it, ask questions. Retrieved chunks are shown with pages and fusion scores so retrieval quality is inspectable.

## Design choices

- **Chunking:** character-based sliding window with overlap. Simple, predictable, keeps page metadata for citations.
- **Embeddings:** `all-MiniLM-L6-v2`, runs locally, no API cost for indexing.
- **Fusion:** min-max normalization of both score distributions before weighted sum (α = 0.6 toward vector). Raw BM25 and cosine scores live on different scales, so normalization is required before fusing.
- **Grounding:** the LLM is instructed to answer only from context and cite page numbers, and to say when the context doesn't contain the answer.

## Next steps

- Reranking retrieved chunks with a cross-encoder
- OCR path for scanned PDFs (Tesseract)
- Retrieval evaluation harness (recall@k on a labeled question set)
