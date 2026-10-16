# DocRAG
<img width="1470" height="831" alt="Screenshot 2026-08-11 at 9 22 34 PM" src="https://github.com/user-attachments/assets/3bc25998-a751-4b76-99b5-ebf31ca28424" />

Ask questions about a technical PDF and get answers with page numbers.

I got the idea while using LLMs to pull parameters out of the RISC-V Privileged Spec. Plain embedding search kept missing exact names like `stvec` or `MTVEC_MODE`, because to an embedding model those are basically noise. So DocRAG mixes keyword search (BM25) with vector search and shows you the chunks it picked and their scores.

## How it works

1. PyMuPDF pulls the text out page by page.
2. Each page gets cut into 900 character chunks with 150 characters of overlap. Chunks never cross pages, so every answer can point to a page.
3. Chunks are embedded locally with `all-MiniLM-L6-v2` and stored in Chroma. A BM25 index is built over the same chunks.
4. For a question, both scores get scaled to 0..1 and mixed: `0.6 * vector + 0.4 * bm25`. They're on totally different scales otherwise.
5. The top 5 chunks go to Gemini 3.5 Flash, which answers only from them and cites pages.

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
streamlit run app.py
```

Upload a PDF, hit index, ask away. Search works without a key, you just won't get the written answer.

## Tests

```bash
pip install pytest
pytest
```

They build a small RISC-V PDF on the fly, so no key needed.

## Todo

- rerank the top chunks with a cross-encoder
- OCR for scanned PDFs
- a small labeled question set to measure recall@k
