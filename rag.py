"""
DocRAG - core pipeline
PDF parsing -> chunking -> embeddings -> vector store (Chroma)
Retrieval: hybrid search (BM25 + vector) with score fusion
Answering: Gemini with retrieved context
"""

import os
import re
import fitz  # PyMuPDF
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900        # chars
CHUNK_OVERLAP = 150     # chars
TOP_K = 5
ALPHA = 0.6             # weight for vector score in hybrid fusion


# ---------- parsing ----------

def parse_pdf(path: str) -> list[dict]:
    """Extract text page by page. Returns [{page, text}]."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


# ---------- chunking ----------

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Sliding-window chunking with overlap, keeps page metadata."""
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if len(chunk) > 50:  # skip tiny fragments
                chunks.append({"page": p["page"], "text": chunk})
            start = end - CHUNK_OVERLAP
    return chunks


# ---------- indexing ----------

class Index:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.Client()
        self.collection = None
        self.chunks: list[dict] = []
        self.bm25 = None

    def build(self, chunks: list[dict], name: str = "doc"):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # vector index
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.create_collection(name)
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(
            ids=[str(i) for i in range(len(texts))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": c["page"]} for c in chunks],
        )

        # bm25 index
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    # ---------- hybrid retrieval ----------

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Hybrid search: normalize BM25 and vector scores, fuse with ALPHA."""
        n = len(self.chunks)
        if n == 0:
            return []

        # vector scores (cosine distance -> similarity)
        q_emb = self.embedder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=n)
        vec_scores = {}
        for id_, dist in zip(res["ids"][0], res["distances"][0]):
            vec_scores[int(id_)] = 1.0 - dist

        # bm25 scores
        bm25_raw = self.bm25.get_scores(query.lower().split())

        def normalize(d: dict) -> dict:
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return {k: 0.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        vec_n = normalize(vec_scores)
        bm_n = normalize({i: s for i, s in enumerate(bm25_raw)})

        fused = {
            i: ALPHA * vec_n.get(i, 0) + (1 - ALPHA) * bm_n.get(i, 0)
            for i in range(n)
        }
        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [
            {
                "text": self.chunks[i]["text"],
                "page": self.chunks[i]["page"],
                "score": round(fused[i], 3),
            }
            for i in ranked
        ]


# ---------- answering ----------

def answer(query: str, retrieved: list[dict]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")

    context = "\n\n".join(
        f"[page {r['page']}]\n{r['text']}" for r in retrieved
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite page numbers. If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    )
    return model.generate_content(prompt).text
