"""
DocRAG - core pipeline
PDF parsing -> chunking -> embeddings -> vector store (Chroma)
Retrieval: hybrid search (BM25 + vector) with score fusion
Answering: Gemini with retrieved context
"""

import os
import re
import uuid
import weakref
from functools import lru_cache

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
            if end >= len(text):
                break  # the next window would only be overlap we already have
            start = end - CHUNK_OVERLAP
    return chunks


# ---------- indexing ----------

def tokenize(text: str) -> list[str]:
    """Word tokens for BM25, so "stvec?" and "(stvec)" both match "stvec"."""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process instead of on every index."""
    return SentenceTransformer(EMBED_MODEL)


def normalize(scores: dict) -> dict:
    """Min-max scale scores to 0..1."""
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _drop_collection(client, name: str):
    try:
        client.delete_collection(name)
    except Exception:
        pass


class Index:
    def __init__(self):
        self.embedder = get_embedder()
        # All chromadb.Client()s in one process share storage, so every Index
        # needs its own collection name or two users overwrite each other.
        self.client = chromadb.Client()
        self.name = f"doc-{uuid.uuid4().hex}"
        self.collection = None
        self.chunks: list[dict] = []
        self.bm25 = None

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # vector index (Chroma defaults to L2, we want cosine)
        self.collection = self.client.create_collection(
            self.name, metadata={"hnsw:space": "cosine"}
        )
        weakref.finalize(self, _drop_collection, self.client, self.name)
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(
            ids=[str(i) for i in range(len(texts))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": c["page"]} for c in chunks],
        )

        # bm25 index
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    # ---------- hybrid retrieval ----------

    def vector_scores(self, query: str) -> dict:
        """Cosine similarity between the query and every chunk, by chunk id."""
        q_emb = self.embedder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=len(self.chunks))
        return {int(i): 1.0 - d for i, d in zip(res["ids"][0], res["distances"][0])}

    def bm25_scores(self, query: str) -> dict:
        return dict(enumerate(self.bm25.get_scores(tokenize(query))))

    def search(self, query: str, top_k: int = TOP_K, alpha: float = ALPHA) -> list[dict]:
        """Hybrid search: normalize both scores, then alpha * vector + (1 - alpha) * bm25."""
        if not self.chunks:
            return []

        vec_n = normalize(self.vector_scores(query))
        bm_n = normalize(self.bm25_scores(query))
        fused = {
            i: alpha * vec_n.get(i, 0) + (1 - alpha) * bm_n.get(i, 0)
            for i in range(len(self.chunks))
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
    model = genai.GenerativeModel("gemini-3.5-flash")

    context = "\n\n".join(
        f"[page {r['page']}]\n{r['text']}" for r in retrieved
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite page numbers. If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    )
    return model.generate_content(prompt).text
