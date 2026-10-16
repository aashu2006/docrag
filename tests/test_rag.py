"""Pipeline tests. They build a small RISC-V PDF on the fly, no API key needed."""

import fitz  # PyMuPDF
import pytest

from rag import CHUNK_SIZE, Index, chunk_pages, parse_pdf, tokenize

PAGES = [
    (
        "RISC-V Privileged Architecture - Trap Handling\n\n"
        "The MTVEC register (Machine Trap-Vector Base-Address Register) holds the base "
        "address of the trap vector table. The MTVEC_MODE field (bits [1:0]) determines "
        "how the processor routes interrupts:\n\n"
        "  MTVEC_MODE = 0 (Direct): All traps jump to BASE.\n"
        "  MTVEC_MODE = 1 (Vectored): Asynchronous interrupts jump to BASE + 4×cause.\n\n"
        "When a trap is taken in machine mode, the hardware writes the faulting PC to "
        "mepc and the cause to mcause. Software must inspect mcause to determine whether "
        "the trap was an interrupt or an exception."
    ),
    (
        "Supervisor-Level Trap Handling\n\n"
        "The supervisor mode uses a parallel set of CSRs. The stvec register is the "
        "Supervisor Trap-Vector Base-Address Register, functioning identically to mtvec "
        "but at S-mode privilege. The stvec register also supports Direct and Vectored "
        "modes via its MODE bits.\n\n"
        "When a trap is delegated to supervisor mode (via medeleg / mideleg), the "
        "processor writes the faulting PC to sepc and the cause to scause. The stvec "
        "base address must be aligned to at least a 4-byte boundary. The sstatus "
        "register's SIE bit globally enables or disables supervisor-level interrupts."
    ),
    (
        "Interrupt Priorities and Delegation\n\n"
        "RISC-V defines a fixed priority ordering among interrupt sources. Machine-level "
        "interrupts have the highest priority, followed by supervisor-level interrupts. "
        "Within a privilege level, external interrupts have higher priority than software "
        "interrupts, which in turn have higher priority than timer interrupts.\n\n"
        "The medeleg and mideleg CSRs allow machine mode to delegate specific exceptions "
        "and interrupts to supervisor mode. When a bit in mideleg is set, the "
        "corresponding interrupt will be handled by stvec instead of mtvec. This "
        "delegation mechanism is critical for efficient OS-level trap handling."
    ),
]


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "riscv.pdf"
    doc = fitz.open()
    for text in PAGES:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(50, 50, 545, 792), text, fontsize=11)
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture(scope="module")
def index(pdf_path):
    idx = Index()
    idx.build(chunk_pages(parse_pdf(pdf_path)))
    return idx


# ---------- parsing + chunking ----------

def test_parse_pdf_keeps_page_numbers(pdf_path):
    pages = parse_pdf(pdf_path)
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "stvec" in pages[1]["text"]


def test_short_page_is_one_chunk():
    # 850 chars used to give a second chunk that was pure overlap
    chunks = chunk_pages([{"page": 1, "text": "a" * 850}])
    assert len(chunks) == 1


def test_long_page_chunks_overlap_and_keep_page():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_pages([{"page": 7, "text": text}])
    assert len(chunks) > 1
    assert all(c["page"] == 7 for c in chunks)
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)
    assert chunks[0]["text"][-100:] in chunks[1]["text"]


# ---------- retrieval ----------

def test_tokenize_drops_punctuation():
    assert tokenize("What is (stvec)?") == ["what", "is", "stvec"]
    assert tokenize("MTVEC_MODE = 1") == ["mtvec_mode", "1"]


def test_bm25_matches_through_punctuation(index):
    scores = index.bm25_scores("what is stvec?")
    best = max(scores, key=scores.get)
    assert scores[best] > 0
    assert index.chunks[best]["page"] == 2


def test_search_finds_the_right_page(index):
    hits = index.search("what is stvec?")
    assert hits[0]["page"] == 2


def test_vector_scores_are_cosine(index):
    text = index.chunks[0]["text"]
    scores = index.vector_scores(text)
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert all(-1.0 <= s <= 1.0001 for s in scores.values())


def test_alpha_one_is_pure_vector(index):
    query = "how does the processor route interrupts"
    vec = index.vector_scores(query)
    expected = sorted(vec, key=vec.get, reverse=True)
    hits = index.search(query, top_k=len(index.chunks), alpha=1.0)
    assert [h["text"] for h in hits] == [index.chunks[i]["text"] for i in expected]


def test_alpha_zero_is_pure_bm25(index):
    hits = index.search("MTVEC_MODE", alpha=0.0)
    assert hits[0]["page"] == 1


def test_two_indexes_dont_clash(index):
    other = Index()
    other.build([{"page": 1, "text": "Completely unrelated text about baking sourdough bread."}])
    hits = index.search("stvec")
    assert len(hits) == 3
    assert all("sourdough" not in h["text"] for h in hits)
    assert other.search("bread")[0]["text"].startswith("Completely")


def test_empty_index_returns_nothing():
    assert Index().search("anything") == []
