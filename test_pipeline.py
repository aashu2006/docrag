"""
Quick smoke test for the RAG pipeline (no API key needed).
Generates a small test PDF, indexes it, and runs a search.
"""

import os
import fitz  # PyMuPDF

# ---- generate a small technical PDF ----
PDF_PATH = "test_doc.pdf"

page_texts = [
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

doc = fitz.open()
for text in page_texts:
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 50, 545, 792), text, fontsize=11)
doc.save(PDF_PATH)
doc.close()
print(f"Created test PDF: {PDF_PATH} ({len(page_texts)} pages)")

# ---- run the pipeline ----
from rag import parse_pdf, chunk_pages, Index

pages = parse_pdf(PDF_PATH)
print(f"\nParsed {len(pages)} pages")
for p in pages:
    print(f"  page {p['page']}: {len(p['text'])} chars")

chunks = chunk_pages(pages)
print(f"\nChunked into {len(chunks)} chunks")

idx = Index()
idx.build(chunks)
print("Index built (vector + BM25)\n")

query = "what is stvec"
print(f'Searching: "{query}"\n')
results = idx.search(query)

for i, r in enumerate(results):
    print(f"--- Result {i+1} | page {r['page']} | score {r['score']} ---")
    print(r["text"][:200] + ("..." if len(r["text"]) > 200 else ""))
    print()

# cleanup
os.remove(PDF_PATH)
print("Test passed ✓")
