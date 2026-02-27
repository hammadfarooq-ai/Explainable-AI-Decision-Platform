from __future__ import annotations

from rag.ingestion import chunk_text


def test_chunk_text_produces_chunks():
    text = "This is a short document used for testing the RAG pipeline."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)

