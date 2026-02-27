from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import get_rag_settings

_VECTOR_STORE: FAISS | None = None


def _build_embeddings() -> OpenAIEmbeddings:
    # This relies on OPENAI_API_KEY in the environment; documented in README.
    return OpenAIEmbeddings()


def _index_path() -> Path:
    settings = get_rag_settings()
    path = Path(settings.faiss_index_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / "rag_index.faiss"


def get_vector_store() -> FAISS:
    """Return a singleton FAISS vector store, loading from disk if present."""

    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    embeddings = _build_embeddings()
    index_path = _index_path()

    if index_path.exists():
        _VECTOR_STORE = FAISS.load_local(
            str(index_path.parent),
            embeddings,
            index_name=index_path.stem,
            allow_dangerous_deserialization=True,
        )
    else:
        index = faiss.IndexFlatL2(1536)  # default OpenAI embedding dimension
        _VECTOR_STORE = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        _VECTOR_STORE.save_local(str(index_path.parent), index_name=index_path.stem)

    return _VECTOR_STORE


def persist_vector_store() -> None:
    """Persist the current FAISS store to disk."""

    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        return
    index_path = _index_path()
    _VECTOR_STORE.save_local(str(index_path.parent), index_name=index_path.stem)


def as_retriever():
    """Convenience method for the default retriever used in services."""

    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": 5})


def add_document(document_id: int, chunks: List[str]) -> None:
    """Add a document's chunks to the FAISS index with metadata."""

    store = get_vector_store()
    docs = [
        Document(page_content=chunk, metadata={"document_id": document_id})
        for chunk in chunks
    ]
    store.add_documents(docs)
    persist_vector_store()

