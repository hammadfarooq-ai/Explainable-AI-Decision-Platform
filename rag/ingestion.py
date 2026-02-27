from __future__ import annotations

from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, *, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split raw text into overlapping chunks suitable for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)

