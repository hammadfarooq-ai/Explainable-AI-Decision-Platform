from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_rag_settings


def get_default_llm() -> ChatOpenAI:
    """Return a default ChatOpenAI LLM instance for recommendations."""

    settings = get_rag_settings()
    return ChatOpenAI(model=settings.llm_model_name, temperature=0.2)

