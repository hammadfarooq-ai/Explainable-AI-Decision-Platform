from __future__ import annotations

from typing import Any, List

from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel

from backend.app.core.config import get_settings


class LLMClient:
    """Wrapper around LangChain LLM and retriever for recommendations.

    This abstraction keeps FastAPI and the RAG module decoupled from concrete
    LLM providers. The RAG module is responsible for constructing the retriever.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def build_recommendation_chain(
        self,
        *,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt_template: str | None = None,
    ) -> RetrievalQA:
        """Return a RetrievalQA chain configured for business recommendations."""

        from langchain import prompts

        if prompt_template is None:
            prompt_template = (
                "You are an AI assistant helping with enterprise decisions.\n"
                "Use the following context from internal documents to provide "
                "actionable, concise business recommendations.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer with clear recommendations and, if relevant, risks and next steps."
            )

        prompt = prompts.PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return chain

    def summarize_sources(self, documents: List[Any], max_len: int = 200) -> str:
        """Return a short textual summary of retrieved documents for logging."""

        texts: list[str] = []
        for doc in documents:
            text = getattr(doc, "page_content", "")[:max_len]
            texts.append(text.replace("\n", " "))
        return " | ".join(texts)


llm_client = LLMClient()

