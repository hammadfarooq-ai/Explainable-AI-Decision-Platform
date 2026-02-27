## RAG Pipeline

The Retrieval-Augmented Generation (RAG) component provides contextual business recommendations based on ingested documents and user questions.

### Components

- **Ingestion (`rag/ingestion.py`)**
  - Uses `RecursiveCharacterTextSplitter` to chunk raw text into overlapping segments.
  - Chunk size and overlap are configurable and optimized for downstream embedding and retrieval.

- **Vector Store (`rag/vectorstore_faiss.py`)**
  - Uses a FAISS index to store and search embeddings.
  - Embeddings are generated using `OpenAIEmbeddings` from `langchain-openai`.
  - The index is persisted to disk in `FAISS_INDEX_DIR` (configured via environment variables).
  - Functions:
    - `get_vector_store()`: returns a singleton FAISS store, creating or loading it from disk.
    - `add_document(document_id, chunks)`: embeds and adds document chunks with `document_id` in metadata.
    - `as_retriever()`: exposes a LangChain-compatible retriever interface for search.

- **LLM Chain (`rag/chains.py` and backend LLM client)**
  - `get_default_llm()` provides a `ChatOpenAI` model configured with `LLM_MODEL_NAME`.
  - `backend/app/infrastructure/llm_client.py` defines `LLMClient`, which:
    - Builds a `RetrievalQA` chain combining an LLM, retriever, and a tailored prompt.
    - Can summarize source documents for logging.

- **Persistence**
  - Document metadata and chunk text are stored in PostgreSQL:
    - `documents` and `document_chunks` tables (see `backend/app/infrastructure/db/models.py`).
  - Recommendation logs are stored in `recommendation_logs` for auditing and future analysis.

### Workflow

1. **Document Ingestion**
   - API: `POST /api/rag/documents`.
   - Request includes a title, raw text, and optional source/tags.
   - The `RAGService`:
     - Persists the document and its chunks.
     - Adds chunks to the FAISS index using `add_document`.

2. **Recommendation Generation**
   - API: `POST /api/rag/recommend`.
   - Request includes a business question and optionally a `model_id`.
   - The `RAGService`:
     - Builds a retriever from the FAISS store.
     - Uses the default LLM and `LLMClient` to construct a `RetrievalQA` chain.
     - Invokes the chain with the question, retrieving and using relevant chunks as context.
     - Returns the generated answer and IDs of the underlying documents.
     - Logs a summary of the context and response for traceability.

### Configuration

- Key environment variables:
  - `FAISS_INDEX_DIR` – directory where the FAISS index is stored.
  - `LLM_PROVIDER`, `LLM_MODEL_NAME`, and provider-specific API keys (e.g., `OPENAI_API_KEY`).
- These values are surfaced via:
  - `backend/app/core/config.py` (`Settings`).
  - `rag/config.py` (`RAGSettings`).

