## Architecture

The platform follows a **clean / hexagonal architecture** to keep domain logic isolated from frameworks and infrastructure concerns.

### Layers

- **Domain layer**
  - Located primarily in `models/` and within `ml_pipeline` / `rag`.
  - Contains business entities and value objects such as `DatasetMetadata`, `ModelMetadata`, and `DriftReportDomain`.
  - Free of FastAPI, SQLAlchemy, or external libraries where possible.

- **Application layer**
  - Implemented in `backend/app/services/`, `ml_pipeline/`, and `rag/`.
  - Encapsulates use cases:
    - `MLService`: training, prediction, EDA, drift detection.
    - `RAGService`: document ingestion, retrieval, recommendation generation.
  - Coordinates domain objects and calls into infrastructure via clear interfaces.

- **Infrastructure layer**
  - Implemented in `backend/app/infrastructure/`.
  - Concerns:
    - `db/`: SQLAlchemy engine, sessions, ORM models, repository classes.
    - `mlflow_client.py`: MLflow integration.
    - `llm_client.py`: LangChain-based LLM and retrieval chain builder.
  - Responsible for persistence, external services, and integration details.

- **Interface layer**
  - Implemented in `backend/app/api/`.
  - FastAPI routers (`ml.py`, `rag.py`, `system.py`) with:
    - Pydantic schemas for request/response models.
    - HTTP-specific concerns (status codes, error mapping, CORS, OpenAPI docs).

### Cross-Cutting Concerns

- **Configuration**
  - Centralized in `backend/app/core/config.py` (Pydantic `BaseSettings`).
  - Environment variables control database URL, MLflow, FAISS index path, LLM provider and model name, logging level, and allowed origins.

- **Logging**
  - Configured via `backend/app/core/logging_config.py`.
  - Uses `dictConfig` to create structured console logs.
  - Request ID middleware in `backend/app/main.py` injects a `request_id` into logs to support correlation.

- **Error Handling**
  - Domain-level exceptions defined in `backend/app/core/exceptions.py`.
  - FastAPI routers convert domain errors into appropriate HTTP responses.

### Data Flow (ML Pipeline)

1. **Dataset upload** (`POST /api/ml/datasets/upload`)
   - CSV stored on disk under `DATA_DIR`.
   - Metadata stored in the `datasets` table.
   - Problem type inferred (classification vs regression).
2. **EDA** (`GET /api/ml/datasets/{id}/eda`)
   - Loads the CSV and computes summary statistics, missingness, and target distribution.
3. **Training** (`POST /api/ml/train`)
   - Loads dataset from disk.
   - Splits into train/validation sets.
   - Builds preprocessing (`ColumnTransformer`) + model `Pipeline`s.
   - Trains multiple algorithms and evaluates them.
   - Logs metrics and the best model to MLflow; stores model metadata in the `models` table.
4. **Prediction** (`POST /api/ml/predict`)
   - Loads the current production model from MLflow.
   - Applies the same preprocessing pipeline to request payloads.
   - Returns predictions, optional probabilities, and optional SHAP explanations.
5. **Drift Detection**
   - Compares new data to the original training data using PSI per numeric feature.
   - Stores reports in `drift_reports`.

### Data Flow (RAG)

1. **Document ingestion** (`POST /api/rag/documents`)
   - Stores document metadata and chunks in `documents` and `document_chunks`.
   - Chunks are embedded and added to the FAISS index with metadata linking back to the source.
2. **Recommendation** (`POST /api/rag/recommend`)
   - Retrieves top-k relevant chunks using FAISS-based retriever.
   - Passes context and question into a LangChain `RetrievalQA` chain.
   - Returns an LLM-generated answer plus the IDs of the underlying documents.

