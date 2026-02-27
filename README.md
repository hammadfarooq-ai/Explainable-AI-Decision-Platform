## Enterprise AI Decision Platform

Production-grade reference implementation for an end-to-end AI decision platform. It combines automated tabular ML (training, selection, explainability, drift) with a RAG pipeline for business recommendations, exposed via a FastAPI backend and a basic React frontend.

### 1. Features

- **Automated ML pipeline**
  - CSV upload and storage.
  - Automated EDA (basic stats, missingness, target distribution).
  - Feature engineering using `ColumnTransformer`.
  - Automatic problem type detection (classification vs regression).
  - Multiple algorithms: Logistic/Linear Regression, Random Forest, XGBoost, LightGBM, simple PyTorch MLP.
  - Cross-validation and metric computation.
  - Hyperparameter tuning scaffolding (Optuna).
  - Model comparison and best-model selection.
  - SHAP-based explainability (global/per-record).
  - MLflow model tracking and loading for inference.
  - Data drift detection via PSI.

- **RAG pipeline**
  - Document ingestion (text).
  - Text chunking.
  - FAISS vector store with OpenAI embeddings.
  - LangChain-based LLM chain for business recommendations.

- **Platform**
  - FastAPI backend with typed schemas.
  - PostgreSQL persistence via SQLAlchemy.
  - Basic React frontend (Vite) to exercise key flows.
  - Dockerized (backend, frontend, Postgres, MLflow).
  - Centralized config, structured logging, and basic tests.

### 2. Project structure

- `backend/`: FastAPI application (API, services, infrastructure, config).
- `ml_pipeline/`: ML components (EDA, feature engineering, models, training, drift, SHAP).
- `rag/`: RAG pipeline (ingestion, FAISS vector store, LLM chain).
- `models/`: Domain entities and events.
- `frontend/`: React + Vite SPA.
- `docker/`: Dockerfiles and `docker-compose.yml`.
- `tests/`: Pytest-based tests for ML, RAG, and backend.
- `data/`: Example datasets and runtime data.
- `config/`: Shared configuration glue.

### 3. Prerequisites

- **Python**: 3.11+
- **Node.js**: 18+ (for frontend)
- **Docker & Docker Compose** (for containerized deployment)
- OpenAI API key (for embeddings + LLM in the RAG pipeline) or swap to another LangChain-compatible provider.

### 4. Python setup (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` from the example:

```bash
cp .env.example .env
```

Adjust values as needed (`DATABASE_URL`, `MLFLOW_TRACKING_URI`, `OPENAI_API_KEY`, etc.).

### 5. Running services with Docker

From the `docker/` directory:

```bash
cd docker
docker compose up --build
```

Services:

- **Backend**: `http://localhost:8000`
- **API docs**: `http://localhost:8000/api/docs`
- **MLflow UI**: `http://localhost:5000`
- **Frontend**: `http://localhost:5173`

### 6. Running backend locally (without Docker)

Ensure PostgreSQL and MLflow are running and `DATABASE_URL`/`MLFLOW_TRACKING_URI` are correctly configured in `.env`, then:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Core API endpoints

Base prefix: `http://localhost:8000/api`

- **System**
  - `GET /system/health` – health check.
  - `GET /system/config` – non-sensitive configuration.

- **ML pipeline**
  - `POST /ml/datasets/upload` – upload CSV (`multipart/form-data`) with `target_column` query param.
  - `GET /ml/datasets/{dataset_id}/eda` – EDA summary.
  - `POST /ml/train` – trigger training and selection for a dataset.
  - `POST /ml/predict` – predict (and optional SHAP explanations) for JSON records.
  - `GET /ml/drift/{dataset_id}` – placeholder: drift calculation is available via the service; you can extend the API for batch uploads.

- **RAG**
  - `POST /rag/documents` – ingest a text document.
  - `GET /rag/documents` – list documents.
  - `POST /rag/recommend` – ask a business question and get recommendations.

Refer to the interactive FastAPI docs at `/api/docs` for full request/response schemas.

### 8. Frontend usage

From `frontend/`:

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`:

- Upload a CSV and specify the target column.
- Inspect the JSON EDA summary.
- Call the prediction endpoint with a JSON payload.
- Ingest a small text document and ask a question via the RAG section.

### 9. Testing

Run tests with:

```bash
pytest
```

This covers:

- Problem-type detection and drift logic (`ml_pipeline`).
- Basic RAG chunking.
- Health check endpoint.

### 10. Notes and extensions

- For production, add:
  - Proper database migrations (e.g., Alembic).
  - Authentication and authorization (e.g., OAuth2/JWT) around ML and RAG endpoints.
  - More robust error mapping and retry logic for external services.
  - Enhanced logging correlation (e.g., distributed tracing, log aggregation).
- To change LLM provider, update the RAG and LLM configuration (see `.env.example` and `rag/config.py`, `backend/app/infrastructure/llm_client.py`).

