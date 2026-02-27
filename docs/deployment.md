## Deployment and Operations

This document describes how to deploy and operate the Enterprise AI Decision Platform in a containerized environment.

### Containers and Services

The reference deployment is defined in `docker/docker-compose.yml` and includes:

- **db**
  - Image: `postgres:15`.
  - Purpose: primary relational database for application metadata (datasets, models, documents, drift reports).
  - Exposed port: `5432`.

- **mlflow**
  - Image: `ghcr.io/mlflow/mlflow`.
  - Purpose: ML experiment tracking and model registry.
  - Uses the `db` service as backend store.
  - Exposed port: `5000`.

- **backend**
  - Build: `docker/Dockerfile.backend`.
  - Purpose: FastAPI application exposing ML and RAG APIs.
  - Exposed port: `8000`.
  - Environment:
    - `DATABASE_URL` – points at Postgres (`db` service).
    - `MLFLOW_TRACKING_URI` – points at `mlflow` service.
    - `DATA_DIR`, `MODELS_DIR`, `FAISS_INDEX_DIR` – internal filesystem paths.

- **frontend**
  - Build: `docker/Dockerfile.frontend`.
  - Purpose: React (Vite) SPA to exercise the platform.
  - Exposed port: `5173`.

### Running with Docker Compose

From the `docker/` directory:

```bash
cd docker
docker compose up --build
```

After a successful startup:

- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/api/docs`
- Frontend: `http://localhost:5173`
- MLflow UI: `http://localhost:5000`

### Environment Configuration

- The backend loads configuration from environment variables via `backend/app/core/config.py`.
- For local development, `.env.example` can be copied to `.env` and customized.
- In containerized environments, prefer setting environment variables directly on services.

Key variables:

- `ENV`, `DEBUG`
- `DATABASE_URL`
- `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACTS_URI` (optional)
- `DATA_DIR`, `MODELS_DIR`, `FAISS_INDEX_DIR`
- `LLM_PROVIDER`, `LLM_MODEL_NAME`, and provider-specific API keys (e.g., `OPENAI_API_KEY`).

### Production Considerations

- **Database**
  - Replace the local Postgres container with a managed instance.
  - Use migrations (e.g., Alembic) instead of `Base.metadata.create_all`.

- **MLflow**
  - Use a persistent backend store and artifact store (e.g., S3-compatible storage).

- **Secrets Management**
  - Store sensitive values (DB passwords, API keys) in a secure secrets manager.

- **Logging & Monitoring**
  - Route logs to a centralized system (e.g., ELK, Loki, or cloud-native logging).
  - Add metrics and traces via an observability stack if required.

