## API Reference

This document summarizes the main HTTP endpoints exposed by the FastAPI backend. All endpoints are prefixed with `/api` by default (configurable via `Settings.api_prefix`).

For interactive documentation, visit `/api/docs` (Swagger UI) or `/api/redoc`.

### System

- **GET `/api/system/health`**
  - Description: Health check endpoint for readiness/liveness probes.
  - Response: `{"status": "ok"}` on success.

- **GET `/api/system/config`**
  - Description: Returns a non-sensitive subset of configuration for debugging.
  - Response:
    - `project_name`, `env`, `debug`, `api_prefix`.

### ML Pipeline

- **POST `/api/ml/datasets/upload`**
  - Description: Upload a CSV dataset and register it in the system.
  - Request:
    - Content type: `multipart/form-data`.
    - Fields:
      - `file`: CSV file.
      - Query parameter `target_column`: name of the target column in the CSV.
  - Response:
    - `dataset_id`: integer.
    - `name`: original file name.
    - `n_rows`, `n_columns`: dataset shape.
    - `problem_type`: `"classification"` or `"regression"`.

- **GET `/api/ml/datasets/{dataset_id}/eda`**
  - Description: Retrieve automated EDA summary for a dataset.
  - Path parameter:
    - `dataset_id`: ID returned by the upload endpoint.
  - Response:
    - `dataset_id`.
    - `basic_stats`: summary statistics per column.
    - `missingness`: fraction of missing values per column.
    - `target_distribution`: distribution summary of the target column.

- **POST `/api/ml/train`**
  - Description: Trigger model training, evaluation, and best-model selection for a dataset.
  - Request body:
    - `dataset_id`: integer.
    - `target_column`: string.
    - `problem_type` (optional): `"classification"` or `"regression"` (overrides auto-detection).
  - Response:
    - `training_run_id`: integer.
    - `problem_type`: resolved problem type.
    - `best_model`: summary for the selected model.
    - `all_models`: list of model summaries (currently contains the selected model; can be extended).

- **POST `/api/ml/predict`**
  - Description: Generate predictions (and optional SHAP explanations) using the current production model.
  - Request body:
    - `model_id` (optional): if omitted, uses the production model.
    - `records`: array of feature dictionaries, each representing one row.
    - `explain` (bool, optional): if `true`, compute SHAP explanations for the records.
  - Response:
    - `model_id`: ID of the model used for prediction.
    - `predictions`: list of predicted values.
    - `probabilities` (optional): list of probability vectors for classification models.
    - `explanation` (optional):
      - `shap_values`: nested list of SHAP values.
      - `expected_value`: base value, if available.
      - `feature_names`: list of feature names.

### RAG

- **POST `/api/rag/documents`**
  - Description: Ingest a raw text document for use in the RAG pipeline.
  - Request body:
    - `title`: string.
    - `text`: raw text content.
    - `source` (optional): origin (URL, system name, etc.).
    - `tags` (optional): comma-separated tags.
  - Response:
    - `document`:
      - `id`, `title`, `source`, `tags`.
    - `n_chunks`: number of chunks created and added to the vector store.

- **GET `/api/rag/documents`**
  - Description: List documents that are available for retrieval.
  - Response:
    - `documents`: array of `{id, title, source, tags}`.

- **POST `/api/rag/recommend`**
  - Description: Generate business recommendations based on ingested documents.
  - Request body:
    - `question`: user question or decision context.
    - `model_id` (optional): ID of a predictive model whose context should be associated with this query.
  - Response:
    - `answer`: LLM-generated answer string.
    - `document_ids`: array of document IDs that contributed context.

