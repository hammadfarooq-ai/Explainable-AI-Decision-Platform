## Project Overview

The **Enterprise AI Decision Platform** is a reference implementation of a production-grade decisioning system that combines automated tabular machine learning with retrieval-augmented generation (RAG). It is designed following clean-architecture principles and is intended to be understandable, extensible, and operationally sound.

### Goals

- Provide an end-to-end pipeline from raw CSV data to a deployed, explainable prediction service.
- Offer RAG-based business recommendations driven by internal documents and model outputs.
- Demonstrate solid engineering practices: modularity, type hints, environment-driven configuration, structured logging, and containerization.

### High-Level Capabilities

- **ML Decisioning**
  - CSV upload and metadata management.
  - Automated EDA and problem-type detection.
  - Feature engineering and multi-model training.
  - Hyperparameter tuning scaffolding via Optuna.
  - Model comparison, selection, and MLflow-based tracking.
  - SHAP-based global and local explainability.
  - Data drift detection and reporting.

- **RAG Recommendations**
  - Document ingestion and text chunking.
  - FAISS vector store storage for document embeddings.
  - LangChain-based LLM pipeline for contextual business recommendations.

- **Platform & Operations**
  - FastAPI backend with typed schemas and OpenAPI docs.
  - PostgreSQL persistence via SQLAlchemy.
  - React (Vite) frontend for core workflows.
  - Dockerized runtime (backend, frontend, Postgres, MLflow).
  - Centralized configuration, logging, and basic tests.

