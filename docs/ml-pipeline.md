## ML Pipeline

This document describes the end-to-end machine learning workflow implemented in `ml_pipeline/` and orchestrated by `MLService`.

### Stages

1. **Data Ingestion**
   - Implemented in `ml_pipeline/data_ingestion.py`.
   - `load_csv(path: Path) -> DataFrame` loads a CSV file into a pandas `DataFrame`.

2. **Automated EDA**
   - Implemented in `ml_pipeline/eda.py`.
   - `compute_eda_summary(df, target_column)` returns:
     - `basic_stats`: `DataFrame.describe(include="all")` output converted to a serializable dict.
     - `missingness`: fraction of missing values per column.
     - `target_distribution`: numeric summary or normalized value counts for classification targets.
   - Exposed via `GET /api/ml/datasets/{dataset_id}/eda`.

3. **Problem Type Detection**
   - Implemented in `ml_pipeline/problem_detection.py`.
   - Heuristic:
     - If target is numeric and has many unique values ⇒ regression.
     - Otherwise ⇒ classification.
   - Used when a dataset is first registered during upload.

4. **Feature Engineering**
   - Implemented in `ml_pipeline/feature_engineering.py`.
   - Builds a `ColumnTransformer` that:
     - For numeric columns: median imputation + standard scaling.
     - For categorical columns: most-frequent imputation + one-hot encoding.
   - Returned preprocessor is attached as the first step in an sklearn `Pipeline`.

5. **Model Candidates**
   - Implemented across:
     - `ml_pipeline/models_sklearn.py`
     - `ml_pipeline/models_xgboost.py`
     - `ml_pipeline/models_lightgbm.py`
     - `ml_pipeline/models_torch.py` (MLP definition for future use)
   - Classification models:
     - Logistic Regression.
     - Random Forest.
     - XGBoost.
     - LightGBM.
   - Regression models:
     - Linear Regression.
     - Random Forest Regressor.
     - XGBoost Regressor.
     - LightGBM Regressor.

6. **Training and Selection**
   - Implemented in `ml_pipeline/training.py`.
   - `train_and_select_models`:
     - Splits data into train/validation (stratified for classification).
     - Trains each candidate inside a `Pipeline(preprocessor -> model)`.
     - Computes metrics via `ml_pipeline/evaluation.py`.
     - Chooses a primary metric:
       - Classification: `f1` (weighted).
       - Regression: `rmse`.
     - Selects the best-performing pipeline based on this metric.
   - `MLService.train_models` wraps this logic and logs metrics plus the best model into MLflow.

7. **Hyperparameter Tuning (Optuna)**
   - Scaffolding in `ml_pipeline/tuning.py`.
   - `tune_model` accepts a model-building callback, data, and `TrainingConfig`.
   - The current core training path uses fixed hyperparameters; you can wire `tune_model` into the loop for more exhaustive model search.

8. **Evaluation Metrics**
   - Implemented in `ml_pipeline/evaluation.py`.
   - Classification:
     - Accuracy, weighted F1.
     - AUC (for binary problems with probability estimates).
   - Regression:
     - RMSE, MAE, R².

9. **Explainability**
   - Implemented in `ml_pipeline/explainability.py`.
   - `compute_shap_for_records(model, df)`:
     - Uses `shap.TreeExplainer` for tree-based models when possible.
     - Falls back to generic `shap.Explainer` otherwise.
     - Returns SHAP values, mean base value, and feature names.
   - Surfaced via the prediction endpoint when `explain=true`.

10. **Drift Detection**
    - Implemented in `ml_pipeline/drift.py`.
    - Computes a per-feature Population Stability Index (PSI) for numeric columns.
    - Returns a dictionary of `{"feature": {"psi": value}}`.
    - `MLService.compute_drift` compares new data against training data and persists a report through the `DriftReportRepository`.

