# Medicine Recommendation System

This project is a complete academic implementation of the guided CS619 Medicine Recommendation System. It includes a Streamlit web application, SQLite database schema and seed data, data preprocessing, multiple machine learning models, a neural-network model, hyperparameter tuning, ranking logic, test cases, and requirement traceability.

The application is for academic demonstration only. It does not diagnose, prescribe, or replace a doctor, pharmacist, emergency service, or local clinical guideline.

## Features

- Collects and preprocesses a reproducible 4,800-row structured synthetic dataset for 160 diseases, 195 medicines, and 358 safety rules.
- Trains and evaluates multiple models: Multinomial Naive Bayes, Logistic Regression, Random Forest, and Neural Network MLP.
- Runs hyperparameter tuning for selected models and saves metrics.
- Predicts likely conditions from entered symptoms.
- Ranks medicine suggestions using disease confidence, relevance, expected effectiveness, and patient safety constraints.
- Uses patient profile details such as age group, medical history, pregnancy, allergy, kidney disease, liver disease, gastritis, hypertension, diabetes, asthma, and anticoagulant use.
- Provides Streamlit screens for recommendation, database overview, model metrics, and requirement coverage.
- Persists safety profiles locally, exports session JSON reports, caches database reads, and down-ranks medicines with repeated irrelevant feedback.
- Includes SQLite schema, reproducible seed scripts, training scripts, and pytest tests.

## Project Structure

```text
app.py
assets/app.css
ui/
data/medicine_dataset.csv
database/schema.sql
docs/design_document.md
docs/dataset_notes.md
docs/requirements_traceability.md
docs/test_cases.md
models/disease_model.joblib
models/model_metrics.json
scripts/initialize_database.py
scripts/generate_professional_dataset.py
scripts/train_models.py
src/config.py
src/data_access.py
src/modeling.py
src/preprocessing.py
src/recommender.py
tests/test_recommender.py
```

## Setup

Install dependencies and create the database/model artifacts:

```powershell
uv sync --extra dev
uv run python scripts/generate_professional_dataset.py
uv run python scripts/initialize_database.py
uv run python scripts/train_models.py
```

Run the application (train models first on a fresh clone):

```powershell
uv run streamlit run app.py
```

Optional Streamlit settings live in `.streamlit/config.toml`. The first launch trains models automatically if artifacts are missing.

Run tests:

```powershell
uv run pytest
```

## Notes For Submission

The folder already contains the official guidance documents. This implementation adds the application code, database schema and seed pipeline, tests, and supporting documentation expected by the final application note in `1-Application.txt`.
