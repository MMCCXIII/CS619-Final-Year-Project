# Requirements Traceability

| Guided requirement | Delivered artifact | Verification |
| --- | --- | --- |
| Data Collection and Pre-processing: collect and preprocess medicine-related datasets for symptoms, diseases, and prescriptions. | `data/medicine_dataset.csv` with 4,800 generated patient-case entities across 160 diseases, `src/knowledge_base.py`, `scripts/generate_professional_dataset.py`, `src/preprocessing.py`, `src/data_access.py`, `database/schema.sql` | `scripts/initialize_database.py`, `tests/test_recommender.py` |
| Model Training and Testing: train and test multiple machine learning and deep learning models. | `src/modeling.py` trains Naive Bayes, Logistic Regression, Random Forest, and Neural Network MLP. | `scripts/train_models.py`, saved metrics JSON |
| Model Fine-Tuning and Optimization: optimize models with hyperparameter tuning and reduce overfitting. | `GridSearchCV` for Logistic Regression and Random Forest; stratified train/test split; weighted F1 model selection. | `models/model_metrics.json` after training |
| Recommendation Engine: personalized medicine suggestions based on symptoms, medical history, and existing conditions. | `src/recommender.py` ranks medicines by disease confidence, relevance, effectiveness, prescription status, and contraindication penalties. | Streamlit recommendation tab and pytest coverage |
| Rank recommendations based on relevance and effectiveness. | `disease_medicines.relevance_score`, `medicines.effectiveness_score`, `rank_medicines`. | Recommendation output table sorted by score |
| Web-Based User Interface: interactive user-friendly web interface. | `app.py` Streamlit UI with patient input, results, data, model, and coverage tabs. | `uv run streamlit run app.py` |
| Complete application code, database, and important files. | Source package, scripts, dataset, schema, docs, tests, dependency files. | README setup and test commands |

## Safety Boundary

This is an academic recommender. It intentionally avoids dosage instructions and treats prescription medicines as clinician-review items. Emergency flags, contraindication penalties, and medical disclaimers are part of the application behavior.
