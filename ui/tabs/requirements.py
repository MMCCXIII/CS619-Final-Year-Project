import pandas as pd
import streamlit as st


@st.fragment
def render_requirements_tab() -> None:
    st.subheader("Requirement coverage")
    st.caption("See docs/requirements_traceability.md for full mapping.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Requirement": "Data collection and preprocessing",
                    "Implementation": "CSV dataset, SQLite seed pipeline, symptom normalization, age features",
                    "Traceability": "docs/requirements_traceability.md",
                },
                {
                    "Requirement": "Model training and testing",
                    "Implementation": "Train/test split, Naive Bayes, Logistic Regression, Random Forest, Neural Network MLP",
                    "Traceability": "docs/requirements_traceability.md",
                },
                {
                    "Requirement": "Fine tuning and optimization",
                    "Implementation": "GridSearchCV for Logistic Regression and Random Forest, weighted F1 model selection",
                    "Traceability": "docs/requirements_traceability.md",
                },
                {
                    "Requirement": "Recommendation engine",
                    "Implementation": "Disease confidence, medicine relevance, effectiveness, contraindication penalties, feedback down-ranking",
                    "Traceability": "docs/requirements_traceability.md",
                },
                {
                    "Requirement": "Web-based user interface",
                    "Implementation": "Streamlit app with profile persistence, export, cached queries, fragments, and safety UX",
                    "Traceability": "docs/requirements_traceability.md",
                },
                {
                    "Requirement": "Complete application package",
                    "Implementation": "Code, dataset, database schema, scripts, tests, docs, reproducible setup, CI workflow",
                    "Traceability": "docs/requirements_traceability.md",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
