import pandas as pd
import streamlit as st

from src.data_access import fetch_dataframe


@st.cache_data(ttl=60, show_spinner=False)
def load_medicine_catalog() -> pd.DataFrame:
    return fetch_dataframe(
        """
        SELECT name, category, prescription_required, effectiveness_score, safety_notes
        FROM medicines
        ORDER BY name
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_history_summary() -> dict[str, int]:
    total_checks = int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_logs").iloc[0]["count"])
    total_feedback = int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_feedback").iloc[0]["count"])
    irrelevant_count = int(
        fetch_dataframe(
            "SELECT COUNT(*) AS count FROM recommendation_feedback WHERE irrelevant_medicine = 1"
        ).iloc[0]["count"]
    )
    return {
        "total_checks": total_checks,
        "total_feedback": total_feedback,
        "irrelevant_count": irrelevant_count,
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_recent_history(limit: int = 25) -> pd.DataFrame:
    return fetch_dataframe(
        f"""
        SELECT created_at, symptoms, age_group, profile_conditions, top_disease, model_confidence
        FROM recommendation_logs
        ORDER BY id DESC
        LIMIT {int(limit)}
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_recent_feedback(limit: int = 20) -> pd.DataFrame:
    return fetch_dataframe(
        f"""
        SELECT created_at, top_disease, medicine_name, rating, irrelevant_medicine, comments
        FROM recommendation_feedback
        ORDER BY id DESC
        LIMIT {int(limit)}
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_feedback_insights() -> pd.DataFrame:
    return fetch_dataframe(
        """
        SELECT top_disease, medicine_name, rating, irrelevant_medicine, comments
        FROM recommendation_feedback
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_table_counts() -> dict[str, int]:
    queries = {
        "Diseases": "SELECT COUNT(*) AS count FROM diseases",
        "Medicines": "SELECT COUNT(*) AS count FROM medicines",
        "Training Cases": "SELECT COUNT(*) AS count FROM training_cases",
        "Safety Rules": "SELECT COUNT(*) AS count FROM medicine_contraindications",
    }
    return {label: int(fetch_dataframe(query).iloc[0]["count"]) for label, query in queries.items()}


@st.cache_data(ttl=120, show_spinner=False)
def load_disease_medicine_catalog() -> pd.DataFrame:
    return fetch_dataframe(
        """
        SELECT d.name AS disease, m.name AS medicine, m.category, dm.relevance_score, dm.rationale
        FROM diseases d
        JOIN disease_medicines dm ON dm.disease_id = d.id
        JOIN medicines m ON m.id = dm.medicine_id
        ORDER BY d.name, dm.relevance_score DESC
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_medicine_conditions(medicine_name: str) -> pd.DataFrame:
    return fetch_dataframe(
        """
        SELECT d.name AS condition, dm.relevance_score, dm.rationale
        FROM medicines m
        JOIN disease_medicines dm ON dm.medicine_id = m.id
        JOIN diseases d ON d.id = dm.disease_id
        WHERE m.name = ?
        ORDER BY dm.relevance_score DESC, d.name
        LIMIT 20
        """,
        params=(medicine_name,),
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_medicine_contraindications(medicine_name: str) -> pd.DataFrame:
    return fetch_dataframe(
        """
        SELECT mc.condition_key, mc.severity, mc.reason
        FROM medicines m
        JOIN medicine_contraindications mc ON mc.medicine_id = m.id
        WHERE m.name = ?
        ORDER BY
          CASE mc.severity WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
          mc.condition_key
        """,
        params=(medicine_name,),
    )


def clear_data_caches() -> None:
    load_medicine_catalog.clear()
    load_history_summary.clear()
    load_recent_history.clear()
    load_recent_feedback.clear()
    load_feedback_insights.clear()
    load_table_counts.clear()
    load_disease_medicine_catalog.clear()
    load_medicine_conditions.clear()
    load_medicine_contraindications.clear()
