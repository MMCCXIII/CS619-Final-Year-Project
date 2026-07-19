import streamlit as st

from ui.cached_data import load_disease_medicine_catalog, load_table_counts
from ui.components import render_metric_card


@st.fragment
def render_clinical_data_tab() -> None:
    st.subheader("Database overview")
    counts = load_table_counts()
    summary_cols = st.columns(4)
    metric_tones = ["green", "blue", "amber", "blue"]
    for column, (label, count), tone in zip(summary_cols, counts.items(), metric_tones):
        with column:
            render_metric_card(label, f"{count:,}", tone)

    st.markdown("#### Disease medicine catalog")
    catalog = load_disease_medicine_catalog()
    catalog = catalog.rename(
        columns={
            "disease": "Condition",
            "medicine": "Medicine",
            "category": "Category",
            "relevance_score": "Relevance score",
            "rationale": "Reason",
        }
    )
    catalog["Relevance score"] = catalog["Relevance score"].map(lambda value: f"{float(value):.2f}")
    st.dataframe(catalog, hide_index=True, width="stretch")
