from html import escape

import streamlit as st

from ui.cached_data import load_medicine_catalog, load_medicine_conditions, load_medicine_contraindications
from ui.components import render_metric_card
from ui.formatting import display_text


@st.fragment
def render_lookup_tab() -> None:
    st.subheader("Medicine lookup")
    medicines = load_medicine_catalog()
    search_text = st.text_input("Search medicines", placeholder="Type a medicine name or category")
    if search_text:
        search_key = search_text.strip().lower()
        filtered = medicines[
            medicines["name"].str.lower().str.contains(search_key, regex=False)
            | medicines["category"].str.lower().str.contains(search_key, regex=False)
        ]
    else:
        filtered = medicines

    if filtered.empty:
        st.info("No medicines matched the search.")
        return

    selected_medicine = st.selectbox("Medicine", filtered["name"].tolist())
    selected_row = medicines[medicines["name"] == selected_medicine].iloc[0]
    lookup_columns = st.columns([1, 1, 1])
    with lookup_columns[0]:
        render_metric_card("Category", selected_row["category"], "blue")
    with lookup_columns[1]:
        prescription_text = "Required" if bool(selected_row["prescription_required"]) else "Not listed"
        render_metric_card(
            "Prescription",
            prescription_text,
            "amber" if bool(selected_row["prescription_required"]) else "green",
        )
    with lookup_columns[2]:
        render_metric_card("Effectiveness score", f"{float(selected_row['effectiveness_score']):.2f}", "green")
    st.markdown(
        f"""
        <div class="lookup-panel">
          <strong>{escape(selected_medicine)}</strong>
          <span>{escape(selected_row["safety_notes"])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    linked_conditions = load_medicine_conditions(selected_medicine)
    contraindications = load_medicine_contraindications(selected_medicine)

    condition_col, warning_col = st.columns(2)
    with condition_col:
        st.markdown("#### Matching conditions")
        if linked_conditions.empty:
            st.info("No condition mappings found.")
        else:
            linked_conditions = linked_conditions.rename(
                columns={"condition": "Condition", "relevance_score": "Relevance", "rationale": "Reason"}
            )
            linked_conditions["Relevance"] = linked_conditions["Relevance"].map(lambda value: f"{float(value):.2f}")
            st.dataframe(linked_conditions, hide_index=True, width="stretch")
    with warning_col:
        st.markdown("#### Safety warnings")
        if contraindications.empty:
            st.info("No contraindication rules found.")
        else:
            contraindications = contraindications.rename(
                columns={"condition_key": "Condition", "severity": "Severity", "reason": "Reason"}
            )
            contraindications["Condition"] = contraindications["Condition"].map(display_text)
            contraindications["Severity"] = contraindications["Severity"].map(display_text)
            st.dataframe(contraindications, hide_index=True, width="stretch")
