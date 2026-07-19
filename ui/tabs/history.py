import streamlit as st

from src.data_access import clear_all_history, clear_feedback_history, clear_recommendation_history
from src.safety import build_privacy_points
from ui.cached_data import clear_data_caches, load_history_summary, load_recent_feedback, load_recent_history
from ui.components import render_feedback_insights, render_feedback_insights_note, render_guidance_items, render_metric_card
from ui.formatting import display_text, format_symptom_list
from ui.tabs.recommendation import reset_recommendation_flow


@st.fragment
def render_history_tab() -> None:
    st.subheader("Recommendation history")
    with st.expander("Privacy and local data"):
        render_guidance_items(build_privacy_points())
        st.caption("History is stored in a local SQLite file without encryption. Export data before clearing if needed.")
    with st.expander("Manage history"):
        st.warning(
            "Clearing history permanently removes saved recommendation logs and/or feedback entries from this local database."
        )
        clear_col_1, clear_col_2, clear_col_3 = st.columns(3)
        confirm_recommendations = st.checkbox("Confirm recommendation clear")
        confirm_feedback = st.checkbox("Confirm feedback clear")
        confirm_all = st.checkbox("Confirm clearing both")
        with clear_col_1:
            if st.button(
                "Clear recommendations",
                disabled=not confirm_recommendations,
                width="stretch",
            ):
                removed = clear_recommendation_history()
                reset_recommendation_flow()
                clear_data_caches()
                st.success(f"Cleared {removed:,} recommendation history entries.")
        with clear_col_2:
            if st.button(
                "Clear feedback",
                disabled=not confirm_feedback,
                width="stretch",
            ):
                removed = clear_feedback_history()
                st.session_state.latest_feedback_saved = False
                clear_data_caches()
                st.success(f"Cleared {removed:,} feedback entries.")
        with clear_col_3:
            if st.button(
                "Clear both",
                disabled=not confirm_all,
                width="stretch",
            ):
                removed = clear_all_history()
                reset_recommendation_flow()
                clear_data_caches()
                st.success(
                    f"Cleared {removed['recommendations']:,} recommendation entries and {removed['feedback']:,} feedback entries."
                )

    summary = load_history_summary()
    history_cols = st.columns(3)
    with history_cols[0]:
        render_metric_card("Saved checks", f"{summary['total_checks']:,}", "blue")
    with history_cols[1]:
        render_metric_card("Feedback entries", f"{summary['total_feedback']:,}", "green")
    with history_cols[2]:
        render_metric_card("Irrelevant flags", f"{summary['irrelevant_count']:,}", "amber")

    history = load_recent_history()
    if history.empty:
        st.info("No recommendation history yet.")
    else:
        history_table = history.rename(
            columns={
                "created_at": "Time",
                "symptoms": "Symptoms",
                "age_group": "Age group",
                "profile_conditions": "Profile conditions",
                "top_disease": "Top condition",
                "model_confidence": "Model confidence",
            }
        )
        history_table["Symptoms"] = history_table["Symptoms"].map(format_symptom_list)
        history_table["Age group"] = history_table["Age group"].map(display_text)
        history_table["Profile conditions"] = history_table["Profile conditions"].map(format_symptom_list)
        history_table["Model confidence"] = history_table["Model confidence"].map(lambda value: f"{float(value):.1%}")
        st.dataframe(history_table, hide_index=True, width="stretch")

    feedback = load_recent_feedback()
    st.markdown("#### Recent feedback")
    if feedback.empty:
        st.info("No feedback submitted yet.")
    else:
        feedback_table = feedback.rename(
            columns={
                "created_at": "Time",
                "top_disease": "Top condition",
                "medicine_name": "Medicine",
                "rating": "Rating",
                "irrelevant_medicine": "Irrelevant medicine flagged",
                "comments": "Comments",
            }
        )
        feedback_table["Medicine"] = feedback_table["Medicine"].replace("", "Not specified")
        feedback_table["Irrelevant medicine flagged"] = feedback_table["Irrelevant medicine flagged"].map(
            lambda value: "Yes" if bool(value) else "No"
        )
        feedback_table["Comments"] = feedback_table["Comments"].replace("", "None")
        st.dataframe(feedback_table, hide_index=True, width="stretch")

    st.markdown("#### Feedback review dashboard")
    render_feedback_insights_note()
    render_feedback_insights()
