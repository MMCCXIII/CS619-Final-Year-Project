from html import escape

import pandas as pd
import streamlit as st

from src.config import CONDITION_OPTIONS
from src.data_access import log_feedback
from src.recommender import recommend
from src.safety import (
    BODY_AREA_SYMPTOMS,
    RED_FLAG_QUESTIONS,
    filter_symptoms_for_body_area,
    red_flags_from_answers,
)
from ui.cached_data import clear_data_caches
from ui.components import (
    build_symptom_severity,
    render_disease_cards,
    render_guidance_items,
    render_interaction_alerts,
    render_low_confidence_warning,
    render_medicine_cards,
    render_result_dashboard,
    render_saved_profile,
    render_score_explainer,
    render_status_band,
    top_condition,
)
from ui.constants import (
    DEFAULT_PROFILE,
    DURATION_OPTIONS,
    FEEDBACK_OPTIONS,
    SEVERITY_OPTIONS,
)
from ui.export import session_report_json
from ui.formatting import display_text
from ui.profile_store import clear_persisted_profile, save_persisted_profile
from ui.validation import can_generate_recommendation


def reset_recommendation_flow(clear_symptoms: bool = False) -> None:
    st.session_state.recommendation_view = "input"
    st.session_state.latest_result = None
    st.session_state.latest_feedback_saved = False


def render_recommendation_tab(bundle: dict) -> None:
    if st.session_state.get("recommendation_view") == "results":
        _render_results_view()
        return
    _render_input_view(bundle)


def _render_input_view(bundle: dict) -> None:
    st.subheader("Patient input")
    st.caption("Enter symptoms and profile details, then generate your recommendation.")

    saved_profile = st.session_state.saved_profile
    symptom_severities = {}
    red_flag_answers = {}
    generate_requested = False

    input_col, safety_col = st.columns([1.25, 1], gap="large")

    with input_col:
        body_area = st.selectbox(
            "Main body area (Optional)",
            ["Not sure"] + list(BODY_AREA_SYMPTOMS),
            key="body_area_select",
            help="Leave as 'Not sure' to browse all symptoms, including general or non-area-specific ones.",
        )

        symptom_options = filter_symptoms_for_body_area(body_area, bundle["vocabulary"])
        if body_area != "Not sure":
            st.caption(
                f"Showing {len(symptom_options)} symptoms related to {body_area}. Choose 'Not sure' for the full list."
            )

        selected_symptoms = st.multiselect(
            "Symptoms",
            options=symptom_options,
            placeholder="Select known symptoms",
            format_func=display_text,
        )
        severity_col, duration_col = st.columns(2)
        with severity_col:
            overall_severity = st.radio(
                "Overall severity",
                SEVERITY_OPTIONS,
                index=1,
                horizontal=True,
            )
        with duration_col:
            symptom_duration = st.selectbox("Duration", DURATION_OPTIONS)

        with st.expander("More symptom details", expanded=False):
            free_text = st.text_area(
                "Additional symptom notes",
                placeholder="Example: high fever with rash and pain behind eyes",
                height=70,
            )
            if selected_symptoms:
                st.markdown("##### Symptom-level severity")
                severity_columns = st.columns(2)
                for index, symptom in enumerate(selected_symptoms[:8]):
                    with severity_columns[index % 2]:
                        symptom_severities[symptom] = st.selectbox(
                            display_text(symptom),
                            SEVERITY_OPTIONS,
                            index=SEVERITY_OPTIONS.index(overall_severity),
                            key=f"severity_{index}_{symptom}",
                        )
                if len(selected_symptoms) > 8:
                    st.caption("Additional selected symptoms use the overall severity.")

    with safety_col:
        age_options = bundle["age_groups"]
        saved_age = saved_profile.get("age_group", "adult")
        age_index = age_options.index(saved_age) if saved_age in age_options else 1
        age_group = st.selectbox(
            "Age group",
            options=age_options,
            index=age_index,
            format_func=display_text,
        )
        saved_conditions = [item for item in saved_profile.get("conditions", []) if item in CONDITION_OPTIONS]

        with st.expander("Safety profile", expanded=False):
            conditions = st.multiselect(
                "Medical history and existing conditions",
                CONDITION_OPTIONS,
                default=saved_conditions,
                format_func=display_text,
            )
            current_medicines = st.text_area(
                "Current medicines",
                value=saved_profile.get("current_medicines", ""),
                placeholder="Example: warfarin, insulin, sertraline",
                height=60,
            )
            allergies = st.text_area(
                "Known medicine allergies",
                value=saved_profile.get("allergies", ""),
                placeholder="Example: penicillin, sulfa, ibuprofen",
                height=60,
            )
            render_saved_profile(saved_profile)
            profile_columns = st.columns(2)
            with profile_columns[0]:
                if st.button("Save safety profile", width="stretch"):
                    profile = {
                        "age_group": age_group,
                        "conditions": conditions,
                        "current_medicines": current_medicines,
                        "allergies": allergies,
                    }
                    st.session_state.saved_profile = profile
                    save_persisted_profile(profile)
                    st.success("Safety profile saved for this session and locally on disk.")
            with profile_columns[1]:
                if st.button("Clear profile", width="stretch"):
                    st.session_state.saved_profile = DEFAULT_PROFILE.copy()
                    clear_persisted_profile()
                    st.success("Saved profile cleared.")

        with st.expander("Red-flag safety questions", expanded=False):
            red_flag_columns = st.columns(2)
            for index, (flag_key, question) in enumerate(RED_FLAG_QUESTIONS):
                with red_flag_columns[index % 2]:
                    red_flag_answers[flag_key] = st.checkbox(question, key=f"red_flag_{flag_key}")

        ready = can_generate_recommendation(selected_symptoms, free_text, bundle["vocabulary"])
        if not ready:
            st.caption("Select at least one symptom or enter recognized symptom terms in the notes.")
        generate_requested = st.button(
            "Generate recommendation",
            type="primary",
            width="stretch",
            disabled=not ready,
        )

    if generate_requested:
        symptom_severity = build_symptom_severity(selected_symptoms, overall_severity, symptom_severities)
        answered_red_flags = red_flags_from_answers(red_flag_answers)
        result = recommend(
            selected_symptoms=selected_symptoms,
            free_text=free_text,
            age_group=age_group,
            profile_conditions=conditions,
            symptom_severity=symptom_severity,
            symptom_duration=symptom_duration,
            red_flag_answers=answered_red_flags,
            current_medicines=current_medicines,
            allergies=allergies,
            bundle=bundle,
        )
        result["body_area"] = body_area

        if result.get("validation_error"):
            st.error(result["message"])
            return

        st.session_state.latest_result = result
        st.session_state.latest_feedback_saved = False
        st.session_state.recommendation_view = "results"
        st.rerun()


def _render_results_view() -> None:
    result = st.session_state.latest_result
    if not result:
        reset_recommendation_flow()
        st.rerun()
        return

    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.subheader("Your recommendations")
        st.caption("Review the suggestions below. You can start again to enter new symptoms.")
    with action_col:
        if st.button("Start again", type="secondary", width="stretch"):
            reset_recommendation_flow(clear_symptoms=True)
            st.rerun()

    if result.get("validation_error"):
        st.error(result["message"])
        if st.button("Back to input", width="stretch"):
            reset_recommendation_flow()
            st.rerun()
        return

    _render_successful_result(result)


def _render_successful_result(result: dict) -> None:
    if result.get("input_alert"):
        st.warning(result["input_alert"])
    render_result_dashboard(result)
    render_low_confidence_warning(result)
    render_score_explainer()
    urgency = result.get("urgency") or {}
    render_status_band(
        urgency.get("tone", "warning"),
        urgency.get("level", "Review"),
        f"{urgency.get('summary', result['message'])} {urgency.get('next_step', '')}".strip(),
    )
    if result["red_flags"]:
        render_status_band(
            "warning",
            "Urgent review flags",
            ", ".join(display_text(item) for item in result["red_flags"]),
        )
    else:
        render_status_band(
            "safe",
            "Safety screen",
            "No urgent review flags were detected from the entered profile.",
        )
    render_interaction_alerts(result.get("interaction_alerts", []))
    st.markdown(
        f'<div class="detected-list"><strong>Detected symptoms:</strong> {escape(", ".join(display_text(item) for item in result["symptoms"]) or "None")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="detected-list"><strong>Input context:</strong> Area: {escape(result.get("body_area", "Not sure"))}'
        f' | Duration: {escape(result.get("symptom_duration", "Not set"))}</div>',
        unsafe_allow_html=True,
    )

    disease_df = pd.DataFrame(result["predicted_diseases"])
    medicine_df = pd.DataFrame(result["medicines"])

    if not disease_df.empty:
        st.markdown("#### Likely conditions")
        render_disease_cards(result["predicted_diseases"])
        disease_table = pd.DataFrame(
            [
                {
                    "Condition": row["disease"],
                    "Confidence": f"{float(row['confidence']):.1%}",
                    "Profile fit": f"{float(row.get('profile_match', 0.0)):.0%}",
                    "Matched symptoms": ", ".join(display_text(item) for item in row.get("matched_symptoms", [])[:5])
                    or "None",
                }
                for row in result["predicted_diseases"]
            ]
        )
        st.dataframe(disease_table, hide_index=True, width="stretch")

    if not medicine_df.empty:
        st.markdown("#### Ranked medicine suggestions")
        render_medicine_cards(result["medicines"])
        display_columns = [
            "medicine",
            "disease",
            "score",
            "therapeutic_fit",
            "status",
            "prescription_required",
            "rationale",
            "warnings",
        ]
        medicine_table = medicine_df[display_columns].rename(
            columns={
                "medicine": "Medicine",
                "disease": "Condition",
                "score": "Rank score",
                "therapeutic_fit": "Symptom fit",
                "status": "Safety status",
                "prescription_required": "Prescription",
                "rationale": "Reason",
                "warnings": "Warnings",
            }
        )
        medicine_table["Rank score"] = medicine_table["Rank score"].map(lambda value: f"{float(value):.3f}")
        medicine_table["Symptom fit"] = medicine_table["Symptom fit"].map(lambda value: f"{float(value):.0%}")
        medicine_table["Prescription"] = medicine_table["Prescription"].map(
            lambda value: "Required" if bool(value) else "Not required"
        )
        medicine_table["Warnings"] = medicine_table["Warnings"].replace("", "None")
        st.dataframe(medicine_table, hide_index=True, width="stretch")
        with st.expander("Safety notes"):
            for row in result["medicines"][:8]:
                st.markdown(f"**{row['medicine']}**: {row['safety_notes']}")
    else:
        st.info(result["message"])
    st.markdown("#### Supportive care")
    render_guidance_items(result.get("care_guidance", []))

    st.download_button(
        "Download session report (JSON)",
        data=session_report_json(result),
        file_name="recommendation_session_report.json",
        mime="application/json",
        width="stretch",
    )

    st.markdown("#### Feedback")
    if st.session_state.latest_feedback_saved:
        st.success("Feedback saved. Thank you.")
    else:
        with st.form("recommendation_feedback"):
            feedback_rating = st.radio(
                "Was this result useful?",
                FEEDBACK_OPTIONS,
                horizontal=True,
            )
            irrelevant_medicine = st.checkbox("A medicine suggestion seemed irrelevant")
            medicine_options = ["Not specified"] + [row["medicine"] for row in result.get("medicines", [])[:8]]
            feedback_medicine = st.selectbox("Medicine to review", medicine_options)
            feedback_comments = st.text_area(
                "Comments",
                placeholder="Optional note about what felt right or wrong",
                height=80,
            )
            submitted_feedback = st.form_submit_button("Submit feedback")
        if submitted_feedback:
            top = top_condition(result)
            log_feedback(
                symptoms=result["symptoms"],
                top_disease=top.get("disease", "Unknown"),
                rating=feedback_rating,
                irrelevant_medicine=irrelevant_medicine or feedback_rating == "Medicine seemed irrelevant",
                comments=feedback_comments,
                medicine_name="" if feedback_medicine == "Not specified" else feedback_medicine,
            )
            clear_data_caches()
            st.session_state.latest_feedback_saved = True
            st.success("Feedback saved. Thank you.")
