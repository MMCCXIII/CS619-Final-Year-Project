from html import escape

import pandas as pd
import streamlit as st

from src.config import LOW_CONFIDENCE_THRESHOLD
from ui.cached_data import load_feedback_insights
from ui.constants import INTERACTION_SCREEN_NOTE, SCORE_EXPLAINER
from ui.formatting import display_text


def render_metric_card(label: str, value, tone: str = "blue") -> None:
    st.markdown(
        f"""
        <div class="metric-card metric-card--{tone}">
          <span>{escape(label)}</span>
          <strong>{escape(str(value))}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_band(kind: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="status-band status-band--{kind}">
          <strong>{escape(title)}</strong>
          <span>{escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disease_cards(predictions: list[dict]) -> None:
    cols = st.columns(min(3, len(predictions)))
    for index, row in enumerate(predictions[:3]):
        confidence = float(row["confidence"])
        tone = "green" if index == 0 else "blue"
        matched = row.get("matched_symptoms") or []
        matched_text = ", ".join(display_text(item) for item in matched[:4]) if matched else "No strong symptom overlap"
        model_confidence = float(row.get("model_confidence", confidence))
        profile_match = float(row.get("profile_match", 0.0))
        with cols[index]:
            st.markdown(
                f"""
                <div class="result-card result-card--{tone}">
                  <span>{escape("Top match" if index == 0 else f"Alternative {index + 1}")}</span>
                  <strong>{escape(row["disease"])}</strong>
                  <div class="confidence-track">
                    <div style="width:{confidence * 100:.1f}%"></div>
                  </div>
                  <small>{confidence:.1%} confidence | {profile_match:.0%} profile fit</small>
                  <em>Matched: {escape(matched_text)}</em>
                  <small>Model only: {model_confidence:.1%}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_medicine_cards(medicines: list[dict]) -> None:
    cols = st.columns(min(3, max(1, len(medicines[:3]))))
    for index, row in enumerate(medicines[:3]):
        tone = "green" if index == 0 else "blue"
        prescription = "Prescription required" if row["prescription_required"] else "No prescription listed"
        warnings = row["warnings"] or "No profile-specific warnings"
        with cols[index]:
            st.markdown(
                f"""
                <div class="medicine-card medicine-card--{tone}">
                  <span>Suggestion {index + 1}</span>
                  <strong>{escape(row["medicine"])}</strong>
                  <div class="medicine-meta">
                    <b>{escape(row["disease"])}</b>
                    <i>{float(row["score"]):.3f} rank score</i>
                  </div>
                  <p>{escape(row["rationale"])}</p>
                  <div class="pill-row">
                    <small>{escape(prescription)}</small>
                    <small>{escape(row["status"])}</small>
                  </div>
                  <em>{escape(warnings)}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_guidance_items(items: list[str]) -> None:
    if not items:
        return
    html_items = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(f'<ul class="care-list">{html_items}</ul>', unsafe_allow_html=True)


def top_condition(result: dict) -> dict:
    predictions = result.get("predicted_diseases") or []
    return predictions[0] if predictions else {}


def metric_tone_for_urgency(urgency: dict) -> str:
    level = urgency.get("level", "")
    if level == "Self-care likely":
        return "green"
    if level == "Add more detail":
        return "amber"
    return "red"


def build_symptom_severity(
    selected_symptoms: list[str],
    overall_severity: str,
    symptom_severities: dict[str, str],
) -> dict[str, str]:
    severity = {"overall": overall_severity}
    for symptom in selected_symptoms:
        severity[symptom] = symptom_severities.get(symptom, overall_severity)
    return severity


def render_result_dashboard(result: dict) -> None:
    top = top_condition(result)
    urgency = result.get("urgency") or {}
    columns = st.columns(4)
    with columns[0]:
        render_metric_card("Urgency", urgency.get("level", "Review"), metric_tone_for_urgency(urgency))
    with columns[1]:
        render_metric_card("Top condition", top.get("disease", "None"), "blue")
    with columns[2]:
        confidence = float(top.get("confidence", 0.0))
        render_metric_card("Confidence", f"{confidence:.1%}", "green" if confidence >= 0.75 else "amber")
    with columns[3]:
        warning_count = len(result.get("red_flags") or [])
        render_metric_card("Safety warnings", warning_count, "red" if warning_count else "green")


def render_saved_profile(profile: dict) -> None:
    conditions = profile.get("conditions") or []
    body = f"Age group: {display_text(profile.get('age_group', 'adult'))}"
    if conditions:
        body += f" | Conditions: {', '.join(display_text(item) for item in conditions)}"
    else:
        body += " | Conditions: none saved"
    if profile.get("current_medicines"):
        body += " | Current medicines saved"
    if profile.get("allergies"):
        body += " | Allergies saved"
    st.markdown(f'<div class="profile-strip">{escape(body)}</div>', unsafe_allow_html=True)


def render_interaction_alerts(alerts: list[dict]) -> None:
    st.caption(INTERACTION_SCREEN_NOTE)
    if not alerts:
        render_status_band(
            "safe",
            "Medicine safety screen",
            "No allergy or current-medicine review alerts were detected from the entered profile.",
        )
        return
    for alert in alerts[:6]:
        kind = "warning" if alert["severity"] in {"high", "medium"} else "safe"
        title = f"{display_text(alert['severity'])} {alert['type']}"
        body = f"{alert['medicine']}: {alert['reason']}"
        render_status_band(kind, title, body)


def render_feedback_insights() -> None:
    feedback = load_feedback_insights()
    if feedback.empty:
        st.info("No feedback insights yet.")
        return

    feedback["medicine_name"] = feedback["medicine_name"].replace("", "Not specified")
    grouped = (
        feedback.groupby(["top_disease", "medicine_name"])
        .agg(
            Feedback=("rating", "count"),
            Irrelevant=("irrelevant_medicine", "sum"),
        )
        .reset_index()
        .sort_values(["Irrelevant", "Feedback"], ascending=False)
    )
    grouped = grouped.rename(
        columns={
            "top_disease": "Condition",
            "medicine_name": "Medicine",
            "Irrelevant": "Irrelevant flags",
        }
    )
    st.dataframe(grouped, hide_index=True, width="stretch")

    comments = feedback[feedback["comments"].astype(str).str.len() > 0].tail(10)
    if not comments.empty:
        comments = comments.rename(
            columns={
                "top_disease": "Condition",
                "medicine_name": "Medicine",
                "rating": "Rating",
                "comments": "Comment",
            }
        )
        st.markdown("#### Recent review notes")
        st.dataframe(comments[["Condition", "Medicine", "Rating", "Comment"]], hide_index=True, width="stretch")


def render_low_confidence_warning(result: dict) -> None:
    top = top_condition(result)
    confidence = float(top.get("confidence", 0.0))
    if confidence and confidence < LOW_CONFIDENCE_THRESHOLD:
        render_status_band(
            "warning",
            "Low confidence match",
            "Top condition confidence is below 50%. Treat results as exploratory only and add more specific symptoms.",
        )




def render_score_explainer() -> None:
    with st.expander("How scores are computed"):
        st.markdown(SCORE_EXPLAINER)


def render_feedback_insights_note() -> None:
    st.caption("Medicines with repeated irrelevant feedback flags receive a small ranking penalty.")
