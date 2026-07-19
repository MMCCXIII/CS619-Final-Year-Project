import streamlit as st

from ui.components import render_status_band
from ui.constants import DEFAULT_PROFILE, MEDICAL_DISCLAIMER
from ui.profile_store import load_persisted_profile
from ui.startup import ensure_artifacts
from ui.styles import inject_styles
from ui.tabs.clinical_data import render_clinical_data_tab
from ui.tabs.history import render_history_tab
from ui.tabs.lookup import render_lookup_tab
from ui.tabs.model_performance import render_model_performance_tab
from ui.tabs.recommendation import render_recommendation_tab
from ui.tabs.requirements import render_requirements_tab

st.set_page_config(
    page_title="Medicine Recommendation System",
    layout="wide",
)

inject_styles()
bundle = ensure_artifacts()

if "saved_profile" not in st.session_state:
    persisted = load_persisted_profile()
    st.session_state.saved_profile = persisted if persisted else DEFAULT_PROFILE.copy()
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_feedback_saved" not in st.session_state:
    st.session_state.latest_feedback_saved = False
if "recommendation_view" not in st.session_state:
    st.session_state.recommendation_view = "input"

st.title("Medicine Recommendation System")
st.markdown(
    '<div class="app-kicker">Clinical decision-support dashboard for symptom-based medicine recommendation.</div>',
    unsafe_allow_html=True,
)
render_status_band("warning", "Safety boundary", MEDICAL_DISCLAIMER)

tab_recommend, tab_lookup, tab_history, tab_data, tab_model, tab_requirements = st.tabs(
    ["Recommendation", "Medicine lookup", "History", "Clinical data", "Model performance", "Requirement coverage"]
)

with tab_recommend:
    render_recommendation_tab(bundle)

with tab_lookup:
    render_lookup_tab()

with tab_history:
    render_history_tab()

with tab_data:
    render_clinical_data_tab()

with tab_model:
    render_model_performance_tab(bundle)

with tab_requirements:
    render_requirements_tab()
