import streamlit as st

from src.config import DATABASE_PATH, MODEL_BUNDLE_PATH
from src.data_access import ensure_runtime_tables, seed_database
from src.modeling import load_model_bundle, train_and_save_models


@st.cache_resource(show_spinner=False)
def ensure_artifacts():
    if not DATABASE_PATH.exists():
        seed_database(DATABASE_PATH)
    ensure_runtime_tables(DATABASE_PATH)
    if not MODEL_BUNDLE_PATH.exists():
        with st.spinner("Training models (first run only). This may take a minute..."):
            train_and_save_models()
    return load_model_bundle()
