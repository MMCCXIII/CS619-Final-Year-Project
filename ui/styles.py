from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
CSS_PATH = ASSETS_DIR / "app.css"


def get_active_color_scheme() -> str:
    theme = getattr(st.context, "theme", {}) or {}
    theme_type = theme.get("type") if isinstance(theme, dict) else getattr(theme, "type", None)
    return "dark" if theme_type == "dark" else "light"


def inject_styles() -> None:
    css = CSS_PATH.read_text(encoding="utf-8").replace("__APP_COLOR_SCHEME__", get_active_color_scheme())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
