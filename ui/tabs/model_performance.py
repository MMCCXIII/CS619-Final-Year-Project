import json

import pandas as pd
import streamlit as st

from src.config import DATASET_PATH, DATASET_VERSION, METRICS_PATH, MODEL_ARTIFACT_VERSION, file_fingerprint
from ui.components import render_metric_card


@st.fragment
def render_model_performance_tab(bundle: dict) -> None:
    st.subheader("Training results")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        metrics = bundle["metrics"]

    st.caption(
        f"Dataset version {DATASET_VERSION} (fingerprint {file_fingerprint(DATASET_PATH)}) | "
        f"Model artifact version {MODEL_ARTIFACT_VERSION} (fingerprint {file_fingerprint(METRICS_PATH)})"
    )

    model_cols = st.columns(3)
    with model_cols[0]:
        render_metric_card("Best model", metrics["best_model"], "green")
    with model_cols[1]:
        render_metric_card("Training cases", f"{metrics['training_cases']:,}", "blue")
    with model_cols[2]:
        render_metric_card("Feature count", f"{metrics['feature_count']:,}", "amber")

    model_rows = []
    for model_name, values in metrics["models"].items():
        model_rows.append(
            {
                "Model": model_name,
                "Accuracy": values["accuracy"],
                "Weighted F1": values["f1_weighted"],
                "Best Params": values["best_params"],
            }
        )
    model_table = pd.DataFrame(model_rows).rename(
        columns={"Weighted F1": "Weighted F1", "Best Params": "Best parameters"}
    )
    model_table["Accuracy"] = model_table["Accuracy"].map(lambda value: f"{float(value):.2%}")
    model_table["Weighted F1"] = model_table["Weighted F1"].map(lambda value: f"{float(value):.2%}")
    model_table["Best parameters"] = model_table["Best parameters"].map(
        lambda value: json.dumps(value) if isinstance(value, dict) else (value or "Default")
    )
    st.dataframe(model_table, hide_index=True, width="stretch")
