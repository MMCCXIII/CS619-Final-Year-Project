ACRONYMS = {
    "covid-19": "COVID-19",
    "gerd": "GERD",
    "mlp": "MLP",
    "f1": "F1",
    "sqlite": "SQLite",
    "csv": "CSV",
    "ui": "UI",
    "uti": "UTI",
}

DURATION_OPTIONS = [
    "Started today",
    "2-3 days",
    "4-7 days",
    "More than a week",
    "Recurring or ongoing",
]

SEVERITY_OPTIONS = ["Mild", "Moderate", "Severe"]
FEEDBACK_OPTIONS = ["Useful", "Partly useful", "Medicine seemed irrelevant", "Not useful"]

DEFAULT_PROFILE = {
    "age_group": "adult",
    "conditions": [],
    "current_medicines": "",
    "allergies": "",
}

MEDICAL_DISCLAIMER = (
    "Academic demo only — not medical advice, diagnosis, or prescription. "
    "Seek qualified care for urgent symptoms."
)

INTERACTION_SCREEN_NOTE = (
    "Medicine interaction screening uses keyword matching on entered names and allergies. "
    "It is not a complete drug-interaction database."
)

SCORE_EXPLAINER = (
    "Condition confidence blends model probability with symptom-profile overlap. "
    "Medicine rank score combines disease confidence, catalog relevance, effectiveness, "
    "symptom fit, severity, duration, age group, safety profile, current medicines, allergies, "
    "red-flag context, contraindication penalties, and feedback-based down-ranking."
)
