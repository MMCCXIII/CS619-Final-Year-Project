import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_VERSION = "1.0.0"
MODEL_ARTIFACT_VERSION = "1.0.0"
LOW_CONFIDENCE_THRESHOLD = 0.50
FEEDBACK_PENALTY_PER_FLAG = 0.08
FEEDBACK_PENALTY_MAX = 0.25
PROFILE_STORE_PATH = BASE_DIR / "database" / "saved_profile.json"
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "medicine_dataset.csv"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
DATABASE_PATH = DATABASE_DIR / "medicine_recommender.sqlite"
MODEL_BUNDLE_PATH = MODEL_DIR / "disease_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

AGE_GROUPS = ["child", "adult", "older adult"]

SAFETY_RED_FLAGS = {
    "chest pain",
    "shortness of breath",
    "confusion",
    "low oxygen",
    "bleeding gums",
    "black stool",
    "blood in urine",
    "severe abdominal pain",
    "rapid breathing",
    "severe dehydration",
}

CONDITION_OPTIONS = [
    "pregnancy",
    "breastfeeding",
    "liver disease",
    "kidney disease",
    "gastritis",
    "stomach ulcer",
    "hypertension",
    "heart disease",
    "heart rhythm disorder",
    "diabetes",
    "asthma",
    "anticoagulant use",
    "bleeding risk",
    "penicillin allergy",
    "sulfa allergy",
    "antibiotic allergy",
    "glaucoma",
    "prostate enlargement",
    "seizure disorder",
    "tendon disorder",
    "bradycardia",
    "electrolyte imbalance",
    "active infection",
    "hypoglycemia",
    "older adult",
    "severe allergic reaction",
]


def file_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
