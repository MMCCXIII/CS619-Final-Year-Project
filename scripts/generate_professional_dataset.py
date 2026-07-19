import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_PATH
from src.knowledge_base import RELATED_NOISE as EXPANDED_RELATED_NOISE
from src.knowledge_base import build_disease_profiles


TARGET_CASES = 4800
RANDOM_SEED = 13357


@dataclass(frozen=True)
class DiseaseProfile:
    name: str
    core_symptoms: tuple[str, ...]
    supporting_symptoms: tuple[str, ...]
    red_flag_symptoms: tuple[str, ...]
    age_weights: dict[str, float]
    severity_weights: dict[str, float]


PROFILES = [
    DiseaseProfile(
        name="Common Cold",
        core_symptoms=("runny nose", "nasal congestion", "sneezing", "sore throat", "mild cough"),
        supporting_symptoms=(
            "post nasal drip",
            "watery eyes",
            "low grade fever",
            "mild headache",
            "fatigue",
            "clear nasal discharge",
            "itchy throat",
            "hoarse voice",
            "ear pressure",
            "reduced smell",
        ),
        red_flag_symptoms=("shortness of breath", "chest pain", "high fever"),
        age_weights={"child": 0.30, "adult": 0.50, "older adult": 0.20},
        severity_weights={"mild": 0.82, "moderate": 0.17, "severe": 0.01},
    ),
    DiseaseProfile(
        name="Influenza",
        core_symptoms=("fever", "high fever", "chills", "body aches", "dry cough", "fatigue"),
        supporting_symptoms=(
            "headache",
            "sore throat",
            "runny nose",
            "muscle pain",
            "loss of appetite",
            "weakness",
            "sweating",
            "chest discomfort",
            "nausea",
            "poor sleep",
        ),
        red_flag_symptoms=("shortness of breath", "chest pain", "confusion", "rapid breathing"),
        age_weights={"child": 0.25, "adult": 0.48, "older adult": 0.27},
        severity_weights={"mild": 0.22, "moderate": 0.64, "severe": 0.14},
    ),
    DiseaseProfile(
        name="Migraine",
        core_symptoms=("severe headache", "one sided headache", "throbbing pain", "nausea", "light sensitivity"),
        supporting_symptoms=(
            "sound sensitivity",
            "visual aura",
            "vomiting",
            "dizziness",
            "neck stiffness",
            "fatigue",
            "blurred vision",
            "scalp tenderness",
            "poor sleep",
            "stress",
        ),
        red_flag_symptoms=("confusion", "new neurological weakness", "worst headache"),
        age_weights={"child": 0.08, "adult": 0.76, "older adult": 0.16},
        severity_weights={"mild": 0.12, "moderate": 0.70, "severe": 0.18},
    ),
    DiseaseProfile(
        name="Hypertension",
        core_symptoms=("high blood pressure", "headache", "dizziness", "blurred vision", "fatigue"),
        supporting_symptoms=(
            "morning headache",
            "neck pain",
            "palpitations",
            "anxiety",
            "nosebleed",
            "chest discomfort",
            "poor sleep",
            "shortness of breath on exertion",
            "ringing ears",
            "flushing",
        ),
        red_flag_symptoms=("chest pain", "shortness of breath", "confusion", "severe headache"),
        age_weights={"child": 0.02, "adult": 0.56, "older adult": 0.42},
        severity_weights={"mild": 0.20, "moderate": 0.63, "severe": 0.17},
    ),
    DiseaseProfile(
        name="Type 2 Diabetes",
        core_symptoms=("frequent urination", "excessive thirst", "fatigue", "blurred vision", "dry mouth"),
        supporting_symptoms=(
            "excessive hunger",
            "unexplained weight loss",
            "slow wound healing",
            "tingling feet",
            "recurrent infections",
            "night urination",
            "weakness",
            "itchy skin",
            "headache",
            "poor concentration",
        ),
        red_flag_symptoms=("confusion", "severe dehydration", "vomiting", "rapid breathing"),
        age_weights={"child": 0.03, "adult": 0.55, "older adult": 0.42},
        severity_weights={"mild": 0.20, "moderate": 0.66, "severe": 0.14},
    ),
    DiseaseProfile(
        name="Gastritis",
        core_symptoms=("upper abdominal pain", "burning stomach", "nausea", "bloating", "loss of appetite"),
        supporting_symptoms=(
            "vomiting",
            "belching",
            "early fullness",
            "acid reflux",
            "weakness",
            "dizziness",
            "stomach cramps",
            "sour taste",
            "heartburn",
            "indigestion",
        ),
        red_flag_symptoms=("black stool", "severe abdominal pain", "vomiting blood", "fainting"),
        age_weights={"child": 0.12, "adult": 0.62, "older adult": 0.26},
        severity_weights={"mild": 0.34, "moderate": 0.55, "severe": 0.11},
    ),
    DiseaseProfile(
        name="Allergic Rhinitis",
        core_symptoms=("sneezing", "itchy eyes", "runny nose", "nasal congestion", "watery eyes"),
        supporting_symptoms=(
            "seasonal trigger",
            "post nasal drip",
            "itchy throat",
            "clear nasal discharge",
            "fatigue",
            "ear pressure",
            "mild cough",
            "reduced smell",
            "itchy nose",
            "allergy trigger",
        ),
        red_flag_symptoms=("wheezing", "shortness of breath", "facial swelling"),
        age_weights={"child": 0.26, "adult": 0.56, "older adult": 0.18},
        severity_weights={"mild": 0.76, "moderate": 0.22, "severe": 0.02},
    ),
    DiseaseProfile(
        name="Urinary Tract Infection",
        core_symptoms=("burning urination", "frequent urination", "urinary urgency", "lower abdominal pain", "cloudy urine"),
        supporting_symptoms=(
            "pelvic pain",
            "blood in urine",
            "foul smelling urine",
            "fever",
            "chills",
            "back pain",
            "fatigue",
            "night urination",
            "pressure in lower belly",
            "nausea",
        ),
        red_flag_symptoms=("high fever", "back pain", "vomiting", "confusion"),
        age_weights={"child": 0.16, "adult": 0.58, "older adult": 0.26},
        severity_weights={"mild": 0.25, "moderate": 0.58, "severe": 0.17},
    ),
    DiseaseProfile(
        name="Bronchitis",
        core_symptoms=("productive cough", "mucus", "wheezing", "chest tightness", "fatigue"),
        supporting_symptoms=(
            "sore throat",
            "chest discomfort",
            "mild fever",
            "body aches",
            "runny nose",
            "persistent cough",
            "shortness of breath on exertion",
            "hoarse voice",
            "night cough",
            "headache",
        ),
        red_flag_symptoms=("shortness of breath", "chest pain", "high fever", "rapid breathing"),
        age_weights={"child": 0.22, "adult": 0.52, "older adult": 0.26},
        severity_weights={"mild": 0.24, "moderate": 0.62, "severe": 0.14},
    ),
    DiseaseProfile(
        name="COVID-19",
        core_symptoms=("fever", "dry cough", "fatigue", "loss of smell", "loss of taste"),
        supporting_symptoms=(
            "sore throat",
            "body aches",
            "headache",
            "nasal congestion",
            "chills",
            "diarrhea",
            "chest tightness",
            "runny nose",
            "nausea",
            "weakness",
        ),
        red_flag_symptoms=("shortness of breath", "chest pain", "low oxygen", "confusion"),
        age_weights={"child": 0.16, "adult": 0.58, "older adult": 0.26},
        severity_weights={"mild": 0.34, "moderate": 0.50, "severe": 0.16},
    ),
    DiseaseProfile(
        name="Asthma",
        core_symptoms=("wheezing", "shortness of breath", "chest tightness", "night cough", "triggered by exercise"),
        supporting_symptoms=(
            "allergy trigger",
            "triggered by cold air",
            "rapid breathing",
            "cough",
            "fatigue",
            "anxiety",
            "difficulty speaking",
            "recurrent wheeze",
            "poor sleep",
            "chest discomfort",
        ),
        red_flag_symptoms=("low oxygen", "severe shortness of breath", "rapid breathing", "confusion"),
        age_weights={"child": 0.32, "adult": 0.46, "older adult": 0.22},
        severity_weights={"mild": 0.22, "moderate": 0.58, "severe": 0.20},
    ),
    DiseaseProfile(
        name="Malaria",
        core_symptoms=("fever", "chills", "sweating", "headache", "body aches"),
        supporting_symptoms=(
            "cyclic fever",
            "nausea",
            "vomiting",
            "fatigue",
            "loss of appetite",
            "weakness",
            "muscle pain",
            "abdominal pain",
            "dizziness",
            "travel to endemic area",
        ),
        red_flag_symptoms=("confusion", "severe weakness", "persistent vomiting", "jaundice"),
        age_weights={"child": 0.26, "adult": 0.55, "older adult": 0.19},
        severity_weights={"mild": 0.09, "moderate": 0.44, "severe": 0.47},
    ),
    DiseaseProfile(
        name="Dengue Fever",
        core_symptoms=("high fever", "severe joint pain", "rash", "pain behind eyes", "muscle pain"),
        supporting_symptoms=(
            "headache",
            "nausea",
            "vomiting",
            "low platelets",
            "fatigue",
            "weakness",
            "loss of appetite",
            "body aches",
            "abdominal pain",
            "mosquito exposure",
        ),
        red_flag_symptoms=("bleeding gums", "severe abdominal pain", "persistent vomiting", "confusion"),
        age_weights={"child": 0.24, "adult": 0.58, "older adult": 0.18},
        severity_weights={"mild": 0.10, "moderate": 0.48, "severe": 0.42},
    ),
    DiseaseProfile(
        name="Pneumonia",
        core_symptoms=("fever", "productive cough", "shortness of breath", "chest pain", "rapid breathing"),
        supporting_symptoms=(
            "chills",
            "fatigue",
            "weakness",
            "low oxygen",
            "mucus",
            "confusion",
            "sweating",
            "loss of appetite",
            "body aches",
            "chest tightness",
        ),
        red_flag_symptoms=("low oxygen", "confusion", "severe shortness of breath", "blue lips"),
        age_weights={"child": 0.22, "adult": 0.42, "older adult": 0.36},
        severity_weights={"mild": 0.08, "moderate": 0.40, "severe": 0.52},
    ),
    DiseaseProfile(
        name="GERD",
        core_symptoms=("heartburn", "acid reflux", "regurgitation", "sour taste", "chronic cough"),
        supporting_symptoms=(
            "chest burning",
            "belching",
            "hoarse voice",
            "sore throat",
            "night cough",
            "bloating",
            "upper abdominal pain",
            "nausea",
            "early fullness",
            "worse after meals",
        ),
        red_flag_symptoms=("chest pain", "trouble swallowing", "black stool", "unexplained weight loss"),
        age_weights={"child": 0.08, "adult": 0.62, "older adult": 0.30},
        severity_weights={"mild": 0.54, "moderate": 0.38, "severe": 0.08},
    ),
    DiseaseProfile(
        name="Tension Headache",
        core_symptoms=("mild headache", "pressure around head", "dull headache", "neck stiffness", "stress"),
        supporting_symptoms=(
            "forehead pressure",
            "shoulder tension",
            "poor sleep",
            "eye strain",
            "fatigue",
            "scalp tenderness",
            "anxiety",
            "jaw tightness",
            "mild dizziness",
            "work stress",
        ),
        red_flag_symptoms=("confusion", "new neurological weakness", "worst headache", "fever"),
        age_weights={"child": 0.12, "adult": 0.64, "older adult": 0.24},
        severity_weights={"mild": 0.70, "moderate": 0.28, "severe": 0.02},
    ),
]


RELATED_NOISE = {
    "respiratory": ("fatigue", "headache", "mild fever", "sore throat", "runny nose", "chest discomfort"),
    "gastro": ("nausea", "loss of appetite", "weakness", "dizziness", "fatigue"),
    "systemic": ("fatigue", "weakness", "headache", "nausea", "loss of appetite"),
    "neuro": ("fatigue", "dizziness", "poor sleep", "stress"),
    "urinary": ("fatigue", "nausea", "lower abdominal pain"),
}


PROFILE_GROUP = {
    "Common Cold": "respiratory",
    "Influenza": "respiratory",
    "Bronchitis": "respiratory",
    "COVID-19": "respiratory",
    "Asthma": "respiratory",
    "Pneumonia": "respiratory",
    "Gastritis": "gastro",
    "GERD": "gastro",
    "Migraine": "neuro",
    "Tension Headache": "neuro",
    "Urinary Tract Infection": "urinary",
    "Hypertension": "systemic",
    "Type 2 Diabetes": "systemic",
    "Malaria": "systemic",
    "Dengue Fever": "systemic",
    "Allergic Rhinitis": "respiratory",
}

PROFILES = build_disease_profiles()
RELATED_NOISE = EXPANDED_RELATED_NOISE
PROFILE_GROUP = {profile.name: profile.group for profile in PROFILES}


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    labels = list(weights)
    values = [weights[label] for label in labels]
    return rng.choices(labels, weights=values, k=1)[0]


def sample_case(rng: random.Random, profile: DiseaseProfile) -> tuple[list[str], str, str]:
    severity = weighted_choice(rng, profile.severity_weights)
    age_group = weighted_choice(rng, profile.age_weights)

    core_count = rng.randint(3, min(5, len(profile.core_symptoms)))
    supporting_count = rng.randint(2, min(5, len(profile.supporting_symptoms)))
    symptoms = set(rng.sample(profile.core_symptoms, core_count))
    symptoms.update(rng.sample(profile.supporting_symptoms, supporting_count))

    red_flag_probability = {"mild": 0.02, "moderate": 0.08, "severe": 0.38}[severity]
    if rng.random() < red_flag_probability:
        symptoms.add(rng.choice(profile.red_flag_symptoms))

    group = PROFILE_GROUP[profile.name]
    if rng.random() < 0.34:
        symptoms.add(rng.choice(RELATED_NOISE[group]))
    if age_group == "older adult" and profile.name in {"Pneumonia", "Urinary Tract Infection", "Influenza"}:
        if rng.random() < 0.18:
            symptoms.add("confusion")
    if age_group == "child" and profile.name in {"Asthma", "Pneumonia", "Influenza"}:
        if rng.random() < 0.14:
            symptoms.add("rapid breathing")

    return sorted(symptoms), age_group, severity


def generate_dataset(target_cases: int = TARGET_CASES, seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    cases_per_profile = target_cases // len(PROFILES)

    for profile in PROFILES:
        generated = 0
        attempts = 0
        while generated < cases_per_profile:
            symptoms, age_group, severity = sample_case(rng, profile)
            key = (profile.name, "|".join(symptoms), age_group, severity)
            attempts += 1
            if key in seen and attempts < cases_per_profile * 50:
                continue
            seen.add(key)
            rows.append(
                {
                    "case_id": str(len(rows) + 1),
                    "symptoms": "|".join(symptoms),
                    "disease": profile.name,
                    "age_group": age_group,
                    "severity": severity,
                    "source_note": "synthetic_professional_profile_v2",
                }
            )
            generated += 1

    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["case_id"] = str(index)
    return rows


def write_dataset(path: Path = DATASET_PATH) -> None:
    rows = generate_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "symptoms", "disease", "age_group", "severity", "source_note"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} training entities: {path}")


if __name__ == "__main__":
    write_dataset()
