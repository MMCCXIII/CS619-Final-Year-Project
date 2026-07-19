import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import AGE_GROUPS


SYMPTOM_SYNONYMS = {
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "cold": "runny nose",
    "cough with phlegm": "productive cough",
    "phlegm": "mucus",
    "temperature": "fever",
    "joint pain": "severe joint pain",
    "stomach pain": "upper abdominal pain",
    "stomach burning": "burning stomach",
    "burning pee": "burning urination",
    "urine burning": "burning urination",
    "bp high": "high blood pressure",
    "high bp": "high blood pressure",
    "sugar": "excessive thirst",
    "loss of smell taste": "loss of smell",
    "breathlessness": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "head pain": "headache",
}


def parse_pipe_list(value: str | Iterable[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[|,\n;]+", value)
    else:
        raw_items = list(value)
    return [normalize_symptom(item) for item in raw_items if normalize_symptom(item)]


def normalize_symptom(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value).strip().lower())
    cleaned = cleaned.replace("-", " ")
    return SYMPTOM_SYNONYMS.get(cleaned, cleaned)


def load_cases(dataset_path) -> pd.DataFrame:
    cases = pd.read_csv(dataset_path)
    cases["symptom_list"] = cases["symptoms"].apply(parse_pipe_list)
    cases["age_group"] = cases["age_group"].str.strip().str.lower()
    cases["disease"] = cases["disease"].str.strip()
    return cases


def build_vocabulary(cases: pd.DataFrame) -> list[str]:
    vocabulary: set[str] = set()
    for symptom_list in cases["symptom_list"]:
        vocabulary.update(symptom_list)
    return sorted(vocabulary)


def extract_symptoms_from_text(text: str, vocabulary: Iterable[str] | None = None) -> list[str]:
    normalized_text = normalize_symptom(text)
    if not normalized_text:
        return []

    direct_tokens = parse_pipe_list(normalized_text)
    if not vocabulary:
        return sorted({item for item in direct_tokens if item})

    vocabulary_set = {normalize_symptom(symptom) for symptom in vocabulary}
    found: set[str] = set()

    for token in direct_tokens:
        if token in vocabulary_set:
            found.add(token)
        canonical = SYMPTOM_SYNONYMS.get(token)
        if canonical in vocabulary_set:
            found.add(canonical)

    for symptom in vocabulary_set:
        if symptom in normalized_text:
            found.add(symptom)
    for synonym, canonical in SYMPTOM_SYNONYMS.items():
        if synonym in normalized_text:
            if canonical in vocabulary_set:
                found.add(canonical)

    return sorted(item for item in found if item)


def normalize_conditions(values: str | Iterable[str]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = re.split(r"[|,\n;]+", values)
    else:
        raw_values = list(values)
    return sorted({normalize_symptom(value) for value in raw_values if normalize_symptom(value)})


def vectorize_symptoms(
    symptom_lists: Iterable[Iterable[str]],
    vocabulary: list[str],
    age_groups: list[str] | None = None,
    supplied_age_groups: Iterable[str] | None = None,
) -> np.ndarray:
    age_groups = age_groups or AGE_GROUPS
    symptom_lists = list(symptom_lists)
    if supplied_age_groups is None:
        supplied_age_groups = ["adult"] * len(symptom_lists)
    else:
        supplied_age_groups = list(supplied_age_groups)
    rows: list[list[float]] = []

    for index, symptoms in enumerate(symptom_lists):
        symptom_set = {normalize_symptom(symptom) for symptom in symptoms}
        symptom_features = [1.0 if symptom in symptom_set else 0.0 for symptom in vocabulary]
        age_group = normalize_symptom(supplied_age_groups[index])
        age_features = [1.0 if age_group == option else 0.0 for option in age_groups]
        rows.append(symptom_features + age_features)

    return np.asarray(rows, dtype=float)


def feature_names(vocabulary: list[str], age_groups: list[str] | None = None) -> list[str]:
    age_groups = age_groups or AGE_GROUPS
    return [f"symptom::{item}" for item in vocabulary] + [f"age::{item}" for item in age_groups]
