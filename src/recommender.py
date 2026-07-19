from pathlib import Path

from src.config import DATABASE_PATH, SAFETY_RED_FLAGS
from src.data_access import (
    fetch_contraindications,
    fetch_feedback_penalties,
    fetch_medicine,
    fetch_recommendation_catalog,
    log_recommendation,
)
from src.knowledge_base import build_disease_profiles
from src.modeling import load_model_bundle, predict_diseases
from src.preprocessing import extract_symptoms_from_text, normalize_conditions, normalize_symptom
from src.safety import screen_current_medicines


DISEASE_PROFILES = {profile.name: profile for profile in build_disease_profiles()}
MIN_ALTERNATIVE_CONFIDENCE = 0.18
MIN_CONFIDENCE_FOR_MEDICINES = 0.20
MIN_MEDICINE_FIT = 0.35

PAIN_OR_FEVER_SYMPTOMS = {
    "fever",
    "high fever",
    "mild fever",
    "low grade fever",
    "body aches",
    "muscle pain",
    "joint pain",
    "severe joint pain",
    "headache",
    "mild headache",
    "severe headache",
    "one sided headache",
    "throbbing pain",
    "sore throat",
    "throat pain",
    "painful swallowing",
    "ear pain",
    "tooth pain",
    "localized pain",
    "pelvic pain",
    "lower abdominal pain",
    "upper abdominal pain",
    "back pain",
    "low back pain",
    "neck pain",
    "burn pain",
    "painful skin",
}

DIRECT_ANALGESIC_SYMPTOMS = {
    "fever",
    "high fever",
    "mild fever",
    "low grade fever",
    "body aches",
    "muscle pain",
    "joint pain",
    "severe joint pain",
    "headache",
    "mild headache",
    "severe headache",
    "one sided headache",
    "throbbing pain",
    "ear pain",
    "tooth pain",
    "pelvic pain",
    "back pain",
    "low back pain",
    "neck pain",
    "burn pain",
    "painful skin",
}

THROAT_PAIN_SYMPTOMS = {"sore throat", "throat pain", "painful swallowing"}

NASAL_ALLERGY_SYMPTOMS = {
    "runny nose",
    "nasal congestion",
    "sneezing",
    "itchy eyes",
    "watery eyes",
    "itchy nose",
    "itchy throat",
    "post nasal drip",
    "clear nasal discharge",
    "seasonal trigger",
    "allergy trigger",
    "hives",
    "itching",
    "rash",
    "welts",
}

ALLERGY_PATTERN_SYMPTOMS = {
    "itchy eyes",
    "watery eyes",
    "itchy nose",
    "itchy throat",
    "seasonal trigger",
    "allergy trigger",
    "hives",
    "itching",
    "welts",
}

COUGH_SYMPTOMS = {
    "cough",
    "dry cough",
    "productive cough",
    "persistent cough",
    "mild cough",
    "night cough",
    "paroxysmal cough",
    "whooping cough",
}

MUCUS_SYMPTOMS = {"productive cough", "mucus", "phlegm"}

BREATHING_SYMPTOMS = {
    "wheezing",
    "shortness of breath",
    "severe shortness of breath",
    "shortness of breath on exertion",
    "chest tightness",
    "rapid breathing",
    "low oxygen",
    "night cough",
    "difficulty speaking",
    "recurrent wheeze",
}

ACID_SYMPTOMS = {
    "heartburn",
    "acid reflux",
    "regurgitation",
    "sour taste",
    "burning stomach",
    "chest burning",
    "upper abdominal pain",
    "indigestion",
    "worse after meals",
}

HYDRATION_SYMPTOMS = {
    "fever",
    "high fever",
    "vomiting",
    "persistent vomiting",
    "diarrhea",
    "watery diarrhea",
    "profuse watery diarrhea",
    "severe dehydration",
    "dehydration signs",
    "very low urine",
    "dry mouth",
    "sweating",
}

NAUSEA_SYMPTOMS = {"nausea", "vomiting", "persistent vomiting", "morning nausea"}

DIARRHEA_SYMPTOMS = {
    "diarrhea",
    "watery diarrhea",
    "profuse watery diarrhea",
    "bloody diarrhea",
    "diarrhea after dairy",
}

CONSTIPATION_SYMPTOMS = {
    "constipation",
    "hard stool",
    "straining",
    "infrequent bowel movement",
}

URINARY_SYMPTOMS = {
    "burning urination",
    "painful urination",
    "urinary urgency",
    "frequent urination",
    "cloudy urine",
    "blood in urine",
    "foul smelling urine",
    "night urination",
    "weak urine stream",
    "hesitancy",
    "flank pain",
    "pain radiating to groin",
}

SKIN_SYMPTOMS = {
    "rash",
    "itching",
    "redness",
    "swelling",
    "skin tenderness",
    "dry skin",
    "dry itchy skin",
    "blisters",
    "itchy blisters",
    "painful blisters",
    "scaling",
    "warm skin",
    "warm red skin",
    "pimples",
    "blackheads",
    "oily skin",
    "ring shaped rash",
    "honey colored crust",
    "skin sores",
    "diaper area rash",
}

EYE_SYMPTOMS = {
    "eye irritation",
    "red eye",
    "watery eyes",
    "eye discharge",
    "gritty feeling",
    "dry eyes",
    "burning eyes",
    "foreign body sensation",
    "severe eye pain",
    "vision loss",
    "halos around lights",
}

EAR_SYMPTOMS = {
    "ear pressure",
    "ear pain",
    "ear fullness",
    "hearing change",
    "ear canal pain",
    "ear discharge",
    "reduced hearing",
}

VAGINAL_SYMPTOMS = {
    "vaginal itching",
    "vaginal discharge",
    "abnormal discharge",
    "vaginal irritation",
    "thin discharge",
    "fishy odor",
    "thick white discharge",
}

CHRONIC_DISEASE_KEYWORDS = {
    "hypertension",
    "diabetes",
    "hyperlipidemia",
    "hypothyroidism",
    "hyperthyroidism",
    "anemia",
    "heart failure",
    "angina",
    "atrial fibrillation",
    "chronic kidney disease",
    "gout",
    "epilepsy",
    "depression",
    "anxiety",
    "insomnia",
}

SEVERE_SYMPTOM_ESCALATIONS = {
    "abdominal pain": {"severe abdominal pain"},
    "lower abdominal pain": {"severe abdominal pain"},
    "upper abdominal pain": {"severe abdominal pain"},
    "pelvic pain": {"severe pelvic pain"},
    "flank pain": {"severe flank pain"},
    "headache": {"severe headache"},
    "shortness of breath": {"severe shortness of breath", "low oxygen"},
    "dehydration signs": {"severe dehydration"},
    "diarrhea": {"severe dehydration"},
    "vomiting": {"persistent vomiting", "severe dehydration"},
    "joint pain": {"severe joint pain"},
    "weakness": {"severe weakness"},
}

PERSISTENT_SYMPTOM_ESCALATIONS = {
    "cough": {"persistent cough", "chronic cough"},
    "dry cough": {"persistent cough"},
    "productive cough": {"persistent cough"},
    "vomiting": {"persistent vomiting"},
    "diarrhea": {"changed bowel habits"},
    "rash": {"persistent rash"},
    "headache": {"morning headache"},
    "wheezing": {"recurrent wheeze"},
}

SAFETY_PENALTIES = {
    "high": 0.72,
    "medium": 0.30,
    "low": 0.12,
}


def build_symptom_profile(
    selected_symptoms: list[str] | None,
    free_text: str,
    vocabulary: list[str],
) -> list[str]:
    selected = [normalize_symptom(symptom) for symptom in (selected_symptoms or [])]
    extracted = extract_symptoms_from_text(free_text or "", vocabulary)
    return sorted({symptom for symptom in selected + extracted if symptom})


def detect_red_flags(symptoms: list[str], profile_conditions: list[str]) -> list[str]:
    symptom_set = {normalize_symptom(symptom) for symptom in symptoms}
    condition_set = {normalize_symptom(condition) for condition in profile_conditions}
    flags = sorted((symptom_set | condition_set) & SAFETY_RED_FLAGS)
    if "pregnancy" in condition_set and {"fever", "burning urination", "back pain"} & symptom_set:
        flags.append("pregnancy with urinary or fever symptoms")
    return sorted(set(flags))


def _normalize_set(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    return {normalize_symptom(value) for value in values if normalize_symptom(value)}


def _split_profile_text(values: list[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [
            item.strip()
            for item in values.replace("\n", ",").replace(";", ",").split(",")
            if item.strip()
        ]
    return [str(item).strip() for item in values if str(item).strip()]


def _conditions_from_allergies(allergies: list[str] | str | None) -> set[str]:
    allergy_text = " ".join(normalize_symptom(item) for item in _split_profile_text(allergies))
    conditions: set[str] = set()
    if not allergy_text:
        return conditions
    if any(term in allergy_text for term in ("penicillin", "amoxicillin", "augmentin")):
        conditions.add("penicillin allergy")
        conditions.add("antibiotic allergy")
    if any(term in allergy_text for term in ("sulfa", "sulfonamide", "sulfamethoxazole")):
        conditions.add("sulfa allergy")
        conditions.add("antibiotic allergy")
    if "antibiotic" in allergy_text:
        conditions.add("antibiotic allergy")
    return conditions


def _conditions_from_current_medicines(current_medicines: list[str] | str | None) -> set[str]:
    medicine_text = " ".join(normalize_symptom(item) for item in _split_profile_text(current_medicines))
    conditions: set[str] = set()
    if not medicine_text:
        return conditions
    if any(term in medicine_text for term in ("warfarin", "apixaban", "rivaroxaban", "clopidogrel", "blood thinner", "anticoagulant")):
        conditions.update({"anticoagulant use", "bleeding risk"})
    if any(term in medicine_text for term in ("insulin", "glipizide", "gliclazide")):
        conditions.add("hypoglycemia")
    return conditions


def _severity_for_symptom(symptom: str, symptom_severity: dict[str, str] | None) -> str:
    if not symptom_severity:
        return ""
    normalized = normalize_symptom(symptom)
    direct = normalize_symptom(symptom_severity.get(symptom, ""))
    normalized_direct = normalize_symptom(symptom_severity.get(normalized, ""))
    overall = normalize_symptom(symptom_severity.get("overall", ""))
    return direct or normalized_direct or overall


def _has_severe_symptoms(symptom_severity: dict[str, str] | None) -> bool:
    return "severe" in _severity_values(symptom_severity)


def augment_symptoms_for_context(
    symptoms: list[str],
    symptom_severity: dict[str, str] | None,
    symptom_duration: str,
    vocabulary: list[str],
) -> list[str]:
    vocabulary_set = {normalize_symptom(item) for item in vocabulary}
    augmented = _normalize_set(symptoms)
    duration_key = normalize_symptom(symptom_duration)
    persistent_duration = duration_key in {"more than a week", "recurring or ongoing"}

    for symptom in list(augmented):
        if _severity_for_symptom(symptom, symptom_severity) == "severe":
            candidates = set(SEVERE_SYMPTOM_ESCALATIONS.get(symptom, set()))
            candidates.add(f"severe {symptom}")
            augmented.update(candidate for candidate in candidates if candidate in vocabulary_set)
        if persistent_duration:
            candidates = PERSISTENT_SYMPTOM_ESCALATIONS.get(symptom, set())
            augmented.update(candidate for candidate in candidates if candidate in vocabulary_set)

    return sorted(augmented)


def _needs_context_review(
    red_flags: list[str] | set[str] | None,
    symptom_severity: dict[str, str] | None,
    symptom_duration: str,
) -> bool:
    duration_key = normalize_symptom(symptom_duration)
    return bool(red_flags) or _has_severe_symptoms(symptom_severity) or duration_key in {
        "more than a week",
        "recurring or ongoing",
    }


def _is_medical_referral(item: dict) -> bool:
    return normalize_symptom(item.get("medicine", "")) == "medical referral"


def _alert_penalty(alerts: list[dict]) -> tuple[float, bool]:
    penalty = 0.0
    blocked = False
    for alert in alerts:
        severity = normalize_symptom(alert.get("severity", "low"))
        penalty += SAFETY_PENALTIES.get(severity, SAFETY_PENALTIES["low"])
        if severity == "high":
            blocked = True
    return penalty, blocked


def disease_profile_match(disease: str, symptoms: list[str], age_group: str = "adult") -> dict:
    profile = DISEASE_PROFILES.get(disease)
    symptom_set = _normalize_set(symptoms)
    if not profile or not symptom_set:
        return {"score": 0.0, "matched_symptoms": []}

    core = _normalize_set(profile.core_symptoms)
    supporting = _normalize_set(profile.supporting_symptoms)
    red_flags = _normalize_set(profile.red_flag_symptoms)
    core_hits = symptom_set & core
    supporting_hits = (symptom_set & supporting) - core_hits
    red_flag_hits = (symptom_set & red_flags) - core_hits - supporting_hits
    matched = core_hits | supporting_hits | red_flag_hits

    weighted_hits = len(core_hits) + (0.55 * len(supporting_hits)) + (0.75 * len(red_flag_hits))
    expected_evidence = max(2.5, min(float(len(symptom_set)), 6.0))
    input_coverage = min(weighted_hits / expected_evidence, 1.0)
    core_coverage = min(len(core_hits) / max(1, min(3, len(core))), 1.0)
    supporting_coverage = min(len(supporting_hits) / max(1, min(4, len(supporting))), 1.0)
    red_flag_coverage = min(len(red_flag_hits) / max(1, min(2, len(red_flags))), 1.0)
    disease_coverage = min(
        (0.65 * core_coverage) + (0.25 * supporting_coverage) + (0.10 * red_flag_coverage),
        1.0,
    )
    age_fit = profile.age_weights.get(normalize_symptom(age_group), 0.33)
    age_score = min(age_fit / max(profile.age_weights.values()), 1.0)
    score = min((0.55 * input_coverage) + (0.35 * disease_coverage) + (0.10 * age_score), 1.0)

    return {
        "score": round(score, 4),
        "matched_symptoms": sorted(matched),
    }


def calibrate_disease_predictions(
    predicted_diseases: list[dict],
    symptoms: list[str],
    age_group: str = "adult",
) -> list[dict]:
    if not predicted_diseases:
        return []

    top_raw_confidence = max(float(row["confidence"]) for row in predicted_diseases)
    symptom_count = len(_normalize_set(symptoms))
    evidence_strength = min(symptom_count / 4.0, 1.0)
    calibrated = []

    for row in predicted_diseases:
        raw_confidence = float(row["confidence"])
        match = disease_profile_match(row["disease"], symptoms, age_group)
        relative_support = 1.0
        if top_raw_confidence > 0:
            relative_support = min((raw_confidence / top_raw_confidence) ** 0.5, 1.0)
        boost = 0.48 * evidence_strength * relative_support
        adjusted_confidence = raw_confidence + ((1.0 - raw_confidence) * match["score"] * boost)
        calibrated.append(
            {
                **row,
                "confidence": round(min(adjusted_confidence, 0.99), 4),
                "model_confidence": round(raw_confidence, 4),
                "profile_match": match["score"],
                "matched_symptoms": match["matched_symptoms"],
            }
        )

    return sorted(
        calibrated,
        key=lambda row: (
            float(row["confidence"]),
            float(row.get("profile_match", 0.0)),
            float(row.get("model_confidence", 0.0)),
        ),
        reverse=True,
    )


def _select_eligible_diseases(predicted_diseases: list[dict], max_diseases: int) -> list[dict]:
    if not predicted_diseases:
        return []

    selected = [predicted_diseases[0]]
    top_confidence = float(predicted_diseases[0]["confidence"])
    min_confidence = max(MIN_ALTERNATIVE_CONFIDENCE, top_confidence * 0.50)

    for row in predicted_diseases[1:]:
        if len(selected) >= max_diseases:
            break
        if float(row["confidence"]) < min_confidence:
            continue
        if float(row.get("profile_match", 0.0)) < 0.45:
            continue
        selected.append(row)

    return selected


def _has_any(symptom_set: set[str], candidates: set[str]) -> bool:
    return bool(symptom_set & candidates)


def medicine_symptom_fit(item: dict, disease: str, symptoms: list[str]) -> float:
    symptom_set = _normalize_set(symptoms)
    disease_key = normalize_symptom(disease)
    medicine = normalize_symptom(item["medicine"])
    category = normalize_symptom(item["category"])

    if medicine == "medical referral":
        return 0.98
    if "emergency" in category or "vaccine" in category or "immunoglobulin" in category:
        return 0.95 if disease_key in {"severe allergic reaction", "rabies exposure", "tetanus risk wound"} else 0.62

    if medicine in {"normal saline", "ringer lactate", "oral rehydration salts"} or "hydration" in category:
        if _has_any(symptom_set, HYDRATION_SYMPTOMS) or disease_key in {"dengue fever", "cholera"}:
            return 0.95
        if _has_any(symptom_set, URINARY_SYMPTOMS):
            return 0.42
        return 0.30

    if "pain" in category or "nsaid" in category or medicine in {"acetaminophen", "ibuprofen", "naproxen"}:
        if _has_any(symptom_set, DIRECT_ANALGESIC_SYMPTOMS):
            return 0.95
        if _has_any(symptom_set, THROAT_PAIN_SYMPTOMS):
            throat_terms = {"common cold", "influenza", "covid 19", "pharyngitis", "tonsillitis", "laryngitis"}
            return 0.82 if disease_key in throat_terms else 0.24
        if any(keyword in disease_key for keyword in ("migraine", "headache", "arthritis", "gout", "pain", "burn")):
            return 0.82
        return 0.22

    if "antihistamine" in category:
        if _has_any(symptom_set, NASAL_ALLERGY_SYMPTOMS):
            return 0.98
        return 0.35

    if "nasal corticosteroid" in category:
        if _has_any(symptom_set, ALLERGY_PATTERN_SYMPTOMS) or disease_key in {"allergic rhinitis", "sinusitis"}:
            return 0.96
        return 0.52 if _has_any(symptom_set, NASAL_ALLERGY_SYMPTOMS) else 0.22

    if "nasal" in category or "decongestant" in category:
        if _has_any(symptom_set, NASAL_ALLERGY_SYMPTOMS):
            return 0.96
        return 0.22

    if "cough suppressant" in category:
        return 0.95 if _has_any(symptom_set, COUGH_SYMPTOMS) else 0.18

    if "expectorant" in category:
        if _has_any(symptom_set, MUCUS_SYMPTOMS):
            return 0.96
        return 0.62 if _has_any(symptom_set, COUGH_SYMPTOMS) else 0.18

    if "bronchodilator" in category or "inhaled corticosteroid" in category or "leukotriene" in category:
        if _has_any(symptom_set, BREATHING_SYMPTOMS):
            return 0.98
        return 0.44 if _has_any(symptom_set, COUGH_SYMPTOMS) else 0.22

    if (
        "proton pump inhibitor" in category
        or "h2 blocker" in category
        or "gastrointestinal therapy" in category
        or "mucosal protectant" in category
    ):
        if _has_any(symptom_set, ACID_SYMPTOMS):
            return 0.96
        return 0.24

    if "antiemetic" in category:
        return 0.95 if _has_any(symptom_set, NAUSEA_SYMPTOMS) else 0.24

    if "antidiarrheal" in category:
        return 0.94 if _has_any(symptom_set, DIARRHEA_SYMPTOMS) else 0.14

    if "laxative" in category or "constipation" in category:
        return 0.95 if _has_any(symptom_set, CONSTIPATION_SYMPTOMS) else 0.14

    if "antispasmodic" in category:
        return 0.88 if _has_any(symptom_set, {"abdominal cramps", "stomach cramps", "lower abdominal pain"}) else 0.24

    if "antibiotic" in category or "antimicrobial" in category or "antiprotozoal" in category:
        if medicine in {"nitrofurantoin", "fosfomycin"} and _has_any(symptom_set, URINARY_SYMPTOMS):
            return 0.96
        infection_terms = {
            "pneumonia",
            "sinusitis",
            "otitis",
            "tonsillitis",
            "pharyngitis",
            "cellulitis",
            "impetigo",
            "typhoid",
            "cholera",
            "prostatitis",
            "pyelonephritis",
            "pelvic inflammatory disease",
            "bacterial vaginosis",
            "conjunctivitis",
            "dental caries",
            "gingivitis",
        }
        return 0.82 if disease_key in infection_terms else 0.26

    if "antiviral" in category:
        viral_terms = {"influenza", "covid 19", "shingles", "herpes simplex", "chickenpox"}
        return 0.88 if disease_key in viral_terms else 0.36

    if "antimalarial" in category:
        return 0.96 if disease_key == "malaria" else 0.24

    if "antiparasitic" in category:
        parasite_terms = {"scabies", "pinworm infection", "worm infestation", "head lice"}
        return 0.92 if disease_key in parasite_terms else 0.30

    if "antifungal" in category:
        fungal_terms = {"fungal skin infection", "candidiasis", "oral thrush", "diaper rash"}
        if disease_key in fungal_terms or _has_any(symptom_set, {"ring shaped rash", "scaling", "vaginal itching"}):
            return 0.92
        return 0.28

    if "dermatology" in category or "topical" in category or "acne" in category or "barrier" in category:
        return 0.93 if _has_any(symptom_set, SKIN_SYMPTOMS) else 0.25

    if "eye" in category:
        if _has_any(symptom_set, EYE_SYMPTOMS) or any(keyword in disease_key for keyword in ("eye", "conjunctivitis", "glaucoma", "cataract", "stye")):
            return 0.94
        return 0.22

    if "ear" in category:
        if _has_any(symptom_set, EAR_SYMPTOMS) or any(keyword in disease_key for keyword in ("ear", "otitis", "tinnitus", "hearing")):
            return 0.92
        return 0.22

    if "urinary" in category or "urology" in category:
        return 0.92 if _has_any(symptom_set, URINARY_SYMPTOMS) else 0.25

    if "gynecology" in category:
        return 0.92 if _has_any(symptom_set, VAGINAL_SYMPTOMS) else 0.25

    if "migraine" in category:
        migraine_symptoms = {"one sided headache", "throbbing pain", "light sensitivity", "visual aura"}
        return 0.96 if _has_any(symptom_set, migraine_symptoms) else 0.35

    if "vertigo" in category or "motion sickness" in category:
        return 0.92 if _has_any(symptom_set, {"dizziness", "spinning sensation", "balance problem"}) else 0.30

    if (
        "cardiovascular" in category
        or "diabetes therapy" in category
        or "lipid lowering" in category
        or "endocrine therapy" in category
        or "neurology therapy" in category
        or "mental health" in category
        or "sleep support" in category
        or "vitamin and mineral" in category
    ):
        return 0.88 if any(keyword in disease_key for keyword in CHRONIC_DISEASE_KEYWORDS) else 0.42

    return 0.62


def _support_candidate_specs(disease: str, symptoms: list[str]) -> list[tuple[str, float, str]]:
    symptom_set = _normalize_set(symptoms)
    disease_key = normalize_symptom(disease)
    specs: list[tuple[str, float, str]] = []

    def add(name: str, relevance: float, rationale: str) -> None:
        specs.append((name, relevance, rationale))

    if _has_any(symptom_set, NASAL_ALLERGY_SYMPTOMS):
        add("Cetirizine", 0.70, "Symptom-specific support for sneezing, itching, runny nose, or hives.")
        add("Saline Nasal Spray", 0.48, "Supportive nasal moisture and congestion relief.")
    if _has_any(symptom_set, ALLERGY_PATTERN_SYMPTOMS) or disease_key in {"allergic rhinitis", "sinusitis"}:
        add("Fluticasone Nasal Spray", 0.66, "Nasal inflammation support when congestion or allergy symptoms are present.")

    if _has_any(symptom_set, COUGH_SYMPTOMS):
        add("Dextromethorphan", 0.48, "Symptom-specific cough support when clinically appropriate.")
    if _has_any(symptom_set, MUCUS_SYMPTOMS):
        add("Guaifenesin", 0.52, "Expectorant support for mucus or productive cough.")

    throat_terms = {"common cold", "influenza", "covid 19", "pharyngitis", "tonsillitis", "laryngitis"}
    if _has_any(symptom_set, DIRECT_ANALGESIC_SYMPTOMS) or (
        _has_any(symptom_set, THROAT_PAIN_SYMPTOMS) and disease_key in throat_terms
    ):
        add("Acetaminophen", 0.58, "Fever or pain support when label-appropriate.")

    if _has_any(symptom_set, BREATHING_SYMPTOMS):
        add("Salbutamol Inhaler", 0.64, "Breathing symptom support within a clinician-directed action plan.")
        add("Medical Referral", 0.70, "Breathing symptoms can need prompt clinical review.")

    if _has_any(symptom_set, NAUSEA_SYMPTOMS):
        add("Ondansetron", 0.56, "Nausea or vomiting support after clinician review.")
        add("Oral Rehydration Salts", 0.58, "Hydration support when nausea or vomiting is present.")

    if _has_any(symptom_set, DIARRHEA_SYMPTOMS):
        add("Oral Rehydration Salts", 0.72, "Hydration support for diarrheal fluid loss.")
        if not _has_any(symptom_set, {"bloody diarrhea", "high fever", "severe abdominal pain"}):
            add("Loperamide", 0.42, "Short-term diarrhea symptom support when red flags are absent.")

    if _has_any(symptom_set, CONSTIPATION_SYMPTOMS):
        add("Polyethylene Glycol", 0.62, "Constipation support when label-appropriate.")
        add("Psyllium Fiber", 0.50, "Fiber support for constipation patterns.")

    if _has_any(symptom_set, ACID_SYMPTOMS):
        add("Omeprazole", 0.66, "Acid reflux or burning stomach symptom support.")
        add("Famotidine", 0.58, "Acid relief support when clinically appropriate.")
        add("Antacid", 0.50, "Short-term acid symptom support.")

    if _has_any(symptom_set, URINARY_SYMPTOMS):
        add("Phenazopyridine", 0.44, "Short-term urinary discomfort support after safety review.")

    if _has_any(symptom_set, SKIN_SYMPTOMS):
        add("Hydrocortisone Cream", 0.48, "Topical itch or inflammation support for suitable rashes.")
        add("Calamine Lotion", 0.44, "Supportive itch relief for suitable skin irritation.")
    if _has_any(symptom_set, {"ring shaped rash", "scaling", "vaginal itching"}):
        add("Clotrimazole Cream", 0.56, "Antifungal support when fungal features are present.")

    if _has_any(symptom_set, EYE_SYMPTOMS):
        add("Artificial Tears", 0.52, "Supportive eye lubrication for irritation or dryness.")
        add("Olopatadine Eye Drops", 0.54, "Allergy eye symptom support when itching or watering is present.")

    if _has_any(symptom_set, EAR_SYMPTOMS):
        add("Carbamide Peroxide Ear Drops", 0.42, "Ear wax support when fullness or reduced hearing suggests wax.")

    return specs


def recommendation_items_for_disease(disease: str, symptoms: list[str], db_path: Path) -> list[dict]:
    catalog = fetch_recommendation_catalog(disease, db_path)
    existing = {normalize_symptom(row["medicine"]) for row in catalog}

    for medicine_name, relevance, rationale in _support_candidate_specs(disease, symptoms):
        medicine_key = normalize_symptom(medicine_name)
        if medicine_key in existing:
            continue
        medicine = fetch_medicine(medicine_name, db_path)
        if not medicine:
            continue
        catalog.append(
            {
                **medicine,
                "disease": disease,
                "relevance_score": relevance,
                "rationale": rationale,
            }
        )
        existing.add(medicine_key)

    return catalog


def rank_medicines(
    predicted_diseases: list[dict],
    profile_conditions: list[str],
    symptoms: list[str] | None = None,
    db_path: Path = DATABASE_PATH,
    max_diseases: int = 3,
    *,
    age_group: str = "adult",
    symptom_severity: dict[str, str] | None = None,
    symptom_duration: str = "",
    red_flags: list[str] | None = None,
    current_medicines: list[str] | str | None = None,
    allergies: list[str] | str | None = None,
) -> list[dict]:
    conditions = set(normalize_conditions(profile_conditions))
    age_key = normalize_symptom(age_group)
    if age_key:
        conditions.add(age_key)
    conditions.update(_conditions_from_current_medicines(current_medicines))
    conditions.update(_conditions_from_allergies(allergies))

    symptoms = [normalize_symptom(symptom) for symptom in (symptoms or [])]
    red_flag_set = _normalize_set(red_flags or [])
    severe_context = _has_severe_symptoms(symptom_severity)
    duration_key = normalize_symptom(symptom_duration)
    persistent_context = duration_key in {"more than a week", "recurring or ongoing"}
    needs_review = _needs_context_review(red_flag_set, symptom_severity, symptom_duration)
    eligible_diseases = _select_eligible_diseases(predicted_diseases, max_diseases)
    if not eligible_diseases or float(eligible_diseases[0]["confidence"]) < MIN_CONFIDENCE_FOR_MEDICINES:
        return []
    feedback_penalties = fetch_feedback_penalties(db_path)
    disease_names = {row["disease"].lower() for row in eligible_diseases}
    if "dengue fever" in disease_names:
        conditions.add("dengue fever")

    ranked: list[dict] = []
    for disease_index, disease_row in enumerate(eligible_diseases):
        disease = disease_row["disease"]
        confidence = float(disease_row["confidence"])
        disease_rank_weight = max(0.70, 1.0 - (0.18 * disease_index))
        for item in recommendation_items_for_disease(disease, symptoms, db_path):
            contraindications = fetch_contraindications(item["medicine"], db_path)
            matching_warnings = [
                warning
                for warning in contraindications
                if normalize_symptom(warning["condition_key"]) in conditions
            ]
            penalty = 0.0
            blocked = False
            for warning in matching_warnings:
                if warning["severity"] == "high":
                    penalty += 0.55
                    blocked = True
                elif warning["severity"] == "medium":
                    penalty += 0.25
                else:
                    penalty += 0.10

            safety_alerts = screen_current_medicines(current_medicines, [item], allergies)
            alert_penalty, alert_blocked = _alert_penalty(safety_alerts)
            penalty += alert_penalty
            blocked = blocked or alert_blocked

            therapeutic_fit = medicine_symptom_fit(item, disease, symptoms)
            if therapeutic_fit < MIN_MEDICINE_FIT and not matching_warnings and not safety_alerts:
                continue

            profile_match = float(disease_row.get("profile_match", 1.0))
            match_weight = 0.72 + (0.28 * profile_match)
            context_multiplier = 1.0
            context_notes: list[str] = []
            if _is_medical_referral(item):
                if red_flag_set:
                    context_multiplier *= 1.45
                if severe_context:
                    context_multiplier *= 1.25
                if persistent_context:
                    context_multiplier *= 1.15
            else:
                if red_flag_set:
                    context_multiplier *= 0.70
                    context_notes.append("Urgent flags are present; medicine use should not delay clinician review.")
                if severe_context and not bool(item["prescription_required"]):
                    context_multiplier *= 0.84
                    context_notes.append("Severe symptoms were marked; suitability needs clinician review.")
                if persistent_context and not bool(item["prescription_required"]):
                    context_multiplier *= 0.92
                    context_notes.append("Persistent or recurring symptoms reduce confidence in self-care-only options.")

            score = (
                confidence
                * float(item["relevance_score"])
                * float(item["effectiveness_score"])
                * therapeutic_fit
                * match_weight
                * disease_rank_weight
                * context_multiplier
            )
            feedback_penalty = feedback_penalties.get(normalize_symptom(item["medicine"]), 0.0)
            adjusted_score = max(score - penalty - feedback_penalty, 0.0)
            warning_reasons = [
                *[warning["reason"] for warning in matching_warnings],
                *[alert["reason"] for alert in safety_alerts],
                *context_notes,
            ]
            if blocked:
                status = "Safety conflict - clinician review"
            elif needs_review or bool(item["prescription_required"]):
                status = "Needs clinician review"
            else:
                status = "Candidate"
            ranked.append(
                {
                    "disease": disease,
                    "confidence": confidence,
                    "model_confidence": disease_row.get("model_confidence", confidence),
                    "profile_match": profile_match,
                    "medicine": item["medicine"],
                    "category": item["category"],
                    "prescription_required": bool(item["prescription_required"]),
                    "score": round(adjusted_score, 4),
                    "therapeutic_fit": round(therapeutic_fit, 4),
                    "context_multiplier": round(context_multiplier, 4),
                    "safety_penalty": round(penalty + feedback_penalty, 4),
                    "status": status,
                    "rationale": item["rationale"],
                    "safety_notes": item["safety_notes"],
                    "warnings": "; ".join(dict.fromkeys(warning_reasons)),
                    "safety_alerts": safety_alerts,
                }
            )

    sorted_rows = sorted(
        ranked,
        key=lambda row: (
            row["score"],
            row["therapeutic_fit"],
            row["profile_match"],
            not row["prescription_required"],
        ),
        reverse=True,
    )
    deduped_rows = []
    seen_medicines = set()
    for row in sorted_rows:
        medicine_key = normalize_symptom(row["medicine"])
        if medicine_key in seen_medicines:
            continue
        seen_medicines.add(medicine_key)
        deduped_rows.append(row)
    return deduped_rows


def _confidence_summary(predicted: list[dict]) -> str:
    if not predicted:
        return "No condition match could be estimated."
    top = predicted[0]
    confidence = float(top["confidence"])
    if confidence < MIN_CONFIDENCE_FOR_MEDICINES:
        return (
            f"Low symptom-profile match for {top['disease']}. "
            "Add more specific symptoms before considering medicine suggestions."
        )
    if confidence >= 0.75:
        band = "Strong"
    elif confidence >= 0.45:
        band = "Moderate"
    else:
        band = "Low"
    return f"{band} symptom-profile match for {top['disease']}. Review safety notes before acting."


def _severity_values(symptom_severity: dict[str, str] | None) -> list[str]:
    if not symptom_severity:
        return []
    return [normalize_symptom(value) for value in symptom_severity.values() if normalize_symptom(value)]


def determine_urgency(
    predicted: list[dict],
    medicines: list[dict],
    red_flags: list[str],
    symptom_severity: dict[str, str] | None = None,
    symptom_duration: str = "",
    age_group: str = "adult",
    profile_conditions: list[str] | None = None,
    interaction_alerts: list[dict] | None = None,
) -> dict:
    top_confidence = float(predicted[0]["confidence"]) if predicted else 0.0
    top_disease = predicted[0]["disease"] if predicted else "the entered symptoms"
    duration_key = normalize_symptom(symptom_duration)
    severities = _severity_values(symptom_severity)
    age_key = normalize_symptom(age_group)
    condition_set = set(normalize_conditions(profile_conditions or []))
    high_safety_alert = any(normalize_symptom(alert.get("severity", "")) == "high" for alert in interaction_alerts or [])

    if red_flags:
        return {
            "level": "Urgent care needed",
            "tone": "warning",
            "summary": "One or more urgent review flags were detected.",
            "next_step": "Seek urgent medical care now, especially if symptoms are worsening.",
        }

    if "severe" in severities:
        summary = "At least one symptom was marked severe."
        if age_key in {"child", "older adult"}:
            summary = f"{age_key.title()} patient with severe symptoms needs careful review."
        return {
            "level": "Doctor recommended",
            "tone": "warning",
            "summary": summary,
            "next_step": "Contact a qualified clinician for a personalized assessment.",
        }

    if condition_set & {"pregnancy", "breastfeeding"} and duration_key not in {"started today"}:
        return {
            "level": "Doctor recommended",
            "tone": "warning",
            "summary": "Pregnancy or breastfeeding was included in the safety profile.",
            "next_step": "Review symptoms and medicine options with a clinician or pharmacist.",
        }

    if duration_key in {"more than a week", "recurring or ongoing"}:
        return {
            "level": "Doctor recommended",
            "tone": "warning",
            "summary": "The symptom duration suggests follow-up is safer than self-managing only.",
            "next_step": "Arrange a clinician or pharmacist review if symptoms persist or recur.",
        }

    if high_safety_alert:
        return {
            "level": "Doctor recommended",
            "tone": "warning",
            "summary": "A high-priority allergy or current-medicine conflict was detected.",
            "next_step": "Do not use flagged medicine candidates until a clinician or pharmacist reviews them.",
        }

    if top_confidence < MIN_CONFIDENCE_FOR_MEDICINES:
        return {
            "level": "Add more detail",
            "tone": "warning",
            "summary": f"The symptom pattern is not specific enough for {top_disease}.",
            "next_step": "Add more symptoms, duration, or severity before relying on the suggestions.",
        }

    if any(row["prescription_required"] for row in medicines[:2]):
        return {
            "level": "Doctor recommended",
            "tone": "warning",
            "summary": "A top medicine candidate requires prescription or clinician supervision.",
            "next_step": "Use prescription medicines only after clinician evaluation.",
        }

    return {
        "level": "Self-care likely",
        "tone": "safe",
        "summary": "No urgent flags were detected and the top suggestions are supportive or over-the-counter.",
        "next_step": "Monitor symptoms and seek care if they worsen, persist, or new warning signs appear.",
    }


def build_care_guidance(
    symptoms: list[str],
    red_flags: list[str],
    medicines: list[dict],
    symptom_duration: str = "",
    interaction_alerts: list[dict] | None = None,
) -> list[str]:
    symptom_set = _normalize_set(symptoms)
    guidance = [
        "Monitor symptoms, hydration, temperature, and whether the overall pattern is improving.",
    ]

    if _has_any(symptom_set, HYDRATION_SYMPTOMS):
        guidance.append("Prioritize fluids or oral rehydration when vomiting, diarrhea, sweating, or fever is present.")
    if _has_any(symptom_set, NASAL_ALLERGY_SYMPTOMS):
        guidance.append("For nasal or allergy symptoms, reduce triggers and consider saline support before stronger medicines.")
    if _has_any(symptom_set, COUGH_SYMPTOMS | THROAT_PAIN_SYMPTOMS):
        guidance.append("Rest, warm fluids, and avoiding smoke or irritants can support cough and throat symptoms.")
    if _has_any(symptom_set, ACID_SYMPTOMS | NAUSEA_SYMPTOMS | DIARRHEA_SYMPTOMS | CONSTIPATION_SYMPTOMS):
        guidance.append("For digestive symptoms, use small meals or fluids and avoid foods that clearly worsen symptoms.")
    if _has_any(symptom_set, URINARY_SYMPTOMS):
        guidance.append("Urinary symptoms often need testing; seek clinician review if pain, fever, blood, pregnancy, or back pain is present.")
    if _has_any(symptom_set, SKIN_SYMPTOMS):
        guidance.append("Keep irritated skin clean, avoid scratching, and avoid new products that may have triggered the rash.")
    if _has_any(symptom_set, EYE_SYMPTOMS | EAR_SYMPTOMS):
        guidance.append("Eye or ear symptoms should be reviewed promptly if pain, discharge, hearing change, or vision change occurs.")
    if symptom_duration and normalize_symptom(symptom_duration) in {"more than a week", "recurring or ongoing"}:
        guidance.append("Persistent or recurring symptoms deserve clinician follow-up even if they seem mild.")
    if medicines:
        guidance.append("Use medicines only as directed on the label or by a clinician; prescription items require professional review.")
    if interaction_alerts:
        guidance.append("Review allergy or current-medicine alerts before using any flagged medicine candidate.")
    if red_flags:
        guidance.append("Urgent flags were detected; do not delay care while trying self-care steps.")

    return list(dict.fromkeys(guidance))[:6]


def recommend(
    selected_symptoms: list[str] | None,
    free_text: str,
    age_group: str,
    profile_conditions: list[str] | str,
    symptom_severity: dict[str, str] | None = None,
    symptom_duration: str = "",
    red_flag_answers: list[str] | None = None,
    current_medicines: list[str] | str | None = None,
    allergies: list[str] | str | None = None,
    db_path: Path = DATABASE_PATH,
    persist_log: bool = True,
    bundle: dict | None = None,
) -> dict:
    bundle = bundle or load_model_bundle()
    profile_conditions = normalize_conditions(profile_conditions)
    selected = [normalize_symptom(symptom) for symptom in (selected_symptoms or [])]
    answered_red_flags = [normalize_symptom(symptom) for symptom in (red_flag_answers or [])]
    profile_conditions = sorted({*profile_conditions, *answered_red_flags})
    extracted = extract_symptoms_from_text(free_text or "", bundle["vocabulary"])
    symptoms = sorted({symptom for symptom in selected + extracted + answered_red_flags if symptom})
    symptoms = augment_symptoms_for_context(symptoms, symptom_severity, symptom_duration, bundle["vocabulary"])
    typed_notes = bool(normalize_symptom(free_text or ""))
    input_alert = None

    if typed_notes and not extracted:
        input_alert = (
            "No recognized symptoms were found in the typed notes. "
            "Select symptoms from the list or enter known symptom terms such as fever, rash, cough, or headache."
        )

    if not symptoms:
        message = (
            "No recognized symptoms found. Please select symptoms from the list or enter known symptom terms."
            if typed_notes
            else "Enter at least one symptom to generate recommendations."
        )
        return {
            "symptoms": [],
            "red_flags": [],
            "predicted_diseases": [],
            "medicines": [],
            "message": message,
            "input_alert": input_alert,
            "validation_error": True,
        }

    red_flags = detect_red_flags(symptoms, profile_conditions)
    predicted = calibrate_disease_predictions(predict_diseases(symptoms, age_group, bundle), symptoms, age_group)
    medicines = rank_medicines(
        predicted,
        profile_conditions,
        symptoms,
        db_path,
        age_group=age_group,
        symptom_severity=symptom_severity,
        symptom_duration=symptom_duration,
        red_flags=red_flags,
        current_medicines=current_medicines,
        allergies=allergies,
    )
    interaction_alerts = screen_current_medicines(current_medicines, medicines, allergies)
    urgency = determine_urgency(
        predicted,
        medicines,
        red_flags,
        symptom_severity,
        symptom_duration,
        age_group=age_group,
        profile_conditions=profile_conditions,
        interaction_alerts=interaction_alerts,
    )
    care_guidance = build_care_guidance(symptoms, red_flags, medicines, symptom_duration, interaction_alerts)

    if persist_log and predicted:
        log_recommendation(
            symptoms=symptoms,
            age_group=age_group,
            profile_conditions=profile_conditions,
            top_disease=predicted[0]["disease"],
            model_confidence=predicted[0].get("model_confidence", predicted[0]["confidence"]),
            db_path=db_path,
        )

    return {
        "symptoms": symptoms,
        "red_flags": red_flags,
        "predicted_diseases": predicted[:5],
        "medicines": medicines,
        "urgency": urgency,
        "care_guidance": care_guidance,
        "interaction_alerts": interaction_alerts,
        "symptom_severity": symptom_severity or {},
        "symptom_duration": symptom_duration,
        "message": _confidence_summary(predicted),
        "input_alert": input_alert,
        "validation_error": False,
    }
