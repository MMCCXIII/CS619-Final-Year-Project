from src.preprocessing import normalize_symptom


BODY_AREA_SYMPTOMS = {
    "Head and nerves": [
        "headache",
        "severe headache",
        "one sided headache",
        "dizziness",
        "light sensitivity",
        "visual aura",
        "confusion",
        "seizure",
    ],
    "Chest and breathing": [
        "cough",
        "dry cough",
        "productive cough",
        "wheezing",
        "shortness of breath",
        "chest pain",
        "chest tightness",
        "rapid breathing",
    ],
    "Stomach and digestion": [
        "nausea",
        "vomiting",
        "diarrhea",
        "abdominal cramps",
        "upper abdominal pain",
        "heartburn",
        "constipation",
        "black stool",
    ],
    "Skin and allergy": [
        "rash",
        "itching",
        "redness",
        "swelling",
        "hives",
        "dry skin",
        "blisters",
        "facial swelling",
    ],
    "Urinary and reproductive": [
        "burning urination",
        "urinary urgency",
        "frequent urination",
        "cloudy urine",
        "blood in urine",
        "pelvic pain",
        "back pain",
        "pregnancy",
    ],
    "Eye, ear, nose, throat": [
        "sore throat",
        "runny nose",
        "nasal congestion",
        "sneezing",
        "red eye",
        "eye irritation",
        "ear pain",
        "ear pressure",
    ],
}

BODY_AREA_KEYWORDS = {
    "Head and nerves": (
        "headache",
        "head",
        "dizz",
        "vertigo",
        "spinning",
        "balance",
        "confusion",
        "seizure",
        "aura",
        "neck",
        "numb",
        "tingling",
        "weakness",
        "speech",
        "consciousness",
        "neurolog",
        "migraine",
        "drool",
        "droop",
        "memory",
        "shaking",
        "mening",
        "jaw",
        "facial droop",
        "slurred",
        "awareness",
        "light sensitivity",
        "visual aura",
        "pressure around head",
        "pain behind eyes",
        "worst headache",
        "morning headache",
        "dull headache",
        "one sided headache",
        "throbbing",
        "asking repetition",
        "fear of water",
        "post seizure",
        "brief shaking",
        "loss of interest",
        "low mood",
        "anxiety",
        "excess worry",
        "stress",
        "irritability",
        "restless",
        "sleep disturbance",
        "difficulty sleeping",
        "insomnia",
        "poor sleep",
        "restless sleep",
        "deep sleep",
        "early awakening",
        "sleep change",
    ),
    "Chest and breathing": (
        "cough",
        "wheez",
        "breath",
        "chest",
        "lung",
        "phlegm",
        "mucus",
        "stridor",
        "palpitation",
        "oxygen",
        "heart",
        "cardiovascular",
        "paroxysmal",
        "barking",
        "whooping",
        "hemoptysis",
        "crushing",
        "left arm",
        "snoring",
        "cardiac",
        "exertion",
        "irregular heartbeat",
        "trouble breathing",
        "rapid breathing",
        "low oxygen",
        "blue lips",
        "productive cough",
        "dry cough",
        "night cough",
        "persistent cough",
        "chronic cough",
        "post cough",
        "coughing blood",
        "recurrent wheeze",
        "shortness of breath",
        "chest tightness",
        "chest pressure",
        "chest discomfort",
        "pain with exertion",
    ),
    "Stomach and digestion": (
        "nausea",
        "vomit",
        "diarrhea",
        "abdominal",
        "stomach",
        "bowel",
        "constipation",
        "heartburn",
        "reflux",
        "indigestion",
        "belch",
        "gas",
        "bloat",
        "appetite",
        "jaundice",
        "black stool",
        "rectal",
        "anal",
        "stool",
        "regurgitation",
        "sour taste",
        "burning stomach",
        "upper abdominal",
        "lower abdominal",
        "right lower abdominal",
        "left lower abdominal",
        "guarding",
        "hard stool",
        "straining",
        "infrequent bowel",
        "changed bowel",
        "painful bowel",
        "early fullness",
        "pain after fatty",
        "pain worse bending",
        "drawing legs up",
        "evening fussiness",
        "poor feeding",
        "prolonged crying",
        "crying",
        "difficulty eating",
        "loss of appetite",
        "dark urine",
        "fruity breath",
        "greasy scaling",
        "sensitivity to sweets",
        "bloating after dairy",
        "diarrhea after dairy",
        "watery diarrhea",
        "bloody diarrhea",
        "profuse watery",
        "post cough vomiting",
        "morning nausea",
        "burning mouth",
        "burning tongue",
        "mouth sore",
        "mouth pain",
        "bad breath",
        "altered taste",
        "loss of taste",
        "white mouth",
        "chewing objects",
    ),
    "Skin and allergy": (
        "rash",
        "skin",
        "itch",
        "hive",
        "blister",
        "acne",
        "pimple",
        "blackhead",
        "scaling",
        "burn pain",
        "burning skin",
        "bite",
        "welt",
        "redness",
        "eczema",
        "ring shaped",
        "honey colored",
        "diaper area",
        "foot rash",
        "hand rash",
        "peeling",
        "sun exposure",
        "burrow",
        "infestation",
        "nits",
        "vesicle",
        "allergy",
        "warm skin",
        "warm red skin",
        "skin tenderness",
        "skin swelling",
        "skin redness",
        "skin crack",
        "skin sore",
        "localized swelling",
        "swelling after injury",
        "rapidly spreading redness",
        "red painful skin",
        "fine rash",
        "itchy rash",
        "one sided rash",
        "rash after exposure",
        "silvery scale",
        "thick plaque",
        "oily skin",
        "dry skin",
        "dry itchy skin",
        "dandruff",
        "scalp itching",
        "night itching",
        "scratch mark",
        "animal bite",
        "bite mark",
        "tingling before rash",
        "painful grouped blister",
        "blistering",
        "hot dry skin",
        "facial redness",
        "facial swelling",
        "visible facial vessel",
        "xanthelasma",
        "pus collection",
        "dirty wound",
        "slow wound healing",
        "worm related",
        "visible tiny worm",
        "anal itching",
    ),
    "Urinary and reproductive": (
        "urin",
        "bladder",
        "pelvis",
        "pelvic",
        "vaginal",
        "testicular",
        "scrotal",
        "pregnancy",
        "menstrual",
        "period",
        "erection",
        "intercourse",
        "flank",
        "groin",
        "foamy urine",
        "cloudy urine",
        "burning urination",
        "painful urination",
        "hesitancy",
        "discharge",
        "fishy odor",
        "frequent urination",
        "night urination",
        "nighttime urination",
        "very low urine",
        "weak urine stream",
        "blood in urine",
        "foul smelling urine",
        "pain radiating to groin",
        "severe flank",
        "severe pelvic",
        "pressure in lower belly",
        "sexual performance",
        "difficulty maintaining erection",
        "painful period",
        "irregular period",
        "abnormal discharge",
        "thin discharge",
        "thick white discharge",
        "wet bed",
        "waist weight gain",
        "excess hair growth",
        "hot flash",
    ),
    "Eye, ear, nose, throat": (
        "eye",
        "ear",
        "nose",
        "throat",
        "nasal",
        "sneeze",
        "tonsil",
        "hoarse",
        "voice",
        "tooth",
        "mouth",
        "conjunctiv",
        "runny nose",
        "facial pressure",
        "post nasal",
        "itchy nose",
        "itchy throat",
        "sore throat",
        "throat pain",
        "throat irritation",
        "painful swallowing",
        "otitis",
        "hearing",
        "eyelid",
        "halo",
        "gritty",
        "koplik",
        "gum",
        "parotid",
        "nasal congestion",
        "nasal dryness",
        "nasal stuffiness",
        "thick nasal",
        "clear nasal",
        "loss of smell",
        "reduced smell",
        "red eye",
        "red eyes",
        "watery eyes",
        "itchy eyes",
        "eye discharge",
        "eye irritation",
        "eye tearing",
        "burning eyes",
        "dry eyes",
        "foreign body sensation",
        "severe eye pain",
        "vision loss",
        "slow vision change",
        "blurred vision",
        "cloudy vision",
        "glare",
        "ear pain",
        "ear pressure",
        "ear fullness",
        "ear canal",
        "ear discharge",
        "ringing ear",
        "reduced hearing",
        "hearing change",
        "pain pulling ear",
        "jaw pain",
        "bad breath",
        "recent nose picking",
        "bleeding from nose",
        "swollen tonsil",
        "voice loss",
        "painful mouth sore",
        "visible cavity",
        "burning tongue",
        "burning mouth",
    ),
}

BODY_AREA_EXPLICIT = {
    area: {normalize_symptom(symptom) for symptom in symptoms}
    for area, symptoms in BODY_AREA_SYMPTOMS.items()
}


def filter_symptoms_for_body_area(body_area: str, vocabulary: list[str]) -> list[str]:
    if not body_area or body_area == "Not sure":
        return list(vocabulary)

    explicit = BODY_AREA_EXPLICIT.get(body_area, set())
    keywords = BODY_AREA_KEYWORDS.get(body_area, ())
    filtered: list[str] = []
    for item in vocabulary:
        normalized = normalize_symptom(item)
        if normalized in explicit or any(keyword in normalized for keyword in keywords):
            filtered.append(item)
    return sorted(filtered, key=str.lower)


def body_area_suggestions(body_area: str, vocabulary: list[str]) -> list[str]:
    vocabulary_set = {normalize_symptom(item) for item in vocabulary}
    return [item for item in BODY_AREA_SYMPTOMS.get(body_area, []) if normalize_symptom(item) in vocabulary_set]


SMART_SYMPTOM_SUGGESTIONS = {
    "fever": ["chills", "body aches", "headache", "rash", "vomiting", "cough"],
    "high fever": ["chills", "severe weakness", "rash", "persistent vomiting", "confusion"],
    "cough": ["sore throat", "runny nose", "chest tightness", "wheezing", "fever"],
    "headache": ["nausea", "light sensitivity", "dizziness", "neck stiffness", "visual aura"],
    "rash": ["itching", "fever", "swelling", "redness", "blisters"],
    "diarrhea": ["vomiting", "abdominal cramps", "dehydration signs", "fever", "nausea"],
    "vomiting": ["nausea", "diarrhea", "abdominal cramps", "severe dehydration", "fever"],
    "burning urination": ["urinary urgency", "frequent urination", "cloudy urine", "blood in urine", "back pain"],
    "shortness of breath": ["chest pain", "wheezing", "rapid breathing", "low oxygen", "chest tightness"],
    "sore throat": ["fever", "cough", "runny nose", "painful swallowing", "hoarse voice"],
}

RED_FLAG_QUESTIONS = [
    ("chest pain", "Chest pain or pressure"),
    ("shortness of breath", "Shortness of breath or trouble breathing"),
    ("confusion", "Confusion, fainting, or severe weakness"),
    ("severe dehydration", "Very low urine, sunken eyes, or severe dehydration"),
    ("bleeding gums", "Bleeding gums, black stool, or unusual bleeding"),
    ("blood in urine", "Blood in urine"),
    ("severe abdominal pain", "Severe abdominal pain"),
    ("low oxygen", "Blue lips or low oxygen reading"),
    ("pregnancy", "Pregnancy with fever, pain, or urinary symptoms"),
]

INTERACTION_RULES = [
    {
        "current_terms": {"warfarin", "apixaban", "rivaroxaban", "clopidogrel", "blood thinner", "anticoagulant"},
        "suggested_terms": {"nsaid", "aspirin", "ibuprofen", "naproxen", "ketorolac", "celecoxib", "antiplatelet", "anticoagulant"},
        "severity": "high",
        "message": "Blood thinner or antiplatelet use can raise bleeding risk with NSAIDs, aspirin, or similar blood-thinning medicines.",
    },
    {
        "current_terms": {"insulin", "glipizide", "gliclazide", "diabetes medicine"},
        "suggested_terms": {"insulin", "glipizide", "gliclazide", "diabetes therapy"},
        "severity": "medium",
        "message": "Diabetes medicines can overlap and may increase low blood sugar risk without clinician dosing guidance.",
    },
    {
        "current_terms": {"lisinopril", "losartan", "ace inhibitor", "arb"},
        "suggested_terms": {"nsaid", "ibuprofen", "naproxen", "ketorolac", "celecoxib"},
        "severity": "medium",
        "message": "NSAIDs may be unsuitable with some blood pressure or kidney-related medicines.",
    },
    {
        "current_terms": {"sertraline", "fluoxetine", "escitalopram", "duloxetine", "ssri", "snri"},
        "suggested_terms": {"nsaid", "aspirin", "ibuprofen", "naproxen"},
        "severity": "medium",
        "message": "Some antidepressants can increase bleeding risk when combined with NSAIDs or aspirin.",
    },
    {
        "current_terms": {"propranolol", "atenolol", "metoprolol", "beta blocker"},
        "suggested_terms": {"salbutamol", "bronchodilator", "beta blocker"},
        "severity": "medium",
        "message": "Beta blockers and breathing medicines can require careful clinician review in asthma or wheezing patterns.",
    },
]


def smart_symptom_suggestions(selected_symptoms: list[str], vocabulary: list[str]) -> list[str]:
    vocabulary_set = {normalize_symptom(item) for item in vocabulary}
    selected = {normalize_symptom(item) for item in selected_symptoms}
    suggestions = []
    for symptom in selected:
        suggestions.extend(SMART_SYMPTOM_SUGGESTIONS.get(symptom, []))
    return sorted({item for item in suggestions if normalize_symptom(item) in vocabulary_set and normalize_symptom(item) not in selected})


def red_flags_from_answers(red_flag_answers: dict[str, bool]) -> list[str]:
    return sorted({normalize_symptom(flag) for flag, present in red_flag_answers.items() if present})


def build_privacy_points() -> list[str]:
    return [
        "Recommendation logs and feedback are stored in the local SQLite database for this project.",
        "The saved safety profile lives only in the current Streamlit session unless you clear or restart it.",
        "Use the History tab to clear recommendation logs, feedback, or both.",
    ]


def _text_matches_terms(text: str, terms: set[str]) -> bool:
    normalized = normalize_symptom(text)
    return any(term in normalized for term in terms)


def screen_current_medicines(
    current_medicines: list[str] | str,
    suggested_medicines: list[dict],
    allergies: list[str] | str,
) -> list[dict]:
    if isinstance(current_medicines, str):
        current_items = [item.strip() for item in current_medicines.replace("\n", ",").split(",") if item.strip()]
    else:
        current_items = list(current_medicines or [])

    if isinstance(allergies, str):
        allergy_items = [item.strip() for item in allergies.replace("\n", ",").split(",") if item.strip()]
    else:
        allergy_items = list(allergies or [])

    alerts: list[dict] = []
    for suggested in suggested_medicines:
        suggested_text = " ".join(
            [
                str(suggested.get("medicine", "")),
                str(suggested.get("category", "")),
                str(suggested.get("safety_notes", "")),
            ]
        )
        for allergy in allergy_items:
            allergy_key = normalize_symptom(allergy)
            if allergy_key and _text_matches_terms(suggested_text, {allergy_key}):
                alerts.append(
                    {
                        "type": "Allergy match",
                        "severity": "high",
                        "medicine": suggested.get("medicine", ""),
                        "reason": f"Suggested medicine may match the listed allergy: {allergy}.",
                    }
                )
        for current in current_items:
            for rule in INTERACTION_RULES:
                if _text_matches_terms(current, rule["current_terms"]) and _text_matches_terms(
                    suggested_text, rule["suggested_terms"]
                ):
                    alerts.append(
                        {
                            "type": "Current medicine review",
                            "severity": rule["severity"],
                            "medicine": suggested.get("medicine", ""),
                            "reason": rule["message"],
                        }
                    )

    unique_alerts = []
    seen = set()
    for alert in alerts:
        key = (alert["type"], alert["severity"], alert["medicine"], alert["reason"])
        if key in seen:
            continue
        seen.add(key)
        unique_alerts.append(alert)
    return unique_alerts


def build_medicine_comparison_rows(medicines: list[dict], limit: int = 3) -> list[dict]:
    return [
        {
            "Medicine": row["medicine"],
            "Condition": row["disease"],
            "Rank score": f"{float(row['score']):.3f}",
            "Symptom fit": f"{float(row.get('therapeutic_fit', 0.0)):.0%}",
            "Prescription": "Required" if row["prescription_required"] else "Not listed",
            "Safety status": row["status"],
            "Warnings": row["warnings"] or "None",
            "Why suggested": row["rationale"],
        }
        for row in medicines[:limit]
    ]
