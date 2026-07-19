from pathlib import Path

from src.config import DATASET_PATH
from src.data_access import (
    clear_all_history,
    clear_feedback_history,
    clear_recommendation_history,
    fetch_dataframe,
    log_feedback,
    log_recommendation,
    seed_database,
)
from src.modeling import load_model_bundle, train_and_save_models
from src.preprocessing import extract_symptoms_from_text, load_cases
from src.recommender import (
    augment_symptoms_for_context,
    build_care_guidance,
    calibrate_disease_predictions,
    determine_urgency,
    rank_medicines,
    recommend,
)
from src.safety import (
    body_area_suggestions,
    build_medicine_comparison_rows,
    filter_symptoms_for_body_area,
    screen_current_medicines,
    smart_symptom_suggestions,
)


def test_symptom_extraction_uses_vocabulary():
    cases = load_cases(DATASET_PATH)
    vocabulary = sorted({symptom for symptoms in cases["symptom_list"] for symptom in symptoms})

    symptoms = extract_symptoms_from_text("high fever, rash and pain behind eyes", vocabulary)

    assert "high fever" in symptoms
    assert "rash" in symptoms
    assert "pain behind eyes" in symptoms


def test_symptom_extraction_rejects_unknown_text():
    symptoms = extract_symptoms_from_text("asdf123, 999, notarealsymptom", ["high fever", "rash"])

    assert symptoms == []


def test_recommendation_rejects_unrecognized_free_text():
    result = recommend(
        selected_symptoms=[],
        free_text="asdf123 999 notarealsymptom",
        age_group="adult",
        profile_conditions=[],
        persist_log=False,
        bundle={"vocabulary": ["high fever", "rash"], "age_groups": ["adult"]},
    )

    assert result["validation_error"] is True
    assert result["predicted_diseases"] == []
    assert "No recognized symptoms" in result["message"]


def test_dataset_has_professional_scale():
    cases = load_cases(DATASET_PATH)

    assert len(cases) >= 2000
    assert cases["disease"].nunique() >= 150


def test_database_seed_creates_core_tables(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)

    disease_count = int(fetch_dataframe("SELECT COUNT(*) AS count FROM diseases", db_path).iloc[0]["count"])
    medicine_count = int(fetch_dataframe("SELECT COUNT(*) AS count FROM medicines", db_path).iloc[0]["count"])
    rule_count = int(
        fetch_dataframe("SELECT COUNT(*) AS count FROM medicine_contraindications", db_path).iloc[0]["count"]
    )

    assert disease_count >= 150
    assert medicine_count >= 150
    assert rule_count >= 150


def test_recommendation_ranks_dengue_safely(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    seed_database(db_path)
    train_and_save_models(DATASET_PATH, model_path, metrics_path)
    bundle = load_model_bundle(model_path)

    result = recommend(
        selected_symptoms=["high fever", "rash", "severe joint pain", "pain behind eyes", "bleeding gums"],
        free_text="",
        age_group="adult",
        profile_conditions=[],
        db_path=db_path,
        persist_log=False,
        bundle=bundle,
    )

    assert result["predicted_diseases"][0]["disease"] == "Dengue Fever"
    assert "bleeding gums" in result["red_flags"]
    assert all(row["medicine"] != "Ibuprofen" for row in result["medicines"][:3])


def test_calibrated_confidence_uses_profile_evidence():
    raw_predictions = [
        {"disease": "Urinary Tract Infection", "confidence": 0.54},
        {"disease": "Dysmenorrhea", "confidence": 0.05},
    ]

    calibrated = calibrate_disease_predictions(
        raw_predictions,
        ["burning urination", "urinary urgency", "cloudy urine", "lower abdominal pain"],
        "adult",
    )

    assert calibrated[0]["disease"] == "Urinary Tract Infection"
    assert calibrated[0]["confidence"] > calibrated[0]["model_confidence"]
    assert calibrated[0]["profile_match"] > calibrated[1]["profile_match"]


def test_allergy_ranking_prioritizes_symptom_specific_medicines(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)

    medicines = rank_medicines(
        [
            {
                "disease": "Allergic Rhinitis",
                "confidence": 0.92,
                "model_confidence": 0.84,
                "profile_match": 0.84,
            }
        ],
        profile_conditions=[],
        symptoms=["sneezing", "itchy eyes", "watery eyes", "runny nose"],
        db_path=db_path,
    )
    top_names = [row["medicine"] for row in medicines[:3]]

    assert "Cetirizine" in top_names
    assert "Fluticasone Nasal Spray" in top_names
    assert "Acetaminophen" not in top_names


def test_low_confidence_inputs_do_not_return_medicine_candidates(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)

    medicines = rank_medicines(
        [{"disease": "Pelvic Inflammatory Disease", "confidence": 0.08, "profile_match": 0.39}],
        profile_conditions=[],
        symptoms=["fever"],
        db_path=db_path,
    )

    assert medicines == []


def test_gastro_support_candidates_are_relevant_and_deduplicated(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)

    medicines = rank_medicines(
        [
            {"disease": "Gastroenteritis", "confidence": 0.61, "profile_match": 0.88},
            {"disease": "Diarrheal Illness", "confidence": 0.46, "profile_match": 0.84},
        ],
        profile_conditions=[],
        symptoms=["diarrhea", "vomiting", "abdominal cramps", "nausea"],
        db_path=db_path,
    )
    names = [row["medicine"] for row in medicines]

    assert len(names) == len(set(names))
    assert names[:3] == ["Oral Rehydration Salts", "Ondansetron", "Loperamide"]
    assert "Omeprazole" not in names[:3]


def test_urgency_uses_severity_duration_and_red_flags():
    base_prediction = [{"disease": "Common Cold", "confidence": 0.72}]

    urgent = determine_urgency(base_prediction, [], ["shortness of breath"])
    severe = determine_urgency(base_prediction, [], [], {"overall": "Severe"})
    persistent = determine_urgency(base_prediction, [], [], {"overall": "Mild"}, "More than a week")

    assert urgent["level"] == "Urgent care needed"
    assert severe["level"] == "Doctor recommended"
    assert persistent["level"] == "Doctor recommended"


def test_severity_and_duration_expand_symptom_evidence():
    vocabulary = ["headache", "severe headache", "cough", "persistent cough", "chronic cough"]

    symptoms = augment_symptoms_for_context(
        ["headache", "cough"],
        {"overall": "Moderate", "headache": "Severe"},
        "More than a week",
        vocabulary,
    )

    assert "severe headache" in symptoms
    assert "persistent cough" in symptoms
    assert "chronic cough" in symptoms


def test_safety_inputs_downrank_allergy_conflicts(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)

    medicines = rank_medicines(
        [{"disease": "Pneumonia", "confidence": 0.9, "profile_match": 1.0, "model_confidence": 0.9}],
        profile_conditions=[],
        symptoms=["cough", "fever", "shortness of breath"],
        db_path=db_path,
        allergies="penicillin",
    )
    amoxicillin = next(row for row in medicines if row["medicine"] == "Amoxicillin-Clavulanate")

    assert medicines[0]["medicine"] == "Medical Referral"
    assert amoxicillin["score"] == 0.0
    assert amoxicillin["status"] == "Safety conflict - clinician review"
    assert "penicillin" in amoxicillin["warnings"].lower()


def test_care_guidance_adds_non_medicine_support():
    guidance = build_care_guidance(
        ["diarrhea", "vomiting", "nausea"],
        red_flags=[],
        medicines=[],
        symptom_duration="2-3 days",
    )

    assert any("fluids" in item.lower() for item in guidance)
    assert any("digestive" in item.lower() for item in guidance)


def test_red_flag_answers_feed_urgency():
    result = recommend(
        selected_symptoms=["fever"],
        free_text="",
        age_group="adult",
        profile_conditions=[],
        red_flag_answers=["shortness of breath"],
        persist_log=False,
    )

    assert "shortness of breath" in result["red_flags"]
    assert result["urgency"]["level"] == "Urgent care needed"


def test_body_area_and_smart_symptom_suggestions_use_vocabulary():
    vocabulary = ["cough", "wheezing", "shortness of breath", "fever", "rash"]

    area = body_area_suggestions("Chest and breathing", vocabulary)
    smart = smart_symptom_suggestions(["cough"], vocabulary)

    assert "shortness of breath" in area
    assert "wheezing" in smart
    assert "rash" not in smart


def test_filter_symptoms_for_body_area_limits_out_of_context_options():
    vocabulary = ["cough", "wheezing", "shortness of breath", "fever", "rash", "headache"]

    chest = filter_symptoms_for_body_area("Chest and breathing", vocabulary)
    all_symptoms = filter_symptoms_for_body_area("Not sure", vocabulary)

    assert "cough" in chest
    assert "wheezing" in chest
    assert "rash" not in chest
    assert "headache" not in chest
    assert all_symptoms == vocabulary


def test_current_medicine_and_allergy_screening_flags_risks():
    medicines = [
        {
            "medicine": "Ibuprofen",
            "category": "NSAID pain and inflammation relief",
            "safety_notes": "Avoid with bleeding risk.",
            "score": 0.4,
            "therapeutic_fit": 0.9,
            "disease": "Pain",
            "prescription_required": False,
            "status": "Candidate",
            "warnings": "",
            "rationale": "Pain relief.",
        },
        {
            "medicine": "Amoxicillin-Clavulanate",
            "category": "Penicillin antibiotic",
            "safety_notes": "Use only for clinician-confirmed bacterial indications.",
            "score": 0.3,
            "therapeutic_fit": 0.8,
            "disease": "Pneumonia",
            "prescription_required": True,
            "status": "Candidate",
            "warnings": "",
            "rationale": "Antibiotic option.",
        },
    ]

    alerts = screen_current_medicines("warfarin", medicines, "penicillin")

    assert any(alert["type"] == "Current medicine review" for alert in alerts)
    assert any(alert["type"] == "Allergy match" for alert in alerts)


def test_medicine_comparison_rows_format_top_candidates():
    medicines = [
        {
            "medicine": "Cetirizine",
            "disease": "Allergic Rhinitis",
            "score": 0.41,
            "therapeutic_fit": 0.98,
            "prescription_required": False,
            "status": "Candidate",
            "warnings": "",
            "rationale": "Symptom-specific support.",
        }
    ]

    rows = build_medicine_comparison_rows(medicines)

    assert rows[0]["Medicine"] == "Cetirizine"
    assert rows[0]["Symptom fit"] == "98%"
    assert rows[0]["Prescription"] == "Not listed"


def test_history_clear_helpers_remove_only_requested_rows(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)
    log_recommendation(
        symptoms=["fever"],
        age_group="adult",
        profile_conditions=[],
        top_disease="Influenza",
        model_confidence=0.6,
        db_path=db_path,
    )
    log_feedback(
        symptoms=["fever"],
        top_disease="Influenza",
        rating="Useful",
        irrelevant_medicine=False,
        comments="",
        medicine_name="Acetaminophen",
        db_path=db_path,
    )

    assert clear_recommendation_history(db_path) == 1
    assert int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_logs", db_path).iloc[0]["count"]) == 0
    assert int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_feedback", db_path).iloc[0]["count"]) == 1
    assert fetch_dataframe("SELECT medicine_name FROM recommendation_feedback", db_path).iloc[0]["medicine_name"] == "Acetaminophen"

    assert clear_feedback_history(db_path) == 1
    assert int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_feedback", db_path).iloc[0]["count"]) == 0


def test_clear_all_history_removes_recommendations_and_feedback(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)
    log_recommendation(
        symptoms=["cough"],
        age_group="adult",
        profile_conditions=[],
        top_disease="Common Cold",
        model_confidence=0.5,
        db_path=db_path,
    )
    log_feedback(
        symptoms=["cough"],
        top_disease="Common Cold",
        rating="Partly useful",
        irrelevant_medicine=True,
        comments="",
        db_path=db_path,
    )

    removed = clear_all_history(db_path)

    assert removed == {"recommendations": 1, "feedback": 1}
    assert int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_logs", db_path).iloc[0]["count"]) == 0
    assert int(fetch_dataframe("SELECT COUNT(*) AS count FROM recommendation_feedback", db_path).iloc[0]["count"]) == 0
