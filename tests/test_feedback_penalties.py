from pathlib import Path

from src.data_access import fetch_feedback_penalties, log_feedback, seed_database
from src.recommender import rank_medicines


def test_feedback_penalties_reduce_medicine_rank(tmp_path: Path):
    db_path = tmp_path / "medicine.sqlite"
    seed_database(db_path)

    predicted = [
        {
            "disease": "Common Cold",
            "confidence": 0.8,
            "model_confidence": 0.8,
            "profile_match": 1.0,
            "matched_symptoms": ["cough"],
        }
    ]
    baseline = rank_medicines(predicted, [], ["cough"], db_path)
    assert baseline

    target = baseline[0]["medicine"]
    log_feedback(
        symptoms=["cough"],
        top_disease="Common Cold",
        rating="Medicine seemed irrelevant",
        irrelevant_medicine=True,
        comments="",
        medicine_name=target,
        db_path=db_path,
    )
    log_feedback(
        symptoms=["cough"],
        top_disease="Common Cold",
        rating="Medicine seemed irrelevant",
        irrelevant_medicine=True,
        comments="",
        medicine_name=target,
        db_path=db_path,
    )

    penalties = fetch_feedback_penalties(db_path)
    assert penalties
    penalized = rank_medicines(predicted, [], ["cough"], db_path)
    baseline_score = next(row["score"] for row in baseline if row["medicine"] == target)
    penalized_score = next(row["score"] for row in penalized if row["medicine"] == target)
    assert penalized_score < baseline_score
