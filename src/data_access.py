import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATABASE_PATH, DATASET_PATH, SCHEMA_PATH
from src.knowledge_base import (
    build_contraindications,
    build_disease_medicine_map,
    build_diseases,
    build_medicines,
)
from src.preprocessing import parse_pipe_list


DISEASES = build_diseases()
MEDICINES = build_medicines()
DISEASE_MEDICINE_MAP = build_disease_medicine_map()
CONTRAINDICATIONS = build_contraindications()


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(db_path) as connection:
        connection.executescript(schema)


def ensure_runtime_tables(db_path: Path = DATABASE_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              symptoms TEXT NOT NULL,
              top_disease TEXT NOT NULL,
              medicine_name TEXT NOT NULL DEFAULT '',
              rating TEXT NOT NULL,
              irrelevant_medicine INTEGER NOT NULL DEFAULT 0,
              comments TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(recommendation_feedback)").fetchall()
        }
        if "medicine_name" not in columns:
            connection.execute("ALTER TABLE recommendation_feedback ADD COLUMN medicine_name TEXT NOT NULL DEFAULT ''")


def seed_database(db_path: Path = DATABASE_PATH, dataset_path: Path = DATASET_PATH) -> None:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM recommendation_feedback")
        connection.execute("DELETE FROM recommendation_logs")
        connection.execute("DELETE FROM medicine_contraindications")
        connection.execute("DELETE FROM disease_medicines")
        connection.execute("DELETE FROM medicines")
        connection.execute("DELETE FROM diseases")
        connection.execute("DELETE FROM training_cases")

        for name, payload in DISEASES.items():
            connection.execute(
                """
                INSERT INTO diseases (name, description, default_severity, emergency_flags)
                VALUES (?, ?, ?, ?)
                """,
                (name, payload["description"], payload["severity"], json.dumps(payload["flags"])),
            )

        for name, payload in MEDICINES.items():
            connection.execute(
                """
                INSERT INTO medicines (name, category, prescription_required, effectiveness_score, safety_notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, payload["category"], payload["rx"], payload["score"], payload["notes"]),
            )

        disease_ids = {
            row["name"]: row["id"] for row in connection.execute("SELECT id, name FROM diseases").fetchall()
        }
        medicine_ids = {
            row["name"]: row["id"] for row in connection.execute("SELECT id, name FROM medicines").fetchall()
        }

        for disease_name, medicines in DISEASE_MEDICINE_MAP.items():
            for medicine_name, relevance, rationale in medicines:
                connection.execute(
                    """
                    INSERT INTO disease_medicines (disease_id, medicine_id, relevance_score, rationale)
                    VALUES (?, ?, ?, ?)
                    """,
                    (disease_ids[disease_name], medicine_ids[medicine_name], relevance, rationale),
                )

        for medicine_name, items in CONTRAINDICATIONS.items():
            for condition, severity, reason in items:
                connection.execute(
                    """
                    INSERT INTO medicine_contraindications (medicine_id, condition_key, severity, reason)
                    VALUES (?, ?, ?, ?)
                    """,
                    (medicine_ids[medicine_name], condition, severity, reason),
                )

        cases = pd.read_csv(dataset_path)
        for row in cases.itertuples(index=False):
            connection.execute(
                """
                INSERT INTO training_cases (id, disease, symptoms, age_group, severity, source_note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row.case_id),
                    str(row.disease),
                    "|".join(parse_pipe_list(str(row.symptoms))),
                    str(row.age_group).strip().lower(),
                    str(row.severity),
                    str(row.source_note),
                ),
            )


def fetch_dataframe(query: str, db_path: Path = DATABASE_PATH, params: tuple = ()) -> pd.DataFrame:
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def fetch_recommendation_catalog(disease_name: str, db_path: Path = DATABASE_PATH) -> list[dict]:
    query = """
        SELECT
          d.name AS disease,
          m.name AS medicine,
          m.category,
          m.prescription_required,
          m.effectiveness_score,
          m.safety_notes,
          dm.relevance_score,
          dm.rationale
        FROM diseases d
        JOIN disease_medicines dm ON dm.disease_id = d.id
        JOIN medicines m ON m.id = dm.medicine_id
        WHERE d.name = ?
        ORDER BY dm.relevance_score DESC, m.effectiveness_score DESC
    """
    return fetch_dataframe(query, db_path, (disease_name,)).to_dict(orient="records")


def fetch_medicine(medicine_name: str, db_path: Path = DATABASE_PATH) -> dict | None:
    query = """
        SELECT
          name AS medicine,
          category,
          prescription_required,
          effectiveness_score,
          safety_notes
        FROM medicines
        WHERE name = ?
    """
    rows = fetch_dataframe(query, db_path, (medicine_name,)).to_dict(orient="records")
    return rows[0] if rows else None


def fetch_contraindications(medicine_name: str, db_path: Path = DATABASE_PATH) -> list[dict]:
    query = """
        SELECT mc.condition_key, mc.severity, mc.reason
        FROM medicine_contraindications mc
        JOIN medicines m ON m.id = mc.medicine_id
        WHERE m.name = ?
    """
    return fetch_dataframe(query, db_path, (medicine_name,)).to_dict(orient="records")


def log_recommendation(
    symptoms: list[str],
    age_group: str,
    profile_conditions: list[str],
    top_disease: str,
    model_confidence: float,
    db_path: Path = DATABASE_PATH,
) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO recommendation_logs
              (symptoms, age_group, profile_conditions, top_disease, model_confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "|".join(symptoms),
                age_group,
                "|".join(profile_conditions),
                top_disease,
                float(model_confidence),
            ),
        )


def log_feedback(
    symptoms: list[str],
    top_disease: str,
    rating: str,
    irrelevant_medicine: bool,
    comments: str,
    medicine_name: str = "",
    db_path: Path = DATABASE_PATH,
) -> None:
    ensure_runtime_tables(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO recommendation_feedback
              (symptoms, top_disease, medicine_name, rating, irrelevant_medicine, comments)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "|".join(symptoms),
                top_disease,
                str(medicine_name or "").strip(),
                str(rating),
                int(bool(irrelevant_medicine)),
                str(comments or "").strip(),
            ),
        )


def clear_recommendation_history(db_path: Path = DATABASE_PATH) -> int:
    with get_connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM recommendation_logs")
        return int(cursor.rowcount)


def clear_feedback_history(db_path: Path = DATABASE_PATH) -> int:
    ensure_runtime_tables(db_path)
    with get_connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM recommendation_feedback")
        return int(cursor.rowcount)


def clear_all_history(db_path: Path = DATABASE_PATH) -> dict[str, int]:
    return {
        "recommendations": clear_recommendation_history(db_path),
        "feedback": clear_feedback_history(db_path),
    }


def fetch_feedback_penalties(db_path: Path = DATABASE_PATH) -> dict[str, float]:
    from src.config import FEEDBACK_PENALTY_MAX, FEEDBACK_PENALTY_PER_FLAG
    from src.preprocessing import normalize_symptom

    ensure_runtime_tables(db_path)
    feedback = fetch_dataframe(
        """
        SELECT medicine_name, SUM(irrelevant_medicine) AS irrelevant_flags
        FROM recommendation_feedback
        WHERE TRIM(medicine_name) != ''
        GROUP BY medicine_name
        HAVING SUM(irrelevant_medicine) > 0
        """,
        db_path,
    )
    penalties: dict[str, float] = {}
    for row in feedback.itertuples(index=False):
        medicine_key = normalize_symptom(row.medicine_name)
        if not medicine_key:
            continue
        flag_count = int(row.irrelevant_flags)
        penalties[medicine_key] = min(FEEDBACK_PENALTY_MAX, FEEDBACK_PENALTY_PER_FLAG * flag_count)
    return penalties
