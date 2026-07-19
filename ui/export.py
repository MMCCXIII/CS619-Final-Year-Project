import json
from datetime import datetime, timezone

from ui.formatting import display_text


def build_session_report(result: dict) -> dict:
    top = (result.get("predicted_diseases") or [{}])[0]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symptoms": [display_text(item) for item in result.get("symptoms", [])],
        "body_area": result.get("body_area"),
        "symptom_duration": result.get("symptom_duration"),
        "urgency": result.get("urgency"),
        "red_flags": [display_text(item) for item in result.get("red_flags", [])],
        "top_condition": top.get("disease"),
        "top_confidence": top.get("confidence"),
        "predicted_diseases": result.get("predicted_diseases", []),
        "medicines": result.get("medicines", []),
        "care_guidance": result.get("care_guidance", []),
        "interaction_alerts": result.get("interaction_alerts", []),
        "message": result.get("message"),
    }


def session_report_json(result: dict) -> str:
    return json.dumps(build_session_report(result), indent=2)
