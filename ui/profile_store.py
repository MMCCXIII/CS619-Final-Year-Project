import json

from src.config import PROFILE_STORE_PATH

from ui.constants import DEFAULT_PROFILE


def load_persisted_profile() -> dict | None:
    if not PROFILE_STORE_PATH.exists():
        return None
    try:
        payload = json.loads(PROFILE_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "age_group": str(payload.get("age_group", DEFAULT_PROFILE["age_group"])),
        "conditions": list(payload.get("conditions") or []),
        "current_medicines": str(payload.get("current_medicines", "")),
        "allergies": str(payload.get("allergies", "")),
    }


def save_persisted_profile(profile: dict) -> None:
    PROFILE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_STORE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def clear_persisted_profile() -> None:
    if PROFILE_STORE_PATH.exists():
        PROFILE_STORE_PATH.unlink()
