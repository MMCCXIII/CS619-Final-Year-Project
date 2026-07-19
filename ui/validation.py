from src.preprocessing import extract_symptoms_from_text


def can_generate_recommendation(
    selected_symptoms: list[str],
    free_text: str,
    vocabulary: list[str],
) -> bool:
    if selected_symptoms:
        return True
    return bool(extract_symptoms_from_text(free_text or "", vocabulary))
