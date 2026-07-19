from ui.constants import ACRONYMS


def display_text(value) -> str:
    text = str(value).replace("_", " ").strip()
    lower = text.lower()
    if lower in ACRONYMS:
        return ACRONYMS[lower]
    words = []
    for word in text.split():
        token = word.lower()
        words.append(ACRONYMS.get(token, word[:1].upper() + word[1:]))
    return " ".join(words)


def format_symptom_list(value: str) -> str:
    return ", ".join(display_text(item) for item in str(value or "").split("|") if item) or "None"
