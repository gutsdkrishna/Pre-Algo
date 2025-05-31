import re, html
import unicodedata


def clean_text(text: str) -> str:
    # Unescape HTML entities
    text = html.unescape(text)
    # Normalize accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove non-word characters (emojis, symbols)
    text = re.sub(r"[^\w\s,\.\!\?\'-]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
