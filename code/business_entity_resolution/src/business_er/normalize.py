"""Conservative text views; raw records remain available separately."""

import unicodedata


def normalize_text(text: str) -> str:
    """NFKC/casefold text, preserving Unicode alphanumerics and combining marks."""
    text = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(
        char if char.isalnum() or unicodedata.category(char).startswith("M") else " "
        for char in text
    )
    return " ".join(cleaned.split())


def fold_latin_accents(text: str) -> str:
    """Make an auxiliary normalized view without deleting non-Latin vowel marks.

    Strip decomposed marks only after a Latin base character. This is accent
    folding, not transliteration; retain the original normalized view as well.
    """
    result = []
    latin_base = False
    for char in unicodedata.normalize("NFD", normalize_text(text)):
        if unicodedata.category(char).startswith("M"):
            if not latin_base:
                result.append(char)
        else:
            latin_base = "LATIN" in unicodedata.name(char, "")
            result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def numeric_tokens(text: str) -> frozenset[str]:
    """Extract digit-bearing tokens as hints, without asserting address semantics."""
    return frozenset(
        token for token in normalize_text(text).split()
        if any(char.isdigit() for char in token)
    )


def token_jaccard(left: str, right: str) -> float:
    """Token overlap feature; absent text is zero evidence, even empty vs empty."""
    a, b = set(normalize_text(left).split()), set(normalize_text(right).split())
    return len(a & b) / len(a | b) if a and b else 0.0
