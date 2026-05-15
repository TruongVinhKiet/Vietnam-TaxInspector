"""Vietnamese text normalization helpers for Tax Agent routing.

The agent receives many short, informal utterances from the UI.  These helpers
keep deterministic routing robust without adding heavyweight fuzzy-search
dependencies to the backend.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_vietnamese_text(value: str) -> str:
    """Lowercase, strip Vietnamese accents, normalize whitespace and punctuation."""
    try:
        text = unicodedata.normalize("NFD", value or "")
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "D")
    except Exception:
        text = value or ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def compact_vietnamese_text(value: str) -> str:
    """Return normalized text without spaces, useful for short typo matching."""
    return normalize_vietnamese_text(value).replace(" ", "")


def bounded_levenshtein(left: str, right: str, max_distance: int = 2) -> int:
    """Compute Levenshtein distance, stopping once it is certainly too large."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, 1):
        current = [i]
        row_min = current[0]
        for j, rc in enumerate(right, 1):
            cost = 0 if lc == rc else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
            row_min = min(row_min, current[-1])
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def fuzzy_phrase_match(value: str, phrases: list[str] | tuple[str, ...], *, max_distance: int = 2) -> bool:
    """Return True if a short normalized utterance approximately matches a phrase."""
    text = normalize_vietnamese_text(value)
    compact = text.replace(" ", "")
    if not compact:
        return False
    for phrase in phrases:
        norm = normalize_vietnamese_text(phrase)
        norm_compact = norm.replace(" ", "")
        if not norm_compact:
            continue
        if norm in text or norm_compact in compact:
            return True
        if bounded_levenshtein(compact, norm_compact, max_distance=max_distance) <= max_distance:
            return True
    return False


GREETING_PHRASES = (
    "xin chao",
    "chao",
    "chao ban",
    "xin chao ban",
    "alo",
    "alo ban oi",
    "hello",
    "hi",
    "hey",
)

THANKS_PHRASES = (
    "cam on",
    "cam on ban",
    "thank you",
    "thanks",
    "ok cam on",
)


def is_probable_greeting(value: str) -> bool:
    """Tolerate missing accents and small typos in short greeting utterances."""
    text = normalize_vietnamese_text(value)
    if len(text.split()) > 5:
        return False
    return fuzzy_phrase_match(text, GREETING_PHRASES, max_distance=2)


def is_probable_thanks(value: str) -> bool:
    """Tolerate missing accents and small typos in short thanks utterances."""
    text = normalize_vietnamese_text(value)
    if len(text.split()) > 6:
        return False
    return fuzzy_phrase_match(text, THANKS_PHRASES, max_distance=2)
