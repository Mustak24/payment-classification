# =============================================================================
# preprocess.py — Text cleaning & normalization
# =============================================================================
"""
All cleaning happens here so the same logic is used during both
training and inference — preventing training/serving skew.
"""

import re
import html
import unicodedata
from config import MAX_TEXT_LENGTH


# ── Regex patterns (compiled once for speed) ─────────────────────────────────
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE   = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE   = re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b")          # Indian mobile
_ACCT_RE    = re.compile(r"\bX+\d{2,4}\b", re.IGNORECASE)           # XXXX1234
_AMOUNT_RE  = re.compile(
    r"(?:rs\.?|inr|₹|\$)\s*[\d,]+(?:\.\d{1,2})?|"                  # ₹1,000.00
    r"[\d,]+(?:\.\d{1,2})?\s*(?:rs\.?|inr|₹)",
    re.IGNORECASE,
)
_DATE_RE    = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"                          # 12/04/2024
    r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|"
    r"jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",
    re.IGNORECASE,
)
_REF_RE     = re.compile(r"\b(?:ref|txn|utr|rrn|imps|neft|rtgs)[\s#:]*\w+\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """
    Clean and normalize a single SMS / message string.

    Steps:
    1. Decode HTML entities  (&amp; → &)
    2. Unicode normalize     (NFC form)
    3. Lowercase
    4. Replace sensitive / high-variance tokens with placeholders
       so the model learns *patterns*, not specific values
    5. Strip leftover punctuation noise
    6. Collapse whitespace
    7. Truncate to MAX_TEXT_LENGTH

    Returns a cleaned string (never empty — falls back to "<empty>").
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. HTML entities
    text = html.unescape(text)

    # 2. Unicode normalise
    text = unicodedata.normalize("NFC", text)

    # 3. Lowercase
    text = text.lower()

    # 4. Token replacement  (order matters)
    text = _URL_RE.sub(" <URL> ", text)
    text = _EMAIL_RE.sub(" <EMAIL> ", text)
    text = _PHONE_RE.sub(" <PHONE> ", text)
    text = _ACCT_RE.sub(" <ACCOUNT> ", text)
    text = _AMOUNT_RE.sub(" <AMOUNT> ", text)
    text = _DATE_RE.sub(" <DATE> ", text)
    text = _REF_RE.sub(" <REFNUM> ", text)

    # 5. Remove stray punctuation but keep angle brackets for placeholders
    text = re.sub(r"[^\w\s<>]", " ", text)

    # 6. Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()

    # 6.5 Remove newlines (FastText uses newlines as record separators)
    text = text.replace("\n", " ").replace("\r", " ")

    # 7. Truncate
    text = text[:MAX_TEXT_LENGTH]

    return text if text else "<empty>"


def clean_batch(texts: list[str]) -> list[str]:
    """Clean a list of texts — convenience wrapper."""
    return [clean(t) for t in texts]
