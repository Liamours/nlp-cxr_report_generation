"""
Radiology report text normalization.

Fixes found in CheXpert+ and MIMIC-CXR impression/findings text:
  - Embedded section headers left in the text  (0.2% of rows)
  - De-identification underscores ___           (11.9%)
  - Newlines inside the text                    (49.5%)
  - Very short unusable strings                 (1.1%)

Deliberately NOT changed:
  - Casing  — BERT-uncased lowercases at tokenization time
  - Numbered lists (1. 2. 3.) — valid medical style, keep
  - Punctuation / abbreviations — medical terms, keep
"""

import re

# Section headers sometimes left inside the text field
_HEADER_RE = re.compile(
    r"^\s*(IMPRESSION|FINDINGS|CONCLUSION|SUMMARY)\s*:?\s*",
    re.IGNORECASE,
)

# De-identification placeholder: one or more underscores (any length)
_DEID_RE = re.compile(r"_+")

# Collapse any run of whitespace (including \n \r \t) to a single space
_WHITESPACE_RE = re.compile(r"\s+")

MIN_LEN = 15  # impressions shorter than this are uninformative


def normalize_report_text(text) -> str | None:
    """
    Clean a single impression or findings string.
    Returns None if the text is missing or too short after cleaning.
    """
    if text is None or (isinstance(text, float)):
        return None

    text = str(text)

    # Strip embedded section header
    text = _HEADER_RE.sub("", text)

    # Remove de-identification underscores
    text = _DEID_RE.sub(" ", text)

    # Normalize all whitespace to single spaces
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Drop if still too short
    return text if len(text) >= MIN_LEN else None
