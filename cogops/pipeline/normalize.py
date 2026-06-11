"""
cogops/pipeline/normalize.py

Lightweight deterministic query normalizer.

Runs on every sub-query AFTER the router but BEFORE Jiggasha embedding.
Purpose:
  1. Map common informal / English loanwords to formal Bengali corpus terms.
  2. Strip conversational fillers.
  3. Normalize whitespace.

This is a safety net: the LLM router already reformulates most queries,
but edge cases (timeout fallback, unexpected LLM output) still benefit
from rule-based cleanup.
"""

import re
import unicodedata
from typing import List

# ---------------------------------------------------------------------------
# Synonym mappings: informal / loanword → formal Bengali (corpus-aligned)
# Each root is expanded with common Bengali suffixes at compile time.
# ---------------------------------------------------------------------------
_SYNONYM_ROOTS = [
    # Transport
    ("প্লেন", "বিমান"),
    ("ট্রেন", "রেল"),
    # Ticket / fee wording
    ("টিকেট", "টিকিট"),
]

# Common Bengali inflectional suffixes (possessive, locative, etc.)
_BN_SUFFIXES = ["ের", "ে", "ি", "া", "ো", "ী", "ু", "ূ", "ৃ", "ং", "ঃ", "়", "ঁ"]

# Compile a regex for every (root + suffix) combination.
# Boundary rule: not preceded / followed by another Bengali character,
# which gives us precise word-level matching inside Bengali text.
_BN_CHAR_CLASS = r"[\u0980-\u09FF]"

_SYNONYM_RES: List[tuple[re.Pattern[str], str]] = []
for informal, formal in _SYNONYM_ROOTS:
    for suffix in [""] + _BN_SUFFIXES:
        old_form = informal + suffix
        new_form = formal + suffix
        pattern = (
            r"(?<!" + _BN_CHAR_CLASS + ")"
            + re.escape(old_form)
            + r"(?!" + _BN_CHAR_CLASS + ")"
        )
        _SYNONYM_RES.append((re.compile(pattern), new_form))

# Filler words that should be removed when they appear as standalone tokens.
_FILLER_RE = re.compile(
    r"(?<![\u0980-\u09FF])"
    r"(আচ্ছা|ভাই|দেখেন|শুনুন|বলুন\s+তো|বলো\s+তো|জানাবেন|জানাব|একটু|কিন্তু|তাই|তাহলে|তো)"
    r"(?![\u0980-\u09FF])",
    re.IGNORECASE,
)

# Multiple punctuation / whitespace cleanup.
_WS_COLLAPSE_RE = re.compile(r"[ \t]+")
_NL_COLLAPSE_RE = re.compile(r"\n{3,}")


def normalize_sub_query(text: str) -> str:
    """Return a cleaned, formalized version of a Bengali sub-query.

    Steps:
      1. NFC Unicode normalization + strip.
      2. Synonym replacement (root + preserved suffixes).
      3. Filler-word removal.
      4. Whitespace collapse.
      5. Strip stray ASCII punctuation that may appear after cleanup.
    """
    if not text:
        return ""

    # 1. Basic normalization
    text = unicodedata.normalize("NFC", text).strip()

    # 2. Synonym replacement (root + preserved suffixes)
    for regex, replacement in _SYNONYM_RES:
        text = regex.sub(replacement, text)

    # 3. Filler removal (standalone tokens only)
    text = _FILLER_RE.sub("", text)

    # 4. Whitespace collapse
    text = _WS_COLLAPSE_RE.sub(" ", text)
    text = _NL_COLLAPSE_RE.sub("\n\n", text)
    text = text.strip()

    # 5. Remove stray leading/trailing ASCII punctuation.
    # Keep Bengali danda (।) and question marks (?) because they are semantic.
    text = text.strip(r".,!;:\-")
    text = text.strip()

    return text


def normalize_sub_queries(queries: List[str]) -> List[str]:
    """Normalize a list of sub-queries, dropping any that become empty."""
    out: List[str] = []
    for q in queries:
        cleaned = normalize_sub_query(q)
        if cleaned:
            out.append(cleaned)
    return out if out else queries
