"""
cogops/pipeline/sanitize.py

Stage 0 of the deterministic pipeline. Pure-code input validation — no LLM call.
Runs in <5ms on the longest valid input. Catches:

  - empty / whitespace-only
  - length over MAX_QUERY_CHARS
  - NUL bytes / excessive control chars / binary blobs
  - obvious prompt-injection markers (regex fast-path; NOT a shield —
    the load-bearing defense is XML data-framing in the composer prompt)
  - single-character repetition spam

Returns (clean_query, refusal_reason). When refusal_reason is non-None, the
pipeline emits the static input-invalid refusal and stops before any LLM call.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Tuple

MAX_QUERY_CHARS = 4096
MAX_REPETITION_RUN = 100
MAX_CONTROL_FRACTION = 0.10

# Entropy: legitimate Bengali text is usually > 2.0 bits/char.
# Below this threshold suggests repetitive / low-information input.
MIN_ENTROPY_BITS = 1.5

# Token bomb: average token length below this with enough tokens suggests
# synthetic low-density input (e.g. "a a a a …" or "অ অ অ অ …").
TOKEN_BOMB_MIN_WORDS = 12
TOKEN_BOMB_MAX_AVG_LEN = 2.0

# Refusal text shown to the user when sanitize fails. Single line, Bengali.
INPUT_INVALID_REFUSAL_BN = (
    "দুঃখিত, প্রশ্নটি বোঝা গেল না বা সীমার বাইরে। "
    "অনুগ্রহ করে স্পষ্ট, সংক্ষিপ্ত প্রশ্ন করুন।"
)

# Reasons returned (used by debug telemetry; never shown to the user).
REASON_EMPTY = "empty"
REASON_TOO_LONG = "too_long"
REASON_BINARY_OR_CONTROL = "binary_or_control"
REASON_INJECTION = "injection_attempt"
REASON_SPAM = "spam"
REASON_LOW_ENTROPY = "low_entropy"
REASON_TOKEN_BOMB = "token_bomb"

# Obvious prompt-injection markers. This is intentionally narrow — its job is
# to short-circuit obvious adversarial inputs before they hit the LLM. The real
# defense lives in the composer system prompt's <context>/<user_query> framing.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (?:all |previous |above |prior )?(?:instructions|prompts|rules|directives)"),
    re.compile(r"(?i)disregard (?:all |previous |the )?(?:instructions|prompts|rules|system)"),
    re.compile(r"(?i)(?:^|\s)system\s*:\s*"),
    re.compile(r"</?(?:context|system|user|assistant|im_start|im_end)\s*>", re.IGNORECASE),
    re.compile(r"\{\{[^}]*system[^}]*\}\}"),
    re.compile(r"<\|[^|]{0,40}\|>"),
    re.compile(r"(?i)ager (?:shob |sob )?kotha (?:vule|bhule) jao"),
]

# Detect runs of the same character (excluding whitespace) longer than the cap.
# Used to catch "aaaaaaaaaaaa…" or single-emoji spam without flagging legit
# Bengali sentences which never repeat one codepoint 100+ times in a row.
_REPETITION_RE = re.compile(r"(.)\1{" + str(MAX_REPETITION_RUN) + r",}", re.DOTALL)


def _control_char_fraction(text: str) -> float:
    """Fraction of characters that are Unicode category Cc, excluding \\n, \\t, \\r."""
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        if ch in ("\n", "\t", "\r"):
            continue
        if unicodedata.category(ch) == "Cc":
            bad += 1
    return bad / len(text)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy in bits per character."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    length = len(text)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _is_token_bomb(text: str) -> bool:
    """Detect low-information-density input: many very short tokens."""
    words = [w for w in text.split() if w.strip()]
    if len(words) < TOKEN_BOMB_MIN_WORDS:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    return avg_len < TOKEN_BOMB_MAX_AVG_LEN


def sanitize(query: str) -> Tuple[str, str | None]:
    """Validate and normalize a user query.

    Returns:
        (clean_query, None)         on success
        ("", reason_code)           on rejection — caller emits INPUT_INVALID_REFUSAL_BN

    Order matters: cheapest checks first. NFC-normalize + trim happens after
    all checks pass (we don't want sanitization itself to mask injection).
    """
    if query is None:
        return "", REASON_EMPTY
    if not isinstance(query, str):
        # Defensive — caller should always pass str; if not, treat as invalid.
        return "", REASON_BINARY_OR_CONTROL

    # 1. Empty / whitespace-only
    if not query.strip():
        return "", REASON_EMPTY

    # 2. Length cap (chars, not bytes — Bengali codepoints count as 1)
    if len(query) > MAX_QUERY_CHARS:
        return "", REASON_TOO_LONG

    # 3. NUL bytes / binary
    if "\x00" in query:
        return "", REASON_BINARY_OR_CONTROL
    if _control_char_fraction(query) > MAX_CONTROL_FRACTION:
        return "", REASON_BINARY_OR_CONTROL

    # 4. Injection markers — check the RAW string so attacker can't bypass by
    # padding whitespace that would be collapsed later.
    for pat in _INJECTION_PATTERNS:
        if pat.search(query):
            return "", REASON_INJECTION

    # 5. Repetition spam
    if _REPETITION_RE.search(query):
        return "", REASON_SPAM

    # 6. Entropy check (catches repetitive / low-diversity input).
    # Only applies to longer inputs; short queries naturally have lower entropy.
    if len(query) >= 50 and _shannon_entropy(query) < MIN_ENTROPY_BITS:
        return "", REASON_LOW_ENTROPY

    # 7. Token bomb (catches synthetic low-density input)
    if _is_token_bomb(query):
        return "", REASON_TOKEN_BOMB

    # All checks passed — NFC-normalize, trim, collapse internal whitespace.
    clean = unicodedata.normalize("NFC", query).strip()
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    return clean, None
