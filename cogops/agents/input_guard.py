"""
cogops/agents/input_guard.py

Layer 0 — InputGuard.

Pure-code input validation: zero LLM latency. Rejects obviously malformed
input before any agent is touched.

All thresholds are configurable via config.yml under `input_guard:`.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GuardConfig:
    max_chars: int = 4096
    max_tokens: int = 1024
    max_repeated_char_run: int = 100
    control_char_threshold: float = 0.10
    entropy_threshold: float = 1.5
    entropy_min_length: int = 50
    token_bomb_min_words: int = 12
    token_bomb_max_avg_len: float = 2.0


# Reasons returned (used by debug telemetry; never shown to the user).
REASON_EMPTY = "empty"
REASON_TOO_LONG = "too_long"
REASON_BINARY_OR_CONTROL = "binary_or_control"
REASON_INJECTION = "injection_attempt"
REASON_SPAM = "spam"
REASON_LOW_ENTROPY = "low_entropy"
REASON_TOKEN_BOMB = "token_bomb"

# Obvious prompt-injection markers. Narrow on purpose — the real defense is
# XML data-framing in the composer prompt.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (?:all |previous |above |prior )?(?:instructions|prompts|rules|directives)"),
    re.compile(r"(?i)disregard (?:all |previous |the )?(?:instructions|prompts|rules|system)"),
    re.compile(r"(?i)(?:^|\s)system\s*:\s*"),
    re.compile(r"</?(?:context|system|user|assistant|im_start|im_end)\s*>", re.IGNORECASE),
    re.compile(r"\{\{[^}]*system[^}]*\}\}"),
    re.compile(r"<\|[^|]{0,40}\|>"),
    re.compile(r"(?i)ager (?:shob |sob )?kotha (?:vule|bhule) jao"),
    re.compile(r"(?i)jailbreak|DAN mode|you are now|new instructions"),
]

_REPETITION_RE = re.compile(r"(.)\1{99,}", re.DOTALL)


class InputGuard:
    """Layer 0 — reject malformed input before any LLM call."""

    def __init__(self, config: Optional[GuardConfig] = None):
        self.cfg = config or GuardConfig()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _control_char_fraction(text: str) -> float:
        if not text:
            return 0.0
        bad = 0
        for ch in text:
            if ch in ("\n", "\t", "\r"):
                continue
            if unicodedata.category(ch) == "Cc":
                bad += 1
        return bad / len(text)

    @staticmethod
    def _shannon_entropy(text: str) -> float:
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

    def _is_token_bomb(self, text: str) -> bool:
        words = [w for w in text.split() if w.strip()]
        if len(words) < self.cfg.token_bomb_min_words:
            return False
        avg_len = sum(len(w) for w in words) / len(words)
        return avg_len < self.cfg.token_bomb_max_avg_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check(self, query: str) -> Tuple[str, Optional[str]]:
        """Validate and normalize a user query.

        Returns:
            (clean_query, None)         on success
            ("", reason_code)           on rejection
        """
        if query is None:
            return "", REASON_EMPTY
        if not isinstance(query, str):
            return "", REASON_BINARY_OR_CONTROL

        # 1. Empty / whitespace-only
        if not query.strip():
            return "", REASON_EMPTY

        # 2. Length cap (chars, not bytes)
        if len(query) > self.cfg.max_chars:
            return "", REASON_TOO_LONG

        # 3. NUL bytes / binary
        if "\x00" in query:
            return "", REASON_BINARY_OR_CONTROL
        if self._control_char_fraction(query) > self.cfg.control_char_threshold:
            return "", REASON_BINARY_OR_CONTROL

        # 4. Injection markers
        for pat in _INJECTION_PATTERNS:
            if pat.search(query):
                return "", REASON_INJECTION

        # 5. Repetition spam
        if _REPETITION_RE.search(query):
            return "", REASON_SPAM

        # 6. Entropy check (long inputs only)
        if len(query) >= self.cfg.entropy_min_length:
            if self._shannon_entropy(query) < self.cfg.entropy_threshold:
                return "", REASON_LOW_ENTROPY

        # 7. Token bomb
        if self._is_token_bomb(query):
            return "", REASON_TOKEN_BOMB

        # All checks passed — NFC-normalize, trim, collapse whitespace.
        clean = unicodedata.normalize("NFC", query).strip()
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        return clean, None
