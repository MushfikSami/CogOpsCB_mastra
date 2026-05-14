"""
cogops/utils/thinking_parser.py

Robust parser for extracting thinking content from LLM responses.

Handles both formats:
  - Angle-bracket tags: <thinking>...</thinking>
  - Markdown code blocks: ```thinking ... ```

Also handles:
  - Multiple thinking blocks in sequence
  - Unclosed tags (truncates thinking at end of stream)
  - Streaming-like incremental input (buffer + flush pattern)
  - Tags that span chunk boundaries (holds back up to tag-length bytes)

Usage:
    parser = ThinkingParser()
    for channel, content in parser.feed(text_chunk):
        yield (channel, content)
    for channel, content in parser.flush():
        yield (channel, content)
"""

import logging
import re
from typing import Iterator, Tuple, Optional

logger = logging.getLogger(__name__)

# Angle-bracket tags
_OPEN_TAG = "<thinking>"
_CLOSE_TAG = "</thinking>"
# Markdown code block markers
_OPEN_BLOCK = "```thinking"
_CLOSE_BLOCK = "```"

# All open/close markers sorted longest-first for unambiguous matching
_OPEN_MARKERS = sorted([_OPEN_TAG, _OPEN_BLOCK], key=len, reverse=True)
_CLOSE_MARKERS = sorted([_CLOSE_TAG, _CLOSE_BLOCK], key=len, reverse=True)

# Longest open marker length — used for chunk-boundary safety
_OPEN_MAX = max(len(m) for m in _OPEN_MARKERS)
_CLOSE_MAX = max(len(m) for m in _CLOSE_MARKERS)
_HOLDBACK = _OPEN_MAX + 4  # enough to protect any open-tag prefix across chunks


class ThinkingParser:
    """Parser for <thinking> tags and ```thinking code blocks."""

    def __init__(self):
        self._buffer = ""
        self._in_thinking = False
        self._open_style: Optional[str] = None  # "tag" or "block"

    def _channel(self) -> str:
        return "thinking" if self._in_thinking else "answer"

    def _find_open_marker(self, buf: str) -> Optional[Tuple[str, str, int]]:
        """Find the first open marker in buf.

        Returns (style, matched_tag, index) or None.
        """
        best = None
        for tag in _OPEN_MARKERS:
            idx = buf.find(tag)
            if idx >= 0 and (best is None or idx < best[2]):
                best = ("tag" if tag == _OPEN_TAG else "block", tag, idx)
        return best

    def _find_close_marker(self, buf: str) -> Optional[Tuple[str, str, int]]:
        """Find the close marker matching how we opened."""
        close_tag = "</thinking>" if self._open_style == "tag" else "```"
        idx = buf.find(close_tag)
        if idx >= 0:
            return ("tag" if self._open_style == "tag" else "block", close_tag, idx)
        return None

    def _emit_safe_tail(self) -> Iterator[Tuple[str, str]]:
        """Emit content that cannot possibly contain a tag boundary.

        Strategy: hold back _HOLDBACK trailing bytes. If the buffer is
        shorter than _HOLDBACK, hold back everything (a tag prefix could
        span a chunk boundary). Only emit when the buffer is long enough
        to safely strip the trailing margin.
        """
        if not self._buffer:
            return

        if len(self._buffer) <= _HOLDBACK:
            # Entire buffer could be a partial tag — hold back everything
            return

        # Emit everything except trailing _HOLDBACK bytes
        safe = self._buffer[:-_HOLDBACK]
        if safe:
            yield (self._channel(), safe)
        self._buffer = self._buffer[-_HOLDBACK:]

    def feed(self, text: str) -> Iterator[Tuple[str, str]]:
        """Feed text chunk and yield (channel, content) pairs."""
        if not text:
            return

        self._buffer += text

        while True:
            if self._in_thinking:
                result = self._find_close_marker(self._buffer)
                if result is None:
                    yield from self._emit_safe_tail()
                    return
                _, close_tag, idx = result
                before = self._buffer[:idx]
                if before:
                    yield (self._channel(), before)
                self._buffer = self._buffer[idx + len(close_tag):]
                self._in_thinking = False
            else:
                result = self._find_open_marker(self._buffer)
                if result is None:
                    yield from self._emit_safe_tail()
                    return
                style, open_tag, idx = result
                before = self._buffer[:idx]
                if before:
                    yield (self._channel(), before)
                self._buffer = self._buffer[idx + len(open_tag):]
                self._open_style = style
                self._in_thinking = True

    def flush(self) -> Iterator[Tuple[str, str]]:
        """Emit remaining buffer content."""
        if not self._buffer:
            return

        if self._in_thinking:
            logger.debug(
                "Stream ended with unclosed thinking; "
                "flushing %d chars as thinking.", len(self._buffer),
            )
            yield ("thinking", self._buffer)
        else:
            yield ("answer", self._buffer)

        self._buffer = ""
        self._open_style = None
