"""
cogops/verifier/citations.py

Citation extraction + Sources block builder.

  - extract_citation_tags(answer) -> ordered list of S# numbers in the answer
  - extract_citations(answer)     -> [(tag, sentence)] pairs (sentence = the sentence
                                     containing the [S#] occurrence; Bengali-aware split)
  - strip_unknown_tags(answer, source_map) -> answer with hallucinated [S#] removed
  - build_sources_block(source_map, used_tags) -> Bengali Sources section, listing
                                                  ONLY the S# tags actually cited.

The system prompt instructs the model NOT to write its own Sources list; this
module is the single authority for the Sources block.
"""

import re
from typing import Dict, List, Tuple

CITE_RE = re.compile(r"\[S(\d+)\]")

# Bengali full stop (danda) + standard latin terminators.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")


def extract_citation_tags(answer: str) -> List[str]:
    """Return tags in document order, preserving duplicates.

    e.g. "x [S1]. y [S2][S1]." -> ["S1", "S2", "S1"]
    """
    return [f"S{m.group(1)}" for m in CITE_RE.finditer(answer or "")]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p for p in parts if p.strip()]


def extract_citations(answer: str) -> List[Tuple[str, str]]:
    """Return (tag, sentence) pairs in document order.

    A "sentence" is the substring around a [S#] occurrence, split on Bengali
    danda `।` and standard punctuation. If a sentence contains multiple [S#]
    tags, one pair is emitted per tag (same sentence text, different tag).
    Tags that fall outside any sentence (very edge case) attach to the empty string.
    """
    if not answer:
        return []
    pairs: List[Tuple[str, str]] = []
    for sentence in _split_sentences(answer):
        for m in CITE_RE.finditer(sentence):
            pairs.append((f"S{m.group(1)}", sentence.strip()))
    return pairs


def strip_unknown_tags(answer: str, source_map: Dict[str, dict]) -> Tuple[str, List[str]]:
    """Remove [S#] tags from answer that don't exist in source_map.

    Returns (cleaned_answer, list_of_dropped_tags). The cleaned answer has the
    bogus tag literal removed (e.g. "fact [S99]." -> "fact .") — downstream
    cosmetics (extra spaces before punctuation) are left as-is since they're
    rare and obvious.
    """
    if not answer:
        return "", []

    dropped: List[str] = []

    def repl(m: re.Match) -> str:
        tag = f"S{m.group(1)}"
        if tag not in source_map:
            dropped.append(tag)
            return ""
        return m.group(0)

    cleaned = CITE_RE.sub(repl, answer)
    return cleaned, dropped


def build_sources_block(
    source_map: Dict[str, dict],
    used_tags: List[str],
) -> str:
    """Render the Bengali Sources block listing only S# tags actually used.

    Order: tags appear in the block in the order they were FIRST cited.
    Duplicates are de-duped. Tags not in source_map are ignored (already
    stripped by strip_unknown_tags upstream, but we re-check for safety).
    """
    seen: List[str] = []
    for tag in used_tags:
        if tag in source_map and tag not in seen:
            seen.append(tag)

    if not seen:
        return ""

    lines = ["", "---", "**সূত্র (Sources)**"]
    for tag in seen:
        meta = source_map[tag]
        category = meta.get("category", "")
        topic = meta.get("topic", "")
        passage_id = meta.get("passage_id", "")
        tool = meta.get("tool", "")
        chunk_type = meta.get("chunk_type", "")
        descriptor_bits = [b for b in (category, topic) if b]
        descriptor = " — ".join(descriptor_bits) if descriptor_bits else "(no metadata)"
        suffix = f" · passage_id {passage_id}" if passage_id != "" else ""
        tool_suffix = f" ({tool})" if tool else ""
        source_label = ""
        if chunk_type == "wiki":
            source_label = " · উইকিপিডিয়া"
        elif chunk_type == "govt_service":
            source_label = " · সরকারি সেবা"
        lines.append(f"- [{tag}] {descriptor}{suffix}{tool_suffix}{source_label}")
    return "\n".join(lines)
