"""
cogops/pipeline/query_expand.py

Document-type guard for the factual pipeline.

Provides:
  - check_document_type_match()  — verify retrieved passages match the user's
    explicitly named document type (e.g. marriage certificate, NID, passport).
  - extract_document_type()      — regex-based extraction of document type
    from a Bengali query string.

The query formalization/expansion step has been REMOVED from the pipeline.
Formalization is now handled by the router (sub_query generation) and by
Jiggasha's dynamic instruction generator.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Backward-compat stubs (query expansion is now handled by router + Jiggasha)
def expand_sub_query(query: str) -> str:
    """Passthrough — no longer used by the pipeline."""
    return query


def expand_sub_queries(queries: List[str]) -> List[str]:
    """Passthrough — no longer used by the pipeline."""
    return [expand_sub_query(q) for q in queries]


async def expand_sub_query_llm(query: str, *args, **kwargs) -> str:
    """Backward-compat stub — returns query unchanged."""
    return query


async def expand_sub_queries_llm(queries: List[str], *args, **kwargs) -> List[str]:
    """Backward-compat stub — returns queries unchanged."""
    return [q for q in queries]


# Backward-compat cache alias (no longer functional but safe to import)
_EXPANDER_CACHE = None


# ---------------------------------------------------------------------------
# Document-type mismatch detection for the pipeline.
# ---------------------------------------------------------------------------

_DOC_TYPE_META_KEYWORDS: Dict[str, List[str]] = {
    "বিবাহ সনদ": [
        "বিবাহ সনদ", "বিয়ের সনদ", "বিবাহ সার্টিফিকেট", "বিয়ের সার্টিফিকেট",
        "বিবাহিত প্রত্যয়ন", "কাবিননামা", "নিকাহনামা", "marriage certificate",
    ],
    "জন্ম সনদ": ["জন্ম", "নিবন্ধন", "birth"],
    "মৃত্যু সনদ": ["মৃত্যু", "death"],
    "এসএসসি সনদ": ["এসএসসি", "ssc", "শিক্ষা", "বোর্ড"],
    "এইচএসসি সনদ": ["এইচএসসি", "hsc", "শিক্ষা", "বোর্ড"],
    "এনআইডি": ["এনআইডি", "পরিচয়পত্র", "স্মার্ট কার্ড", "ভোটার", "nid", "smart card", "voter"],
    "পাসপোর্ট": ["পাসপোর্ট", "passport", "e-passport"],
    "ড্রাইভিং লাইসেন্স": ["ড্রাইভিং", "driving"],
    "ট্রেড লাইসেন্স": ["ট্রেড", "trade"],
    "চারিত্রিক সনদ": ["চারিত্রিক", "character"],
    "পুলিশ ক্লিয়ারেন্স": ["পুলিশ", "ক্লিয়ারেন্স", "police", "clearance"],
    "প্রতিবন্ধী সনদ": ["প্রতিবন্ধী", "disability"],
    "ভূমি": ["ভূমি", "জমি", "খতিয়ান", "দলিল", "land"],
    "বিদ্যুৎ": ["বিদ্যুৎ", "নেসকো", "ডেসকো", "ডিপিডিসি", "electricity", "desco", "nesco", "dpdc"],
    "গ্যাস": ["গ্যাস", "gas"],
    "ওয়াসা": ["ওয়াসা", "পানি", "wasa", "water"],
    "ট্যাক্স": ["ট্যাক্স", "ভ্যাট", "কর", "মূসক", "tax", "vat"],
    "সঞ্চয়পত্র": ["সঞ্চয়পত্র", "savings"],
    "বিসিএস": ["বাংলাদেশ কর্ম কমিশন", "বিসিএস", "BPSC", "সরকারি চাকরি কমিশন"],
    "মেট্রোরেল": ["মেট্রো", "MRT", "metro"],
    "বিমান": ["বিমান", "এয়ারলাইন্স", "plane", "air"],
    "রেল": ["রেল", "ট্রেন", "rail", "train"],
}


def check_document_type_match(
    raw_query: str,
    source_map: Dict[str, Dict[str, Any]],
) -> bool:
    """Return True if at least one passage matches the user's document type.

    When the user explicitly names a document type (e.g., বিবাহ সনদ) and
    NONE of the retrieved passages mention that document type in their
    metadata, we consider this a mismatch and should refuse instead of
    disambiguating with irrelevant options.
    """
    doc_type = extract_document_type(raw_query)
    if doc_type is None:
        # User did not explicitly name a document type — skip this check.
        return True

    keywords = _DOC_TYPE_META_KEYWORDS.get(doc_type, [])
    if not keywords:
        return True

    for meta in source_map.values():
        # Check structured metadata fields only — NOT the full passage text.
        # A passage that mentions a document type tangentially (e.g. as a
        # required supporting document for NID) should NOT count as a match.
        meta_text = unicodedata.normalize(
            "NFC",
            " ".join(
                str(meta.get(k, "") or "")
                for k in ("category", "sub_category", "service", "topic")
            ),
        ).lower()
        if any(
            unicodedata.normalize("NFC", kw.lower()) in meta_text
            for kw in keywords
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Document-type extraction for downstream mismatch detection.
# ---------------------------------------------------------------------------

_BN_CHAR_CLASS = r"[\u0980-\u09FF]"
_BN_SUFFIX_CHARS = "ািীুূৃেোৈৌংঃ়ৎ্ঁসরয"


def extract_document_type(query: str) -> Optional[str]:
    """Return the formal document type if the query explicitly names one.

    This is used by the pipeline to detect whether retrieved passages
    actually match the user's specific document type.
    """
    if not query:
        return None

    # Priority-ordered list: more specific patterns first.
    _DOC_TYPE_PATTERNS: List[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বিয়ের\s+সার্টিফিকেট|বিয়ের\s+সার্টিফিকেট|বিবাহ\s+সার্টিফিকেট|বিয়ের\s+সনদ|বিয়ের\s+সনদ|বিবাহ\s+সনদ|বিবাহিত\s+প্রত্যয়ন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "বিবাহ সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(জন্ম\s+সার্টিফিকেট|জন্ম\s+সনদ|জন্ম\s+নিবন্ধন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "জন্ম সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(মৃত্যু\s+সার্টিফিকেট|মৃত্যু\s+সনদ|মৃত্যু\s+নিবন্ধন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "মৃত্যু সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(এসএসসি\s+সার্টিফিকেট|এসএসসি\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "এসএসসি সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(এইচএসসি\s+সার্টিফিকেট|এইচএসসি\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "এইচএসসি সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(এনআইডি\s+কার্ড|এন\.আই\.ডি|এনআইডি|জাতীয়\s+পরিচয়পত্র|স্মার্ট\s+কার্ড|ভোটার\s+আইডি)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "এনআইডি"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ই-পাসপোর্ট|পাসপোর্ট)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "পাসপোর্ট"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ড্রাইভিং\s+লাইসেন্স)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "ড্রাইভিং লাইসেন্স"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ট্রেড\s+লাইসেন্স)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "ট্রেড লাইসেন্স"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(চারিত্রিক\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "চারিত্রিক সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(পুলিশ\s+ক্লিয়ারেন্স)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "পুলিশ ক্লিয়ারেন্স"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(প্রতিবন্ধী\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "প্রতিবন্ধী সনদ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(খতিয়ান|জমি\s+দলিল)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "ভূমি"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বিদ্যুৎ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "বিদ্যুৎ"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(গ্যাস)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "গ্যাস"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ওয়াসা|ওয়াসা)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "ওয়াসা"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ট্যাক্স|ভ্যাট|মূসক)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "ট্যাক্স"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(সঞ্চয়পত্র)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "সঞ্চয়পত্র"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বাংলাদেশ\s+কর্ম\s+কমিশন(?:ের)?|বিসিএস(?:ের)?|BPSC|সরকারি\s+চাকরি\s+কমিশন(?:ের)?)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "বিসিএস"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(মেট্রোরেল|MRT)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "মেট্রোরেল"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বিমান|প্লেন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "বিমান"),
        (re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(রেল|ট্রেন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"), "রেল"),
    ]

    for regex, formal_type in _DOC_TYPE_PATTERNS:
        if regex.search(query):
            return formal_type

    return None
