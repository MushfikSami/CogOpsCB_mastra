"""
cogops/pipeline/query_expand.py

Document-type-aware query expander with hardcoded fast-path + LLM fallback.

Runs on every sub-query AFTER the normalizer but BEFORE Jiggasha embedding.
Purpose:
  1. Hardcoded regex catches the ~20 most common document types instantly.
  2. LLM fallback catches rare / new / OOS document types without code changes.
  3. Append formal corpus-aligned synonyms so the embedding model sees
     the exact terms used in the corpus (improving recall).

This is NOT a replacement for the LLM router's reformulation; it is a
safety-net that catches colloquial document-type references the router
may have missed or softened.
"""

import asyncio
import json
import logging
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bengali character class for word-boundary checks.
# NOTE: \b in Python regex does NOT work correctly for Bengali because
# vowel signs (e.g., া, ি) are not matched by \w, causing false boundaries.
_BN_CHAR_CLASS = r"[\u0980-\u09FF]"

# Bengali suffix characters commonly appended to nouns (possessive, locative, etc.)
_BN_SUFFIX_CHARS = "ািীুূৃেোৈৌংঃ়ৎ্ঁস"

# ---------------------------------------------------------------------------
# Hardcoded fast-path expansion map.
# Key   = regex pattern that detects the informal/short form in a query.
# Value = list of formal corpus-aligned terms to append.
#
# Patterns use a negative lookbehind + optional suffix consumption so that
# inflected forms (e.g., সার্টিফিকেটে, সনদের) are still matched.
# ---------------------------------------------------------------------------
_DOC_TYPE_EXPANSIONS: List[tuple[re.Pattern[str], List[str]]] = [
    # Marriage / divorce certificates
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বিয়ের\s+সার্টিফিকেট|বিয়ের\s+সার্টিফিকেট|বিবাহ\s+সার্টিফিকেট|বিয়ের\s+সনদ|বিয়ের\s+সনদ|বিবাহ\s+সনদ|বিবাহিত\s+প্রত্যয়ন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["বিবাহ সনদ", "বিবাহিত প্রত্যয়ন"],
    ),
    # NID / smart card
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(এনআইডি\s+কার্ড|এন\.আই\.ডি|এনআইডি|ভোটার\s+আইডি|স্মার্ট\s+কার্ড)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["জাতীয় পরিচয়পত্র", "স্মার্ট কার্ড"],
    ),
    # Passport
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(পাসপোর্ট|ই-পাসপোর্ট|ই\s+পাসপোর্ট)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["ই-পাসপোর্ট", "পাসপোর্ট"],
    ),
    # Birth certificate
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(জন্ম\s+সার্টিফিকেট|জন্ম\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["জন্ম নিবন্ধন সনদ"],
    ),
    # Death certificate
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(মৃত্যু\s+সার্টিফিকেট|মৃত্যু\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["মৃত্যু নিবন্ধন সনদ"],
    ),
    # Education certificates
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(এসএসসি\s+সার্টিফিকেট|এস\.এস\.সি\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["এসএসসি সনদ"],
    ),
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(এইচএসসি\s+সার্টিফিকেট|এইচ\.এস\.সি\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["এইচএসসি সনদ"],
    ),
    # Driving license
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ড্রাইভিং\s+লাইসেন্স|ড্রাইভার\s+লাইসেন্স)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["ড্রাইভিং লাইসেন্স"],
    ),
    # Trade license
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ট্রেড\s+লাইসেন্স|ব্যবসা\s+লাইসেন্স)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["ট্রেড লাইসেন্স"],
    ),
    # Character certificate
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(চারিত্রিক\s+সনদ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["চারিত্রিক সনদ"],
    ),
    # Police clearance
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(পুলিশ\s+ক্লিয়ারেন্স|পুলিশ\s+ক্লিয়ারেন্স)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["পুলিশ ক্লিয়ারেন্স"],
    ),
    # Disability certificate
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(প্রতিবন্ধী\s+সনদ|প্রতিবন্ধী\s+কার্ড)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["প্রতিবন্ধী সনদ"],
    ),
    # Land records
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(জমি\s+দলিল|খতিয়ান|সিএস\s+খতিয়ান|বিএস\s+খতিয়ান)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["ভূমি রেকর্ড", "জমি"],
    ),
    # Utilities
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বিদ্যুৎ\s+বিল|বিদ্যুৎ\s+সংযোগ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["বিদ্যুৎ"],
    ),
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(গ্যাস\s+বিল|গ্যাস\s+সংযোগ)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["গ্যাস"],
    ),
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(পানি\s+বিল|ওয়াসা\s+বিল|ওয়াসা)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["ওয়াসা"],
    ),
    # Tax
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ট্যাক্স|ভ্যাট|মূসক)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["ট্যাক্স", "ভ্যাট", "কর", "মূসক"],
    ),
    # Savings certificate
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(সঞ্চয়পত্র)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["সঞ্চয়পত্র"],
    ),
    # BPSC / Bangladesh Public Service Commission
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বাংলাদেশ\s+কর্ম\s+কমিশন(?:ের)?|বিসিএস(?:ের)?|BPSC|সরকারি\s+চাকরি\s+কমিশন(?:ের)?)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["বাংলাদেশ কর্ম কমিশন", "বিসিএস"],
    ),
    # Metro / MRT
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(মেট্রো|মেট্রোরেল|MRT)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["মেট্রোরেল", "MRT"],
    ),
    # Plane / air
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(বিমান|এয়ারলাইন্স|প্লেন)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["বিমান"],
    ),
    # Train / rail
    (
        re.compile(r"(?<!" + _BN_CHAR_CLASS + r")(ট্রেন|রেল)(?:[" + _BN_SUFFIX_CHARS + r"]*)(?!" + _BN_CHAR_CLASS + r")"),
        ["রেল"],
    ),
]


# ---------------------------------------------------------------------------
# Hardcoded fast-path expander (synchronous, zero latency).
# ---------------------------------------------------------------------------

def expand_sub_query(query: str) -> str:
    """Append formal corpus-aligned synonyms for detected document types.

    Returns the original query plus expansion terms (separated by ' | ').
    If no document type is detected, returns the query unchanged.
    """
    if not query:
        return query

    expansions: List[str] = []
    for regex, formal_terms in _DOC_TYPE_EXPANSIONS:
        if regex.search(query):
            for term in formal_terms:
                if term not in query and term not in expansions:
                    expansions.append(term)

    if not expansions:
        return query

    return query + " | " + " | ".join(expansions)


def expand_sub_queries(queries: List[str]) -> List[str]:
    """Expand a list of sub-queries using the hardcoded fast-path only."""
    return [expand_sub_query(q) for q in queries]


# ---------------------------------------------------------------------------
# LLM fallback expander (async, called only when hardcoded path misses).
# ---------------------------------------------------------------------------

_EXPANDER_SYSTEM_PROMPT = """\
You are a Bengali government-service term formalizer.

Given a user query, detect if it EXPLICITLY names a specific document,
certificate, license, commission, or government service type.
If so, return 1-3 formal Bengali terms used in official government corpora.

Rules:
- Only detect EXPLICITLY named types (e.g., "বিয়ের সার্টিফিকেট",
  "হজ সনদ", "বাংলাদেশ কর্ম কমিশন", "চারিত্রিক সনদ").
- Do NOT invent terms. Only return formal synonyms for types the user
  actually named.
- If no specific type is named, return an empty list.

Return compact JSON only, no markdown, no prose:
{"detected":true|false,"terms":["term1",...]}"""


class _ExpanderCache:
    """Thread-safe LRU cache for LLM expansion results."""

    def __init__(self, maxsize: int = 256):
        self._store: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)


_EXPANDER_CACHE = _ExpanderCache(maxsize=256)


def _parse_expander_response(raw: str) -> List[str]:
    """Extract terms from the LLM JSON response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return []
        if not data.get("detected"):
            return []
        terms = data.get("terms", []) or data.get("formal_terms", [])
        if isinstance(terms, list):
            return [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


async def expand_sub_query_llm(
    query: str,
    secondary_client: Any,
    secondary_model: str,
    timeout: float = 8.0,
) -> str:
    """Expand a single sub-query using hardcoded fast-path + optional LLM fallback.

    1. Try the hardcoded regex map first (zero latency).
    2. If that misses, check the in-memory LRU cache.
    3. If cache misses too, call the secondary LLM to detect document types
       and return formal synonyms.
    4. Cache the LLM result and append terms to the query.

    On any LLM error (timeout, malformed JSON, etc.) the query is returned
    unchanged so the pipeline never blocks on the expander.
    """
    if not query:
        return query

    # 1. Fast path — hardcoded regex.
    hardcoded = expand_sub_query(query)
    if hardcoded != query:
        return hardcoded

    # 2. Cache lookup.
    cache_key = unicodedata.normalize("NFC", query).strip()
    cached = await _EXPANDER_CACHE.get(cache_key)
    if cached is not None:
        if cached != query:
            logger.debug("expander cache hit: %r → %r", query, cached)
        return cached

    # 3. LLM fallback.
    try:
        resp = await asyncio.wait_for(
            secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _EXPANDER_SYSTEM_PROMPT},
                    {"role": "user", "content": f'Query: "{query}"\n\nReturn compact JSON:'},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=128,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        terms = _parse_expander_response(raw)
        if terms:
            # Deduplicate and filter out terms already present in the query.
            new_terms = [t for t in terms if t.lower() not in query.lower()]
            if new_terms:
                result = query + " | " + " | ".join(new_terms)
                await _EXPANDER_CACHE.set(cache_key, result)
                logger.info("expander LLM: %r → terms=%s", query, new_terms)
                return result

        # No terms found — cache the unexpanded query to avoid re-calling.
        await _EXPANDER_CACHE.set(cache_key, query)
        return query

    except asyncio.TimeoutError:
        logger.warning("expander LLM timed out after %.1fs for %r", timeout, query)
    except Exception as e:
        logger.warning("expander LLM failed for %r: %s", query, e)

    # 4. On any error, return unchanged (fail-open).
    return query


async def expand_sub_queries_llm(
    queries: List[str],
    secondary_client: Any,
    secondary_model: str,
) -> List[str]:
    """Expand a list of sub-queries using fast-path + conditional LLM fallback.

    Each query is processed independently; the LLM is only called for queries
    that the hardcoded map does not recognize.
    """
    results = []
    for q in queries:
        expanded = await expand_sub_query_llm(q, secondary_client, secondary_model)
        results.append(expanded)
    return results


# ---------------------------------------------------------------------------
# Document-type mismatch detection for the pipeline.
# ---------------------------------------------------------------------------

_DOC_TYPE_META_KEYWORDS: Dict[str, List[str]] = {
    # Marriage certificate — keep keywords document-type-specific.
    # Generic terms like "বিয়ে" / "বিবাহ" match NID name-change-after-marriage
    # passages and cause false negatives; use certificate-specific terms only.
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
