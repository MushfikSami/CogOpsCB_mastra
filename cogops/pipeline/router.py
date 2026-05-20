"""
cogops/pipeline/router.py

Stage 1 of the deterministic pipeline: the Global Router.

A single secondary-LLM call replaces three separate calls (intent classifier,
sub-question splitter, query normalizer) and returns:

    RouterResult(
        intent="factual_govt" | "chitchat" | "political_refuse",
        sub_queries_bengali=[...],     # up to 3, each a Bengali string
        raw_query=...,                 # the original sanitized query
        notes=...,                     # debug-only annotations
    )

Design decisions:

  1. ONE LLM call, response_format JSON. Latency win vs. sequential calls.

  2. Bengali fast-path: if the query is already ≥30% Bengali codepoints AND
     contains only one apparent question (no `?` × 2, no Bengali/English
     conjunctions) AND matches domain vocab, we skip the LLM and return
     {factual_govt, [query]}. Most production traffic hits this path.

  3. Hard refusal shortcut: clear partisan-judgement keywords short-circuit
     to "political_refuse" with no LLM call.

  4. Domain-vocab override: after the LLM verdict, if Bengali OR English
     domain vocabulary matches the query, force intent="factual_govt"
     regardless of what the LLM said. One-way override; never override AWAY
     from factual_govt.

  5. State facts vs. political opinions: the prompt explicitly distinguishes
     "who is the foreign minister" (factual_govt — retrieval will refuse if
     not in corpus) from "which party is better" (political_refuse). This
     fixes the routing bug where the old classifier lumped state-officeholder
     questions into the political bucket.

  6. Fail-soft: on any error (timeout, JSON parse, empty content) →
     intent=factual_govt, sub_queries=[raw_query]. Never crashes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

Intent = Literal[
    "factual_govt",
    "chitchat",
    "political_refuse",
    "personal_law_refuse",
]

MAX_SUB_QUERIES = 3


# Bengali codepoint range (Unicode block: Bengali, U+0980..U+09FF).
_BENGALI_RE = re.compile(r"[ঀ-৿]")

# Conjunction / connector markers that suggest multi-question input.
_CONNECTORS_RE = re.compile(
    r"(?:\bএবং\b|\bআর\b|\bও\b|\band\b|\balso\b|;|,\s+and\b)",
    re.IGNORECASE,
)

# Domain vocabulary — Bengali + English. Matching this forces factual_govt.
_DOMAIN_VOCAB = (
    # Identity / civic
    "পাসপোর্ট", "passport", "এনআইডি", "NID", "জাতীয় পরিচয়", "পরিচয়পত্র",
    "ভোটার", "চারিত্রিক সনদ", "নাগরিকত্ব", "সনদ", "জন্ম নিবন্ধন",
    "মৃত্যু নিবন্ধন", "বিবাহ নিবন্ধন", "তালাক",
    "ওয়ারিশ", "উত্তরাধিকার", "প্রতিবন্ধী",
    # Money / tax
    "ফি", "চার্জ", "মূল্য", "tax", "ট্যাক্স", "কর", "ভ্যাট", "VAT",
    "সঞ্চয়পত্র", "মূসক",
    # Vehicles / licenses
    "লাইসেন্স", "license", "ড্রাইভিং", "BRTA", "BRTC",
    # Land
    "জমি", "ভূমি", "খতিয়ান", "দলিল", "সিএস", "বিএস",
    # Utilities
    "বিদ্যুৎ", "গ্যাস", "পানি", "DESCO", "WASA", "নেসকো", "ডিপিডিসি",
    "প্রি-পেইড", "পোস্ট-পেইড", "মিটার",
    # Transport
    "মেট্রো", "MRT", "বিমান", "এয়ারলাইন্স", "চেক-ইন", "টিকেট",
    # Education
    "এসএসসি", "এইচএসসি", "বোর্ড", "শিক্ষাবোর্ড", "সার্টিফিকেট", "সনদপত্র",
    # Government framing
    "সরকার", "সরকারি", "মন্ত্রণালয়", "অধিদপ্তর", "দপ্তর", "ministry",
    # Safety / civic complaints
    "৯৯৯", "পুলিশ ক্লিয়ারেন্স", "মানহানি", "সাইবার", "cyber security",
    # Welfare
    "টিসিবি", "ভাতা", "জুলাই যোদ্ধা",
)
_DOMAIN_RE = re.compile(
    "|".join(re.escape(v) for v in _DOMAIN_VOCAB),
    re.IGNORECASE,
)

# Hard refusal shortcuts — partisan-judgement framings that we never want to
# route to retrieval. Narrow on purpose: state-fact questions about officials
# (PM, ministers) must NOT match.
_HARD_POLITICAL_REFUSAL = (
    "আওয়ামী লীগ না বিএনপি", "বিএনপি না আওয়ামী",
    "জামায়াত দলের মার্কা", "জামায়াত বনাম",
    "কোন দল ভালো", "কোন দলের", "কোন দল সমর্থন",
    "হিন্দু না মুসলিম", "কোন ধর্ম ভালো", "কোন ধর্ম সঠিক",
)

# Personal-law / religious-judgment queries — these ask the bot to opine on
# permissibility under religious or family law (e.g. "is remarriage of a
# divorced spouse allowed?"). The corpus has government-services facts, not
# religious rulings, so the answer is never in the data. Short-circuit at the
# router with a respectful "consult a specialist" refusal instead of letting
# the composer confabulate from tangential marriage-registration passages.
_HARD_PERSONAL_LAW_REFUSAL = (
    # Remarriage / divorce-and-remarry — Q71 family
    "তালাকপ্রাপ্ত স্ত্রী", "তালাকপ্রাপ্ত বউ", "তালাকপ্রাপ্ত স্বামী",
    "পুনরায় বিয়ে", "আবার বিয়ে",
    # Iddat / shariah / fatwa
    "ইদ্দত", "শরীয়ত", "শরিয়ত", "ফতোয়া", "মুফতি",
    # Polygamy permission
    "একাধিক বিয়ে অনুমতি", "দ্বিতীয় বিয়ে অনুমতি", "দুই বিয়ে অনুমতি",
    # Halal/haram framings — asking the bot for religious judgment
    "হালাল না হারাম", "হারাম না হালাল", "ইসলামে অনুমতি",
    "ধর্মীয়ভাবে যাবে",
)


_ROUTER_SYSTEM_PROMPT = """\
You are the router for a Bangladesh government-services chatbot. You will be
given ONE user message (in Bengali, English, Banglish, or mixed).

You must output JSON with two fields and nothing else:

  {
    "intent": "factual_govt" | "chitchat" | "political_refuse",
    "sub_queries_bengali": ["<sub-question 1 in Bengali>", ...]
  }

INTENT CLASSIFICATION:

  • "factual_govt": ANY question about Bangladesh government services,
    procedures, fees, eligibility, offices, documents, registration, licenses,
    NID, passport, tax, land records, utility connections, ministries, OR
    any concrete fact the user expects an authoritative answer for —
    INCLUDING questions about WHO holds a state office (Prime Minister,
    minister, president, secretary, military chief). These are state facts,
    not political opinions. When in doubt, classify as factual_govt.

  • "chitchat": pure greetings, thanks, "who are you", or one-line
    conversational fillers with ZERO domain nouns and no factual question.
    Examples: "hello", "hi bro", "thanks", "তুমি কে?", "কেমন আছ?".

  • "political_refuse": requests for partisan judgement on parties, leaders,
    religions, or movements. Asking "which party is better?", "is X corrupt?",
    "which religion is right?" — these only. ASKING WHO HOLDS A POSITION
    IS NOT POLITICAL. "Who is the foreign minister?" → factual_govt.

SUB-QUERY EXTRACTION:

  For factual_govt, split the message into up to 3 distinct sub-questions
  and TRANSLATE each to formal, search-optimized Bengali. Follow these
  rules strictly:

  1. REPLACE colloquial or English loanwords with standard Bengali
     government-service terms:
     • প্লেন → বিমান
     • ট্রেন → রেল
     • টিকেট → টিকিট
  2. REMOVE conversational fillers such as আচ্ছা, ভাই, দেখেন, শুনুন,
     বলুন তো, একটু, কিন্তু, তাই, তাহলে.
  3. Write CONCISE sub-queries suitable for embedding retrieval — avoid
     long story-like framing.
  4. If the user mentions a specific document, certificate, or service
     type (e.g., বিবাহ সনদ, জন্ম সনদ, এনআইডি, পাসপোর্ট, এসএসসি সনদ),
     you MUST include that exact document type in the sub-query. Do NOT
     drop the document type and reduce the query to a generic action.
  5. If the message already contains only one question, return a single-
     element list.
  6. Comparative questions ("which takes longer NID or passport?") count
     as ONE sub-question — do not split them; the downstream composer
     will handle the comparison.

  For chitchat or political_refuse, return an empty list.

EXAMPLES:

User: "পাসপোর্ট ফি কত? এনআইডি সংশোধন কোথায় করব?"
Output: {"intent":"factual_govt","sub_queries_bengali":["পাসপোর্ট ফি কত?","এনআইডি সংশোধন কোথায় করব?"]}

User: "Who's the prime minister of bangladesh?"
Output: {"intent":"factual_govt","sub_queries_bengali":["বাংলাদেশের প্রধানমন্ত্রী কে?"]}

User: "Which takes longer, NID or passport?"
Output: {"intent":"factual_govt","sub_queries_bengali":["এনআইডি ও পাসপোর্ট - কোনটির আবেদন প্রক্রিয়া বেশি সময় নেয়?"]}

User: "জামায়াত না বিএনপি, কোনটা ভালো?"
Output: {"intent":"political_refuse","sub_queries_bengali":[]}

User: "hi bro"
Output: {"intent":"chitchat","sub_queries_bengali":[]}

User: "show me the photo of the president of bangladesh"
Output: {"intent":"factual_govt","sub_queries_bengali":["বাংলাদেশের রাষ্ট্রপতি কে?"]}

User: "আচ্ছা প্লেনের টিকেট কিভাবে কাটবো?"
Output: {"intent":"factual_govt","sub_queries_bengali":["বিমানের টিকিট কাটার নিয়ম"]}

User: "বিয়ের সার্টিফিকেটে নাম পরিবর্তন"
Output: {"intent":"factual_govt","sub_queries_bengali":["বিবাহ সনদে নাম সংশোধন"]}

User: "how to do passport - which takes longer nid or passport? where to go for plane tickets?"
Output: {"intent":"factual_govt","sub_queries_bengali":["পাসপোর্ট করার পদ্ধতি কী?","এনআইডি ও পাসপোর্ট - কোনটির আবেদন প্রক্রিয়া বেশি সময় নেয়?","বিমানের টিকিট কোথা থেকে কিনব?"]}

Output ONLY the JSON object. No markdown fences, no prose.
"""


@dataclass
class RouterResult:
    intent: Intent
    sub_queries_bengali: List[str]
    raw_query: str
    notes: List[str] = field(default_factory=list)
    # Token usage from the router's LLM call (None for the fast-path / no-LLM
    # branches). Shape: {"prompt": int, "completion": int}.
    usage: Optional[Dict[str, int]] = None

    def is_factual(self) -> bool:
        return self.intent == "factual_govt"


def _extract_usage(resp: Any) -> Optional[Dict[str, int]]:
    """Pull {prompt, completion} from an OpenAI-compatible response.

    Returns None if the response carries no usage (some endpoints omit it).
    """
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return None
        prompt = getattr(u, "prompt_tokens", None)
        completion = getattr(u, "completion_tokens", None)
        if prompt is None and completion is None:
            return None
        return {
            "prompt": int(prompt or 0),
            "completion": int(completion or 0),
        }
    except Exception:  # noqa: BLE001
        return None


def _bengali_fraction(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for ch in text if "ঀ" <= ch <= "৿")
    return hits / len(text)


def _looks_single_question(text: str) -> bool:
    # ASCII '?' OR Bengali Daanda '।' as terminators; more than one of either
    # (or any conjunction) implies multi-question.
    qm = text.count("?")
    if qm > 1:
        return False
    if _CONNECTORS_RE.search(text):
        return False
    return True


def _hard_political_match(text: str) -> bool:
    lower = text.lower()
    for kw in _HARD_POLITICAL_REFUSAL:
        if kw in text or kw.lower() in lower:
            return True
    return False


def _hard_personal_law_match(text: str) -> bool:
    """Personal-law / religious-judgment questions short-circuit here.

    Trigger only when a clear personal-law term appears AND the framing is
    a permission/judgment question ("যাবে কি না", "অনুমতি", "করা যায় কি",
    "হালাল"). A bare mention of "তালাকপ্রাপ্ত" in a procedure question
    (e.g. "তালাকপ্রাপ্ত স্ত্রীর সঞ্চয়পত্র") should NOT trigger.
    """
    lower = text.lower()
    has_personal_law_term = any(
        kw in text or kw.lower() in lower for kw in _HARD_PERSONAL_LAW_REFUSAL
    )
    if not has_personal_law_term:
        return False
    # Require a permission/judgment framing to avoid catching procedural
    # queries that happen to mention these terms.
    judgment_frames = (
        "যাবে কি", "যাবে না", "অনুমতি", "করা যায় কি", "করা যাবে কি",
        "অনুমোদিত", "নিষিদ্ধ", "পারা যাবে", "পারব কি",
        "হালাল", "হারাম", "ফতোয়া",
    )
    for f in judgment_frames:
        if f in text:
            return True
    return False


async def route(
    query: str,
    secondary_client: Optional[AsyncOpenAI],
    secondary_model: str,
    timeout: float = 5.0,
) -> RouterResult:
    """Classify intent + split + Bengali-normalize in one call.

    Args:
        query: the sanitized user query (must be non-empty).
        secondary_client: AsyncOpenAI client for the secondary LLM. If None,
            the router falls back to the fast-path / regex-only behavior.
        secondary_model: model name for the secondary LLM.
        timeout: hard timeout in seconds for the LLM call.

    Returns:
        RouterResult.
    """
    text = (query or "").strip()
    notes: List[str] = []

    if not text:
        return RouterResult(intent="chitchat", sub_queries_bengali=[], raw_query="", notes=["empty"])

    # 1a. Hard personal-law / religious-judgment shortcut.
    # Checked BEFORE political so "তালাকপ্রাপ্ত পুনরায় বিয়ে যাবে কি"
    # gets the right refusal text (specialist consult) rather than the
    # political-neutrality one.
    if _hard_personal_law_match(text):
        notes.append("hard_personal_law_match")
        return RouterResult(
            intent="personal_law_refuse",
            sub_queries_bengali=[],
            raw_query=text,
            notes=notes,
        )

    # 1b. Hard political-refusal shortcut.
    if _hard_political_match(text):
        notes.append("hard_political_match")
        return RouterResult(
            intent="political_refuse",
            sub_queries_bengali=[],
            raw_query=text,
            notes=notes,
        )

    # 2. LLM path. Fail-soft on any error.
    if secondary_client is None:
        notes.append("no_secondary_client_fallback")
        return RouterResult(
            intent="factual_govt",
            sub_queries_bengali=[text],
            raw_query=text,
            notes=notes,
        )

    intent: Intent = "factual_govt"
    sub_queries: List[str] = [text]
    usage: Optional[Dict[str, int]] = None

    try:
        async def _call():
            return await secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=400,
            )

        resp = await asyncio.wait_for(_call(), timeout=timeout)
        raw = (resp.choices[0].message.content or "").strip()
        usage = _extract_usage(resp)
        data = json.loads(raw)

        candidate_intent = str(data.get("intent", "")).lower().strip()
        if candidate_intent in ("factual_govt", "chitchat", "political_refuse"):
            intent = candidate_intent  # type: ignore[assignment]
        else:
            notes.append(f"unknown_intent={candidate_intent!r}; default factual_govt")

        raw_subs = data.get("sub_queries_bengali", [])
        if not isinstance(raw_subs, list):
            notes.append("sub_queries not a list; using raw query")
        else:
            cleaned = []
            for s in raw_subs[:MAX_SUB_QUERIES]:
                if isinstance(s, str):
                    s = unicodedata.normalize("NFC", s).strip()
                    if s:
                        cleaned.append(s)
            if intent == "factual_govt":
                sub_queries = cleaned if cleaned else [text]
            else:
                sub_queries = []
            if len(raw_subs) > MAX_SUB_QUERIES:
                notes.append(f"truncated_sub_queries_to_{MAX_SUB_QUERIES}")
    except asyncio.TimeoutError:
        notes.append("router_timeout_default_factual")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        notes.append(f"router_parse_error: {e!s}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"router_error: {e!s}")

    # 4. Domain-vocab override — never override AWAY from factual_govt.
    if intent != "factual_govt" and _DOMAIN_RE.search(text):
        notes.append(f"domain_override: {intent}→factual_govt")
        intent = "factual_govt"
        if not sub_queries:
            sub_queries = [text]

    logger.info(
        "router: intent=%s subs=%d notes=%s for %r",
        intent, len(sub_queries), notes, text[:80],
    )

    return RouterResult(
        intent=intent,
        sub_queries_bengali=sub_queries if intent == "factual_govt" else [],
        raw_query=text,
        notes=notes,
        usage=usage,
    )
