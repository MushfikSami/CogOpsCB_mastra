"""
cogops/verifier/intent.py

Pre-loop intent classifier. Determines whether the user message should:

  - "factual"  → run the full grounded-RAG pipeline (forced tool call + citation + NLI)
  - "chitchat" → run the no-tools streaming path (greetings, thanks, identity questions)
  - "refuse"   → emit the neutral political/abusive refusal without an LLM call

Implementation:
  1. Secondary-LLM structured-JSON classifier (deterministic temperature=0).
  2. Belt-and-braces regex post-check on Bengali government-domain vocabulary —
     if any domain noun appears, override to "factual" regardless of classifier
     verdict. The override is one-way: never override AWAY from "factual".

This is the load-bearing gate that lets greetings bypass forced retrieval while
keeping every factual question on the grounded path.
"""

import json
import logging
import re
from typing import Literal, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

Intent = Literal["factual", "chitchat", "refuse"]


# Bengali domain vocabulary — if any of these appear in the user message we
# force "factual" regardless of what the classifier said. Hard-bias toward
# the grounded path (false positives on factual are cheap; false negatives are
# the actual failure mode we're guarding against).
_DOMAIN_VOCAB = (
    "পাসপোর্ট", "passport",
    "এনআইডি", "NID", "জাতীয় পরিচয়", "পরিচয়পত্র",
    "ভোটার", "ভোটাধিকার",
    "ফি", "চার্জ", "মূল্য",
    "লাইসেন্স", "license", "ড্রাইভিং",
    "ট্যাক্স", "কর", "tax", "ভ্যাট", "VAT",
    "জন্ম নিবন্ধন", "মৃত্যু নিবন্ধন", "বিবাহ নিবন্ধন",
    "চারিত্রিক সনদ", "নাগরিকত্ব", "সনদ",
    "জমি", "ভূমি", "খতিয়ান", "দলিল",
    "বিদ্যুৎ", "গ্যাস", "পানি",  # utility bills/connections
    "মন্ত্রণালয়", "অধিদপ্তর", "দপ্তর", "ministry",
    "BRTA", "BRTC", "DESCO", "WASA",
    "আবেদন", "অনুমোদন", "নিবন্ধন",
    "সরকার", "সরকারি",
    "ই-পাসপোর্ট", "ই-গভার্ন্যান্স",
)

_DOMAIN_RE = re.compile("|".join(re.escape(v) for v in _DOMAIN_VOCAB), re.IGNORECASE)


# Topics that should be refused outright (political, religious, abusive).
# Triggers the "refuse" branch without an LLM call when the classifier flags them.
_REFUSE_KEYWORDS_BN = (
    # Political parties — only refuse on clear partisan framing, not on policy
    "আওয়ামী লীগ", "বিএনপি", "জামায়াত", "জাতীয় পার্টি",
    # Religion-vs-politics framings
    "হিন্দু না মুসলিম", "কোন ধর্ম ভালো",
)


_SYSTEM = """\
You are an intent classifier for a Bangladesh government-services chatbot.

Classify the user's Bengali (or mixed Bengali+English) message into EXACTLY ONE of:

  - "factual": ANY question about Bangladesh government services, procedures,
    fees, eligibility, offices, documents, registration, licenses, NID, passport,
    tax, land records, utility connections, ministries, or any concrete fact the
    user expects a citable answer for. When in doubt, classify as factual.

  - "chitchat": pure greetings, thanks, identity questions ("who are you"),
    or one-line conversational fillers that contain ZERO domain nouns and
    require no factual answer.

  - "refuse": political opinion requests (which party is better), religious
    opinion requests (which religion is correct), or abusive/threatening
    messages. NOT general policy questions — only opinion-on-controversy.

Respond with ONLY a JSON object: {"intent": "<value>"}. No prose.
"""


async def classify_intent(
    text: str,
    secondary_client: AsyncOpenAI,
    secondary_model: str,
    timeout: float = 4.0,
) -> Intent:
    """Run the classifier and apply the domain-vocab override.

    Returns one of "factual" | "chitchat" | "refuse". On any error
    (secondary LLM down, malformed JSON, etc.) defaults to "factual" — fail
    closed toward grounding, never toward bypassing tool calls.
    """
    text = (text or "").strip()
    if not text:
        return "chitchat"

    # Hard refusal shortcut — no LLM call needed for obvious partisan framings.
    lower = text.lower()
    if any(kw in text or kw.lower() in lower for kw in _REFUSE_KEYWORDS_BN):
        logger.info("Intent: refuse (matched refuse keyword)")
        return "refuse"

    # Default if anything goes wrong with the LLM call.
    verdict: Intent = "factual"

    try:
        import asyncio

        async def _call():
            return await secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=32,
            )

        resp = await asyncio.wait_for(_call(), timeout=timeout)
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        candidate = data.get("intent", "").lower()
        if candidate in ("factual", "chitchat", "refuse"):
            verdict = candidate  # type: ignore[assignment]
        else:
            logger.warning("Classifier returned unknown intent %r; defaulting to factual.", candidate)
    except Exception as e:
        logger.warning("Intent classifier failed (%s); defaulting to factual.", e)

    # Domain-vocab override: if ANY domain noun is in the message and the
    # classifier said anything other than "factual", force "factual". This is
    # the belt-and-braces guard against false-negative classifier errors.
    if verdict != "factual" and _DOMAIN_RE.search(text):
        logger.info("Intent override: %r → factual (domain vocab matched in %r)", verdict, text[:80])
        verdict = "factual"

    logger.info("Intent: %s for %r", verdict, text[:80])
    return verdict
