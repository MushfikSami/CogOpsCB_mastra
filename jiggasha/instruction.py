"""Dynamic retrieval instruction generation for embedding queries.

Uses the secondary LLM to produce a short English task instruction that is
prefixed to the query before embedding.  Improves cosine scores by focusing
the embedding model on the information need.

Guards against common small-model pitfalls:
  - hallucination of foreign entities  (explicit prompt rules + example)
  - empty / generic instructions       (temperature 0.0, max_tokens 128)
  - latency spikes on repeated queries (bounded in-memory cache)
  - LLM failure                        → returns None, caller falls back to raw
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a retrieval instruction writer. Your ONLY job is to write a short English instruction (2-3 lines) that helps an embedding model find relevant passages.

STRICT RULES:
1. Write ONLY the instruction text. NO explanations, NO markdown, NO JSON, NO bullet points.
2. The instruction MUST be in English.
3. Mention the specific document type, service, or topic from the query.
4. ALWAYS include action words like: procedures, steps, requirements, rules, application process, official methods, necessary documents, status check, correction process.
5. NEVER add information not mentioned in the query. NEVER hallucinate countries, organizations, or entities.
6. NEVER mention India, Aadhaar, UIDAI, or foreign contexts unless explicitly in the query.
7. NEVER mention countries other than Bangladesh unless the query explicitly asks about them.
8. NEVER fabricate government portal names, URLs, or contact numbers.
9. If the query is about a specific person, use ONLY the exact name from the query.
10. Keep the instruction SHORT — 2 to 3 sentences max. Every extra word dilutes the embedding signal.
11. Start with "Retrieve passages about..." or "Find passages describing..."
12. End your response immediately after the instruction. Do NOT add "Note:", "Disclaimer:", or any footer.

FORBIDDEN HALLUCINATIONS — NEVER OUTPUT ANY OF THESE UNLESS IN THE QUERY:
- India, Aadhaar, UIDAI, PAN card
- USA, UK, Europe, China, Pakistan
- NREGA, PM-KISAN, GST Portal (India)
- Facebook, WhatsApp, Twitter, Instagram
- Wikipedia, Google, ChatGPT
- Any phone number, email, or URL

EXAMPLE 1:
Query: এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?
Instruction: Retrieve passages about checking the status of a National ID (NID) card. Focus on official procedures, required steps, and relevant government service portals.

EXAMPLE 2:
Query: পাসপোর্ট করার নিয়ম কি?
Instruction: Retrieve passages about passport application procedures, requirements, and necessary steps. Focus on official rules, required documents, and the application process.

EXAMPLE 3:
Query: বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়?
Instruction: Retrieve passages about the start date of the Bangladesh Liberation War. Focus on historical events, specific dates, and the timeline of the conflict.

EXAMPLE 4:
Query: বাংলাদেশের প্রধানমন্ত্রী কে?
Instruction: Retrieve passages stating the name of the current Prime Minister of Bangladesh. Focus on factual information about political leadership.

EXAMPLE 5:
Query: বিবাহ সনদে নাম সংশোধনের নিয়ম?
Instruction: Retrieve passages about name correction in a marriage certificate. Focus on the official correction process, required documents, and application steps.

EXAMPLE 6:
Query: জন্ম সনদ পাওয়ার প্রক্রিয়া?
Instruction: Retrieve passages about obtaining a birth certificate in Bangladesh. Focus on application procedures, required documents, and official rules.

EXAMPLE 7:
Query: মেট্রোরেলের ভাড়া কত?
Instruction: Retrieve passages about Dhaka Metro Rail fares and ticketing rules. Focus on pricing, route information, and official metro service guidelines.

EXAMPLE 8:
Query: চারিত্রিক সনদের জন্য কী কী লাগে?
Instruction: Retrieve passages about character certificate requirements in Bangladesh. Focus on necessary documents, application steps, and official procedures.

Now write the instruction for the given query/queries.
{query_block}

Instruction:"""


def _build_query_block(query: str) -> str:
    """Format the query block for the instruction prompt.

    If *query* is a single query, returns the standard single-query format.
    If *query* contains multiple queries (semicolon-separated or multi-line),
    the block makes it clear that multiple topics are being searched.
    """
    lines = [line.strip() for line in query.split("\n") if line.strip()]
    if len(lines) == 1:
        return f"Query: {lines[0]}"
    out = ["Queries:"]
    for i, line in enumerate(lines, start=1):
        out.append(f"  {i}. {line}")
    return "\n".join(out)

# Simple bounded cache: query → instruction
_instruction_cache: Dict[str, str] = {}
_instruction_cache_lock = asyncio.Lock()
_MAX_CACHE_SIZE = 1024


async def generate_instruction(
    query: str,
    secondary_client: Any,
    secondary_model: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
    timeout: float = 5.0,
) -> Optional[str]:
    """Generate a retrieval instruction for *query*.

    Returns the instruction string, or ``None`` if the LLM call fails or
    produces empty output.  The caller should fall back to the raw query
    when ``None`` is returned.
    """
    if not query or not query.strip():
        return None

    # Check cache (fast path)
    cache_key = query.strip()
    async with _instruction_cache_lock:
        cached = _instruction_cache.get(cache_key)
        if cached is not None:
            return cached

    if secondary_client is None or not secondary_model:
        logger.warning("instruction: no secondary LLM configured")
        return None

    t0 = time.time()
    try:
        async def _call():
            # Build kwargs safely: extra_body is vLLM-specific and may error
            # on non-vLLM backends (OpenAI, Gemini, etc.).
            create_kwargs = {
                "model": secondary_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT.format(query_block=_build_query_block(cache_key))},
                    {"role": "user", "content": _build_query_block(cache_key) + "\n\nInstruction:"},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            except Exception:  # noqa: BLE001
                pass
            resp = await secondary_client.chat.completions.create(**create_kwargs)
            return resp.choices[0].message.content or ""

        raw = await asyncio.wait_for(_call(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("instruction: secondary LLM timed out after %.1fs", timeout)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("instruction: secondary LLM call failed (%s)", e)
        return None

    instruction = raw.strip()
    if not instruction:
        logger.warning("instruction: secondary LLM returned empty text")
        return None

    elapsed = (time.time() - t0) * 1000
    logger.info(
        "instruction: generated for query=%r len=%d elapsed=%.0fms",
        cache_key[:60], len(instruction), elapsed,
    )

    # Store in cache (bounded)
    async with _instruction_cache_lock:
        if len(_instruction_cache) >= _MAX_CACHE_SIZE:
            # Evict oldest (simple: clear half)
            keys = list(_instruction_cache.keys())
            for k in keys[: _MAX_CACHE_SIZE // 2]:
                del _instruction_cache[k]
        _instruction_cache[cache_key] = instruction

    return instruction


def clear_instruction_cache() -> None:
    """Drop all cached instructions.  Useful in tests."""
    _instruction_cache.clear()
