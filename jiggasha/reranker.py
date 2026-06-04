"""LLM cross-encoder reranker that outputs a discrete 0–10 relevance score.

Designed for vLLM prefix caching:
- The system prompt is long and identical for every call.
- Only the user message (query + passage) changes.
- Parallel evaluation via asyncio.gather.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RERANK_SYSTEM_PROMPT = """You are a passage relevance scorer for a Bengali document retrieval system. Your ONLY job is to output a single integer from 0 to 10 representing how relevant the passage is to the user's query.

SCORING GUIDE:
- 10: Perfect match — the passage directly and completely answers the query.
- 7–9: Highly relevant — contains key information needed for the answer.
- 4–6: Somewhat relevant — partial, tangential, or indirectly related information.
- 1–3: Barely relevant — mostly unrelated, only a passing mention.
- 0: Completely irrelevant — nothing to do with the query.

STRICT RULES:
1. Output EXACTLY one integer between 0 and 10. No words, no punctuation, no markdown.
2. BREADCRUMB CHECK: Every passage starts with a breadcrumb like "Category > Sub-category > Service". If the breadcrumb shows the passage is about a DIFFERENT specific topic than the query, score it LOW (0–3) even if the body text mentions the query topic incidentally.
3. NO BACKGROUND ONLY: If the passage only provides background context for an unrelated main subject, score it LOW.
4. NO PASSING MENTIONS: A single passing mention of the query topic does NOT make a passage relevant.
5. BE CONSERVATIVE: When in doubt, score LOWER. It is better to miss a marginally relevant passage than to include an off-topic one.

EXAMPLES:

Query: বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়?
Passage: বাংলাদেশের স্বাধীনতা যুদ্ধের কালপঞ্জি > বাংলাদেশের স্বাধীনতা যুদ্ধের কালপঞ্জি
বাংলাদেশের স্বাধীনতা যুদ্ধ শুরু হয় মার্চ ২৬, ১৯৭১ এবং শেষ হয় ডিসেম্বর ১৬, ১৯৭১।
Score: 10

Query: বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়?
Passage: বাংলাদেশ ফিল্ড হাসপাতাল > বাংলাদেশ ফিল্ড হাসপাতাল > পটভূমি
বাংলাদেশের প্রায় নয় মাসব্যাপী মুক্তিযুদ্ধ শুরু হয়েছিল ১৯৭১ সালের ২৫ মার্চ...
Score: 2

Query: এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?
Passage: স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র > স্মার্ট কার্ড > নতুন এনআইডি কার্ড চেক করার ধাপ
নতুন এনআইডি কার্ড বা স্মার্ট কার্ড চেক করার জন্য নিম্নোক্ত ধাপগুলো অনুসরণ করতে হবে...
Score: 9

Now score the following passage."""


def _parse_score(raw: str) -> Optional[int]:
    """Extract the first integer 0–10 from the model output."""
    if not raw:
        return None
    m = re.search(r"\b(\d{1,2})\b", raw.strip())
    if not m:
        return None
    val = int(m.group(1))
    if 0 <= val <= 10:
        return val
    return None


async def rerank_passages(
    query: str,
    passages: List[Dict[str, Any]],
    secondary_client: Any,
    secondary_model: str,
    max_passage_chars: int = 1500,
) -> tuple[List[Dict[str, Any]], int, int]:
    """Rerank passages using the secondary LLM as a cross-encoder.

    The LLM outputs a single integer 0–10.  We normalize it to 0.0–1.0.
    Parallel calls via asyncio.gather.

    Returns:
        (reranked_passages, total_prompt_tokens, total_completion_tokens)
    """
    if not passages or secondary_client is None or not secondary_model:
        return passages, 0, 0

    async def _score_one(p: Dict[str, Any]) -> Dict[str, Any]:
        text = p.get("text", "")[:max_passage_chars]
        user_msg = f"Query: {query}\n\nPassage breadcrumb and text:\n{text}\n\nScore:"

        try:
            resp = await secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=3,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("rerank: LLM call failed for passage_id=%s: %s", p.get("passage_id"), e)
            return {**p, "rerank_score": p.get("score", 0.0)}

        raw = (resp.choices[0].message.content or "").strip()
        score_int = _parse_score(raw)

        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        compl_tok = getattr(usage, "completion_tokens", 0) if usage else 0

        if score_int is None:
            logger.warning(
                "rerank: could not parse score %r for passage_id=%s; using cosine fallback",
                raw, p.get("passage_id"),
            )
            return {
                **p,
                "rerank_score": p.get("score", 0.0),
                "_rerank_prompt_tokens": prompt_tok,
                "_rerank_completion_tokens": compl_tok,
            }

        return {
            **p,
            "rerank_score": round(score_int / 10.0, 4),
            "_rerank_prompt_tokens": prompt_tok,
            "_rerank_completion_tokens": compl_tok,
        }

    scored = await asyncio.gather(*[_score_one(p) for p in passages], return_exceptions=True)

    results: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, item in enumerate(scored):
        if isinstance(item, Exception):
            logger.warning("rerank: exception for passage_id=%s: %s", passages[i].get("passage_id"), item)
            p = passages[i]
            results.append({**p, "rerank_score": p.get("score", 0.0)})
            continue
        results.append(item)
        total_prompt_tokens += item.pop("_rerank_prompt_tokens", 0)
        total_completion_tokens += item.pop("_rerank_completion_tokens", 0)

    # Sort by rerank_score descending
    results.sort(key=lambda x: -x.get("rerank_score", 0.0))
    return results, total_prompt_tokens, total_completion_tokens
