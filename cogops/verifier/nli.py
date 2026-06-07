"""
cogops/verifier/nli.py

Batched NLI (Natural Language Inference) verifier for cited claims.

Input is a list of (tag, sentence) pairs from `cogops.verifier.citations.extract_citations()`
plus the source_map. For each pair we feed the LLM the cited passage and the
claim sentence, and ask whether the passage supports the claim.

ONE call to the secondary LLM per turn — pairs are batched into a single
structured-JSON request. Timeout is hard-capped (default 6s); on failure or
timeout we return all-"entailed" verdicts so the answer is not blocked
(degrade-to-warn policy is then applied upstream in policy.py).
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import AsyncOpenAI

from cogops.prompts.time_reminder import build_time_reminder

logger = logging.getLogger(__name__)

Verdict = Literal["entailed", "partial", "not_entailed"]


def _extract_usage(resp: Any) -> Optional[Dict[str, int]]:
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return None
        prompt = getattr(u, "prompt_tokens", None)
        completion = getattr(u, "completion_tokens", None)
        if prompt is None and completion is None:
            return None
        return {"prompt": int(prompt or 0), "completion": int(completion or 0)}
    except Exception:  # noqa: BLE001
        return None


_SYSTEM = """\
You are a strict factual-entailment verifier for a Bangladesh government-services chatbot.

For each (claim, evidence) pair, decide whether the evidence DIRECTLY SUPPORTS the
factual content of the claim. Be strict — partial support is not full support.

Verdict rules:
  - "entailed":     evidence fully supports every factual element in the claim
  - "partial":      evidence supports SOME elements but contradicts or omits others
  - "not_entailed": evidence does not support the claim, or contradicts it

Numbers, dates, fees, URLs, office names, and procedure steps must match exactly.
A claim that mentions a specific fee/date/number not present in the evidence is
"not_entailed", even if the surrounding context is related.

Output ONLY a JSON object of the form:
  {"verdicts": [{"i": 0, "v": "entailed"}, {"i": 1, "v": "not_entailed"}, ...]}
where "i" is the input index (0-based) and "v" is one of the three verdicts.
Include exactly one verdict per input pair, in order. No prose.
"""


_PAREN_RE = re.compile(r"\s*\([^)]*(?:অর্থাৎ|তার|অর্থা|যেমন|যেমনটি)[^)]*\)\s*")


def _strip_explanatory_parentheticals(text: str) -> str:
    """Remove parentheticals that connect subjects across sources.

    These are reader aids, not factual claims. Stripping them lets NLI
    focus on the core factual assertion that the cited source must support.
    """
    return _PAREN_RE.sub(" ", text).strip()


def _build_user_prompt(pairs: List[Tuple[str, str]], source_map: Dict[str, Dict[str, Any]]) -> str:
    """Render the batched (claim, evidence) pairs as a single user message."""
    blocks: List[str] = []
    for i, (tag, sentence) in enumerate(pairs):
        evidence = source_map.get(tag, {}).get("text", "") or "(evidence missing)"
        core_claim = _strip_explanatory_parentheticals(sentence)
        blocks.append(
            f"### Pair {i}\n"
            f"Claim (cites [{tag}]): {core_claim}\n"
            f"Evidence [{tag}]: {evidence}\n"
        )
    return (
        "Verify each claim against its cited evidence.\n\n"
        + "\n".join(blocks)
        + "\nReturn a single JSON object as specified."
    )


async def verify_claims(
    pairs: List[Tuple[str, str]],
    source_map: Dict[str, Dict[str, Any]],
    secondary_client: AsyncOpenAI,
    secondary_model: str,
    timeout: float = 6.0,
) -> Tuple[List[Verdict], Optional[Dict[str, int]]]:
    """Verify each (tag, sentence) pair against its source.

    Returns (verdicts, usage):
      - `verdicts` is aligned to `pairs`. On timeout/error, returns all
        "entailed" — the answer is not blocked.
      - `usage` is {"prompt": int, "completion": int} from the LLM, or None
        if no LLM call was made (empty input or all fast-path) or the call
        failed before returning a usable response.
    """
    if not pairs:
        return [], None

    # Fast-path: if any cited tag is not in source_map at all, mark that pair
    # as not_entailed immediately (no LLM call needed for the obviously bogus
    # ones; the upstream tag-stripper should have removed them already, but
    # belt-and-braces).
    fastpath: Dict[int, Verdict] = {}
    valid_indices: List[int] = []
    valid_pairs: List[Tuple[str, str]] = []
    for i, (tag, sentence) in enumerate(pairs):
        if tag not in source_map:
            fastpath[i] = "not_entailed"
        else:
            valid_indices.append(i)
            valid_pairs.append((tag, sentence))

    if not valid_pairs:
        return [fastpath[i] for i in range(len(pairs))], None

    verdicts_by_idx: Dict[int, Verdict] = dict(fastpath)
    usage: Optional[Dict[str, int]] = None

    try:
        async def _call():
            return await secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "assistant", "content": build_time_reminder()},
                    {"role": "user", "content": _build_user_prompt(valid_pairs, source_map)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )

        resp = await asyncio.wait_for(_call(), timeout=timeout)
        raw = (resp.choices[0].message.content or "").strip()
        usage = _extract_usage(resp)
        data = json.loads(raw)
        items = data.get("verdicts", []) or []

        for item in items:
            i_local = item.get("i")
            v = item.get("v", "").lower()
            if not isinstance(i_local, int) or i_local < 0 or i_local >= len(valid_pairs):
                continue
            if v not in ("entailed", "partial", "not_entailed"):
                continue
            verdicts_by_idx[valid_indices[i_local]] = v  # type: ignore[assignment]
    except Exception as e:
        logger.warning("NLI verifier failed (%s); degrading to all-entailed.", e)
        for i in valid_indices:
            verdicts_by_idx.setdefault(i, "entailed")

    # Fill any gap with "entailed" (LLM might have omitted some indices).
    return [verdicts_by_idx.get(i, "entailed") for i in range(len(pairs))], usage
