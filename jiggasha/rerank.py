"""
jiggasha/rerank.py — LLM relevance reranker, embedded in the search service.

One batched secondary-LLM call judges which candidate passages are relevant
per sub-query. Output is the compact per-query class-index format:

    {"1": [[passage_id, class], ...], "2": [...], ...}

where class is `0` (yes — directly answers) or `1` (weak — tangential,
backfill only). Passages omitted from the LLM output are treated as "no".

Policy applied to the LLM verdicts:
  - All "yes" passages are kept.
  - For each sub-question lacking "yes" coverage, up to `weak_per_sub_cap`
    "weak" passages are backfilled.
  - Kept set is capped at `keep_cap` and sorted by (verdict priority, cosine).

Failure modes (timeout, malformed JSON, no client) fall back to a cosine
safety net — keep candidates with `score >= fallback_cosine_min`, tag them
"weak", and mark the result `degraded=True`.

Concurrency: an asyncio.Semaphore bounds the number of secondary-LLM calls
made simultaneously. Beyond the cap, callers queue (waiting in the same
process) instead of being rejected by the upstream LLM with 503. This
absorbs e2e-test bursts (8+ simultaneous users) without requiring a
separate task queue.

The module is async (call from FastAPI async handlers).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CLASS_YES = 0
CLASS_WEAK = 1


# ----- Concurrency control --------------------------------------------------
# Single per-process semaphore guarding the secondary-LLM call. Default is
# conservative (4) — the qwen36 endpoint at :5000 also serves the chatbot's
# router + NLI verifier, so we want headroom. Override with env var
# JIGGASHA_RERANK_MAX_CONCURRENCY.
_DEFAULT_MAX_CONCURRENCY = int(
    os.environ.get("JIGGASHA_RERANK_MAX_CONCURRENCY", "4")
)
_RERANK_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init the semaphore in the running event loop (FastAPI's loop)."""
    global _RERANK_SEMAPHORE
    if _RERANK_SEMAPHORE is None:
        _RERANK_SEMAPHORE = asyncio.Semaphore(_DEFAULT_MAX_CONCURRENCY)
        logger.info(
            "rerank: concurrency cap = %d (set JIGGASHA_RERANK_MAX_CONCURRENCY to override)",
            _DEFAULT_MAX_CONCURRENCY,
        )
    return _RERANK_SEMAPHORE


@dataclass
class RerankCandidate:
    """One Qdrant hit, merged across sub-queries."""
    passage_id: int
    text: str
    score: float
    category: str = ""
    sub_category: str = ""
    service: str = ""
    topic: str = ""
    sub_indices: List[int] = field(default_factory=list)


@dataclass
class RerankResult:
    """Output of run_rerank.

    Attributes:
        passages:  Kept passages, sorted by verdict (yes first) then cosine.
        per_query: 1-based string key -> list of [passage_id, class] tuples.
                   Every passage_id appearing here also appears in `passages`.
        degraded:  True if the LLM call failed and the cosine safety net ran.
        usage:     Token usage from the rerank LLM call, or None on safety-net.
                   Shape: {"prompt": int, "completion": int}.
    """
    passages: List[RerankCandidate]
    per_query: Dict[str, List[List[int]]]
    degraded: bool = False
    usage: Optional[Dict[str, int]] = None


def _extract_usage(resp: Any) -> Optional[Dict[str, int]]:
    """Pull {prompt, completion} from an OpenAI-compatible response."""
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


_SYSTEM_PROMPT = """\
You are a relevance filter for a Bangladesh government-services chatbot.

INPUT:
  - SUB-QUESTIONS: a numbered list (1-based) of user questions.
  - CANDIDATE PASSAGES: each tagged with passage_id and a metadata header.

For each SUB-QUESTION, identify the user's SPECIFIC SUBJECT — the exact
action / document / situation being asked about, NOT the general domain.
Examples of specific subject:

  • "পার্কিং মামলা তুলবো কীভাবে?"        → subject = case WITHDRAWAL
                                            (NOT case filing, NOT FIR)
  • "এসএসসি সনদে বোর্ড পরিবর্তন কীভাবে?" → subject = BOARD change
                                            (NOT name correction, NOT age)
  • "ডবল বিল ফেরত পাব কীভাবে?"           → subject = refund of overpayment
                                            (NOT how to pay a bill)
  • "এনআইডিতে ড. যোগ করব?"               → subject = adding titles to NID
                                            (NOT general NID correction)

Then classify each passage:

  - class 0 ("yes")  → the passage NAMES the user's SPECIFIC SUBJECT and
                       gives facts about IT (procedure, fee, eligibility,
                       or explicitly says it is not allowed/possible).
                       Negative answers about the SAME subject ARE yes.
  - class 1 ("weak") → the passage is in the same general domain but
                       covers a DIFFERENT specific subject (case-filing
                       passages for a case-withdrawal question; name
                       correction passages for a board-change question).
                       Useful background; NOT a direct answer.
  - DROP unrelated passages by omitting them (treated as "no").

Be strict on "yes". If you cannot find a sentence in the passage that
mentions the user's exact subject by name, the passage is at most "weak".
"Same domain" is not enough — there must be subject overlap. When in
doubt between yes and weak, pick weak.

OUTPUT — strict compact JSON, no whitespace, no markdown, no prose:
{"<sub_num>":[[<passage_id>,<class>], ...], ...}

Example for 2 sub-questions:
{"1":[[5,0],[12,1]],"2":[[5,0]]}

Use the actual passage_id (NOT array index). Sub_num is the 1-based index
of the sub-question (as a STRING key). Class is exactly 0 or 1. A passage
that helps multiple sub-questions appears in each of their lists.
"""


def _build_user_prompt(
    sub_queries: List[str],
    candidates: List[RerankCandidate],
) -> str:
    parts: List[str] = ["SUB-QUESTIONS:"]
    for idx, q in enumerate(sub_queries, start=1):
        parts.append(f"  [{idx}] {q}")
    parts.append("")
    parts.append("CANDIDATE PASSAGES:")
    for c in candidates:
        hb: List[str] = []
        if c.category:
            hb.append(f"বিভাগ: {c.category}")
        if c.sub_category:
            hb.append(f"উপ-বিভাগ: {c.sub_category}")
        if c.service:
            hb.append(f"সেবা: {c.service}")
        if c.topic:
            hb.append(f"বিষয়: {c.topic}")
        header = " | ".join(hb) if hb else "—"
        parts.append("")
        parts.append(f"=== passage_id={c.passage_id} ({header}) ===")
        parts.append(c.text)
    parts.append("")
    parts.append('Output compact JSON: {"<sub_num>":[[<passage_id>,<class>], ...]}')
    return "\n".join(parts)


def _parse_per_query(
    raw: str,
    known_pids: set[int],
    n_subs: int,
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """Parse the LLM JSON. Returns {sub_idx (0-based): [(pid, cls), ...]} or None."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    out: Dict[int, List[Tuple[int, int]]] = {}
    for key, entries in data.items():
        try:
            sub_num = int(key)
        except (TypeError, ValueError):
            continue
        sub_idx = sub_num - 1
        if not (0 <= sub_idx < n_subs):
            continue
        if not isinstance(entries, list):
            continue
        seen_pids: set[int] = set()
        bucket: List[Tuple[int, int]] = []
        for entry in entries:
            pid: Optional[int] = None
            cls: Optional[int] = None
            if isinstance(entry, list) and len(entry) >= 2:
                try:
                    pid = int(entry[0])
                    cls = int(entry[1])
                except (TypeError, ValueError):
                    continue
            elif isinstance(entry, dict):
                try:
                    pid_v = (
                        entry.get("id")
                        if entry.get("id") is not None
                        else entry.get("passage_id")
                        if entry.get("passage_id") is not None
                        else entry.get("i")
                    )
                    pid = int(pid_v) if pid_v is not None else None
                    cls_v = (
                        entry.get("class")
                        if entry.get("class") is not None
                        else entry.get("c")
                        if entry.get("c") is not None
                        else entry.get("v")
                    )
                    if isinstance(cls_v, str):
                        cls = CLASS_YES if cls_v.lower() == "yes" else (
                            CLASS_WEAK if cls_v.lower() == "weak" else None
                        )
                    elif cls_v is not None:
                        cls = int(cls_v)
                except (TypeError, ValueError):
                    continue
            else:
                continue
            if pid is None or pid not in known_pids:
                continue
            if cls not in (CLASS_YES, CLASS_WEAK):
                continue
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            bucket.append((pid, cls))
        out[sub_idx] = bucket
    return out


def _apply_policy(
    candidates: List[RerankCandidate],
    per_query: Dict[int, List[Tuple[int, int]]],
    keep_cap: int,
    weak_per_sub_cap: int,
    n_subs: int,
) -> Tuple[List[RerankCandidate], Dict[str, List[List[int]]]]:
    """Apply yes-core + per-sub-capped weak inclusion + global cap.

    Yes passages are always kept (subject to the global keep_cap). Weak
    passages are admitted up to weak_per_sub_cap per sub — regardless of
    whether that sub already has yes coverage. This preserves enough
    same-domain candidates for the chatbot to disambiguate short/generic
    queries without losing the yes-first ordering that focused queries rely
    on.

    Returns:
        kept passages (yes first by cosine desc, then weak by cosine desc, capped)
        per_query dict (1-based STRING keys), filtered to surviving pids only.
    """
    by_pid = {c.passage_id: c for c in candidates}

    yes_subs_by_pid: Dict[int, List[int]] = {}
    weak_subs_by_pid: Dict[int, List[int]] = {}
    for sub_idx, entries in per_query.items():
        for pid, cls in entries:
            target = yes_subs_by_pid if cls == CLASS_YES else weak_subs_by_pid
            if pid not in target:
                target[pid] = []
            if sub_idx not in target[pid]:
                target[pid].append(sub_idx)

    # Pass 1: all yes (cosine desc).
    yes_pids = sorted(
        (p for p in yes_subs_by_pid if p in by_pid),
        key=lambda p: -by_pid[p].score,
    )
    kept_pids: List[int] = list(yes_pids)

    # Pass 2: include weak passages up to weak_per_sub_cap per sub. Yes
    # passages do NOT consume the weak budget. This lets generic queries
    # (where everything is "weak") still surface candidates spanning
    # several services for disambiguation.
    weak_used_per_sub: Dict[int, int] = {}
    weak_pids = sorted(
        (p for p in weak_subs_by_pid if p in by_pid and p not in yes_subs_by_pid),
        key=lambda p: -by_pid[p].score,
    )
    for pid in weak_pids:
        subs = list(weak_subs_by_pid[pid])
        if not subs:
            continue
        if not any(
            weak_used_per_sub.get(s, 0) < weak_per_sub_cap for s in subs
        ):
            continue
        kept_pids.append(pid)
        for s in subs:
            weak_used_per_sub[s] = weak_used_per_sub.get(s, 0) + 1

    kept_pids = kept_pids[:keep_cap]
    survived = set(kept_pids)

    out_per_query: Dict[str, List[List[int]]] = {}
    for sub_idx in range(n_subs):
        entries = per_query.get(sub_idx, [])
        bucket: List[List[int]] = []
        for pid, cls in entries:
            if pid in survived:
                bucket.append([pid, cls])
        out_per_query[str(sub_idx + 1)] = bucket

    return [by_pid[p] for p in kept_pids], out_per_query


def _cosine_safety_net(
    candidates: List[RerankCandidate],
    n_subs: int,
    fallback_cosine_min: float,
    keep_cap: int,
) -> Tuple[List[RerankCandidate], Dict[str, List[List[int]]]]:
    """Fallback when the LLM call fails. Tags everything `weak`."""
    pool = [c for c in candidates if c.score >= fallback_cosine_min]
    pool.sort(key=lambda c: -c.score)
    kept = pool[:keep_cap]

    per_query: Dict[str, List[List[int]]] = {}
    for sub_idx in range(n_subs):
        bucket: List[List[int]] = []
        for c in kept:
            if sub_idx in c.sub_indices:
                bucket.append([c.passage_id, CLASS_WEAK])
        per_query[str(sub_idx + 1)] = bucket
    return kept, per_query


async def run_rerank(
    sub_queries: List[str],
    candidates: List[RerankCandidate],
    secondary_client: Any,
    secondary_model: str,
    timeout: float = 30.0,
    keep_cap: int = 24,
    weak_per_sub_cap: int = 3,
    fallback_cosine_min: float = 0.50,
) -> RerankResult:
    """Filter and rank candidate passages by LLM-judged per-query relevance.

    Args:
        sub_queries: 1..N user sub-questions (normalized Bengali).
        candidates: merged, deduped Qdrant hits across all sub-queries.
        secondary_client: openai.AsyncOpenAI (or None → cosine safety net).
        secondary_model: secondary LLM model name.
        timeout: hard timeout in seconds for the LLM call.
        keep_cap: max passages returned to the chatbot.
        weak_per_sub_cap: max "weak" backfill per uncovered sub-question.
        fallback_cosine_min: cosine threshold for the safety net.

    Returns:
        RerankResult — never raises.
    """
    n_subs = len(sub_queries)
    if not candidates or n_subs == 0:
        return RerankResult(passages=[], per_query={}, degraded=False)

    if secondary_client is None:
        kept, per_q = _cosine_safety_net(
            candidates, n_subs, fallback_cosine_min, keep_cap,
        )
        return RerankResult(passages=kept, per_query=per_q, degraded=True)

    user_prompt = _build_user_prompt(sub_queries, candidates)
    max_tokens = max(400, len(candidates) * 30 + 200)

    try:
        async def _call():
            async with _get_semaphore():
                return await secondary_client.chat.completions.create(
                    model=secondary_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

        resp = await asyncio.wait_for(_call(), timeout=timeout)
        raw = (resp.choices[0].message.content or "").strip()
        usage = _extract_usage(resp)
    except asyncio.TimeoutError:
        logger.warning("rerank LLM timed out after %.1fs — cosine safety net", timeout)
        kept, per_q = _cosine_safety_net(candidates, n_subs, fallback_cosine_min, keep_cap)
        return RerankResult(passages=kept, per_query=per_q, degraded=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("rerank LLM call failed (%s) — cosine safety net", e)
        kept, per_q = _cosine_safety_net(candidates, n_subs, fallback_cosine_min, keep_cap)
        return RerankResult(passages=kept, per_query=per_q, degraded=True)

    parsed = _parse_per_query(
        raw, known_pids={c.passage_id for c in candidates}, n_subs=n_subs,
    )
    if parsed is None:
        logger.warning("rerank LLM returned malformed JSON — cosine safety net")
        kept, per_q = _cosine_safety_net(candidates, n_subs, fallback_cosine_min, keep_cap)
        # LLM was called and consumed tokens even though parsing failed;
        # surface that so callers can attribute the cost.
        return RerankResult(passages=kept, per_query=per_q, degraded=True, usage=usage)

    kept, per_q = _apply_policy(
        candidates=candidates,
        per_query=parsed,
        keep_cap=keep_cap,
        weak_per_sub_cap=weak_per_sub_cap,
        n_subs=n_subs,
    )

    yes_total = sum(
        1 for entries in parsed.values() for _, cls in entries if cls == CLASS_YES
    )
    weak_total = sum(
        1 for entries in parsed.values() for _, cls in entries if cls == CLASS_WEAK
    )
    logger.info(
        "rerank: n_subs=%d candidates=%d kept=%d (yes_votes=%d weak_votes=%d) usage=%s",
        n_subs, len(candidates), len(kept), yes_total, weak_total, usage,
    )
    return RerankResult(passages=kept, per_query=per_q, degraded=False, usage=usage)
