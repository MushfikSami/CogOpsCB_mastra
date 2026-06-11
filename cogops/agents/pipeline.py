"""
cogops/agents/pipeline.py

Deterministic factual-query pipeline. Stages:

    Stage 0  sanitize                  (pure code, no LLM)             — caller
    Stage 1  router                     (1 secondary-LLM call)          — caller
    Stage 2  Jiggasha retrieval         (1 HTTP POST, instruction-based) — here
    Stage 3  compose                    (1 primary-LLM streaming call)   — here
    Stage 4  post-flight                (NLI verify + Sources block)     — here

Stage 2 POSTs sub-queries to Jiggasha, which prefixes a dynamic English
instruction, embeds the query, fetches top-K from Qdrant, and filters by
cosine threshold + token budget.  Jiggasha returns:
    {
      sub_queries: [...],
      passages: [{passage_id, text, category, ..., score}, ...],
      instruction: "...",
      elapsed_ms: 123
    }

This module is intentionally framework-light: depends only on
  - cogops.pipeline.router (RouterResult)
  - cogops.prompts.composer / time_reminder
  - cogops.verifier.citations / nli / policy
  - cogops.utils.thinking_parser
  - openai.AsyncOpenAI + httpx
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from cogops.pipeline.normalize import normalize_sub_queries
from cogops.pipeline.query_expand import check_document_type_match
from cogops.pipeline.router import RouterResult
from cogops.prompts.composer import get_composer_prompt
from cogops.prompts.time_reminder import build_time_reminder
from cogops.utils.thinking_parser import ThinkingParser
from cogops.verifier.citations import (
    build_sources_block,
    extract_citation_tags,
    extract_citations,
    strip_unknown_tags,
)
from cogops.verifier.nli import verify_claims
from cogops.verifier.policy import apply_policy

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================

@dataclass
class PipelineConfig:
    """Tunables for the factual pipeline."""

    # Jiggasha retrieval
    jiggasha_endpoint: str = "http://localhost:10000/search"
    jiggasha_timeout: float = 45.0
    top_k_fetch: int = 50
    chunk_type: Optional[str] = None   # DEPRECATED: no longer sent to Jiggasha

    # Instruction-based retrieval
    use_instruction: bool = True
    cosine_threshold: Optional[float] = None
    token_budget: Optional[int] = None   # forwarded to Jiggasha; enforced there
    rerank_threshold: Optional[float] = 0.50   # min LLM rerank score (0–1)

    # Composer
    composer_temperature: float = 0.1
    composer_top_p: float = 0.95
    composer_max_tokens: int = 2048
    agent_name: str = "GovOps সহকারী"

    # NLI verifier (Stage 4)
    verifier_enabled: bool = True
    verifier_timeout: float = 6.0
    verifier_policy: str = "redact"   # redact | refuse | warn

    # Disambiguation
    disambig_min_distinct_services: int = 2
    disambig_short_query_token_cap: int = 8
    disambig_candidate_cap: int = 6   # max candidates shown to the user

    # ReAct retrieval loop
    max_react_iterations: int = 0   # 0 = disabled; 1-2 = judge + refine

    # Refusals
    refusal_text_bn: str = (
        "দুঃখিত, এই প্রশ্নের জন্য নির্ভরযোগ্য সরকারি তথ্য পাওয়া যায়নি।"
    )


# ============================================================
# Event helpers
# ============================================================

def _evt(type_: str, channel: str = "debug", **payload: Any) -> Dict[str, Any]:
    return {"type": type_, "channel": channel, **payload}


# ============================================================
# Stage 2 — Single Jiggasha call (instruction-based retrieval)
# ============================================================

async def _call_jiggasha(
    http: httpx.AsyncClient,
    endpoint: str,
    query: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """POST one /search to Jiggasha (single-query).

    Retries on transient 5xx / network errors. Two retries with exponential
    backoff are enough to mask brief overload signals. Raises on the final
    failure.
    """
    payload = {
        "query": query,
        "top_k": cfg.top_k_fetch,
        "use_instruction": cfg.use_instruction,
        "cosine_threshold": cfg.cosine_threshold,
        "token_budget": cfg.token_budget,
        "rerank_threshold": cfg.rerank_threshold,
    }
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await http.post(endpoint, json=payload)
            if resp.status_code >= 500 and attempt < 2:
                # Retry on 5xx — likely a transient overload signal.
                await asyncio.sleep(0.25 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < 2:
                await asyncio.sleep(0.25 * (attempt + 1))
                continue
            raise
    # Should not reach here; defensive.
    if last_exc:
        raise last_exc
    raise RuntimeError("jiggasha call failed without exception")


async def _call_jiggasha_multi(
    http: httpx.AsyncClient,
    endpoint: str,
    queries: List[str],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Call Jiggasha once per query in parallel, then merge & deduplicate.

    Returns a unified dict with the same shape as a single Jiggasha response
    so downstream code doesn't need to change:
        {
            "queries": [...],
            "results": [merged_passages],
            "hits_total": N,
            "instructions": [...],
            "elapsed_ms": <wall_clock>,
            "timing_ms": {...},
            "token_usage": {...},
        }
    """
    if not queries:
        return {
            "queries": [],
            "results": [],
            "hits_total": 0,
            "instructions": [],
            "elapsed_ms": 0,
            "timing_ms": {},
            "token_usage": {},
        }

    coros = [_call_jiggasha(http, endpoint, q, cfg) for q in queries]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    merged_passages: Dict[int, Dict[str, Any]] = {}
    all_instructions: List[Optional[str]] = []
    max_elapsed = 0
    timing_agg: Dict[str, List[int]] = {"instruction": [], "embedding": [], "qdrant": [], "rerank": []}
    token_agg: Dict[str, int] = {
        "instruction_prompt": 0,
        "instruction_completion": 0,
        "rerank_prompt": 0,
        "rerank_completion": 0,
    }
    errors: List[str] = []
    first_exc: Optional[Exception] = None

    for res in raw_results:
        if isinstance(res, Exception):
            errors.append(str(res))
            if first_exc is None:
                first_exc = res
            continue

        # Response timing
        max_elapsed = max(max_elapsed, res.get("elapsed_ms", 0) or 0)
        tm = res.get("timing_ms") or {}
        for k in timing_agg:
            v = tm.get(k)
            if isinstance(v, (int, float)):
                timing_agg[k].append(int(v))

        # Token usage
        tu = res.get("token_usage") or {}
        for k in token_agg:
            v = tu.get(k)
            if isinstance(v, (int, float)):
                token_agg[k] += int(v)

        # Instruction
        all_instructions.append(res.get("instruction"))

        # Passages — deduplicate by passage_id, keep highest rerank_score
        for p in (res.get("results") or []):
            pid = p.get("passage_id")
            if pid is None:
                continue
            existing = merged_passages.get(pid)
            if existing is None:
                merged_passages[pid] = dict(p)
            else:
                # Keep the one with higher rerank_score, or higher cosine score
                new_score = p.get("rerank_score")
                old_score = existing.get("rerank_score")
                if new_score is not None and old_score is not None:
                    if new_score > old_score:
                        merged_passages[pid] = dict(p)
                elif new_score is not None:
                    merged_passages[pid] = dict(p)
                elif p.get("score", 0.0) > existing.get("score", 0.0):
                    merged_passages[pid] = dict(p)

    # If every call failed, raise the first exception so upstream can emit
    # a jiggasha_failed refusal rather than a no_passages refusal.
    if not merged_passages and first_exc is not None:
        raise first_exc

    # Sort merged passages: rerank_score desc, then cosine score desc
    def _sort_key(p: Dict[str, Any]) -> Tuple[float, float]:
        rs = p.get("rerank_score")
        return (
            -(rs if rs is not None else 0.0),
            -(p.get("score", 0.0)),
        )

    sorted_passages = sorted(merged_passages.values(), key=_sort_key)

    # Aggregate timing: max per category (wall-clock for parallel calls)
    timing_merged = {k: max(v) if v else 0 for k, v in timing_agg.items()}

    return {
        "queries": queries,
        "results": sorted_passages,
        "hits_total": len(sorted_passages),
        "instructions": [i for i in all_instructions if i],
        "elapsed_ms": max_elapsed,
        "timing_ms": timing_merged,
        "token_usage": token_agg,
        "errors": errors if errors else None,
    }


# ============================================================
# RetrievalJudge — ReAct loop helper
# ============================================================

_JUDGE_SYSTEM_PROMPT = """\
You are a retrieval judge for a Bangladesh government-services chatbot.

You will be given a user query and up to 5 retrieved passages. Decide whether
the passages are SUFFICIENT to directly answer the query.

"Sufficient" means at least one passage explicitly covers the user's exact
subject (procedure, fee, eligibility, contact, office, etc.), not just a
topically related area.

Output ONLY a JSON object with this exact shape:
  {"sufficient": true}
or
  {"sufficient": false, "refined_query": "<improved formal Bengali query>"}

If insufficient, provide a refined_query that is more specific and formal.
Keep the refined query concise (under 20 words)."""


async def _judge_retrieval_sufficiency(
    query: str,
    passages: List[Dict[str, Any]],
    secondary_client: AsyncOpenAI,
    secondary_model: str,
    timeout: float = 5.0,
) -> Tuple[bool, Optional[str]]:
    """Judge whether retrieved passages are sufficient to answer the query.

    Returns (is_sufficient, refined_query_or_none).
    On any failure, returns (True, None) — fail-open so retrieval never blocks.
    """
    if not passages:
        return False, query

    # Summarise top-5 passages for the judge (token budget friendly).
    summary_lines: List[str] = []
    for i, p in enumerate(passages[:5], start=1):
        text = (p.get("text") or "")[:300]
        summary_lines.append(f"[{i}] {text}")
    passage_block = "\n\n".join(summary_lines)

    user_msg = (
        f"Query: {query}\n\n"
        f"Retrieved passages:\n{passage_block}\n\n"
        "Is this sufficient to answer the query directly?"
    )

    try:
        resp = await asyncio.wait_for(
            secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=128,
            ),
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        sufficient = bool(data.get("sufficient", False))
        refined = data.get("refined_query")
        if isinstance(refined, str):
            refined = refined.strip()
            if not refined:
                refined = None
        else:
            refined = None
        return sufficient, refined
    except Exception as e:  # noqa: BLE001
        logger.warning("RetrievalJudge failed (%s); assuming sufficient.", e)
        return True, None


def _build_source_map_from_passages(
    passages: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build the `[S#] → passage meta` map from Jiggasha's passages.

    All passages returned by Jiggasha have already passed cosine threshold
    and token budget filters, so every passage is tagged `verdict: "yes"`.
    """
    source_map: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(passages, start=1):
        try:
            pid = int(p.get("passage_id", 0))
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        tag = f"S{i}"
        source_map[tag] = {
            "passage_id": pid,
            "text": p.get("text", ""),
            "category": p.get("category", "") or "",
            "sub_category": p.get("sub_category", "") or "",
            "service": p.get("service", "") or "",
            "topic": p.get("topic", "") or "",
            "chunk_type": p.get("chunk_type", "") or "",
            "score": float(p.get("score", 0.0)),
            "verdict": "yes",
            "tool": "jiggasha",
        }
    return source_map


# ============================================================
# Disambiguation
# ============================================================

_WORD_SPLIT = re.compile(r"\s+")


def _token_count(text: str) -> int:
    if not text:
        return 0
    cleaned = re.sub(r"[?।!,.;:()\[\]\"'`]+", " ", text).strip()
    if not cleaned:
        return 0
    return sum(1 for t in _WORD_SPLIT.split(cleaned) if t)


def _intent_is_short(
    sub_queries: List[str],
    raw_query: str,
    token_cap: int,
) -> bool:
    """Decide whether the question's INTENT is short — independent of how
    verbose the raw user text is.

    The router normalizes a verbose query into a focused Bengali sub-question;
    its length is a much better proxy for intent specificity than raw text.
    Falls back to the raw query when no normalized sub-query is available.

    For multi-sub queries the user has already pre-split the work; we don't
    treat any of them as "short" for disambiguation purposes.
    """
    if len(sub_queries) > 1:
        return False
    if sub_queries:
        return _token_count(sub_queries[0]) <= token_cap
    return _token_count(raw_query) <= token_cap


def _detect_disambiguation(
    source_map: Dict[str, Dict[str, Any]],
    raw_query: str,
    sub_queries: List[str],
    cfg: PipelineConfig,
    intent: str = "factual_govt",
) -> Tuple[bool, List[Tuple[str, str]]]:
    """Decide whether the answer should ask for clarification.

    Skip disambiguation for factual_wiki / factual_mixed intents:
    wiki corpus naturally spans many distinct (category, sub_category)
    pairs, so disambiguation is not meaningful for encyclopedia queries.

    Triggers when:
      - the question has a SINGLE sub-question (multi-sub queries are already
        split by the user; per-sub disambiguation doesn't apply);
      - the normalized intent (sub_queries[0], or raw_query as fallback) is
        short — gauged in tokens, not characters; and
      - ≥ cfg.disambig_min_distinct_services distinct (category, sub_category)
        tuples appear among kept passages.

    The distinct key is (category, sub_category) and deliberately NOT
    `service`: the corpus's `service` field encodes per-aspect labels
    (e.g. "চারিত্রিক সনদ: ভাষা" vs "চারিত্রিক সনদ: সেবারমূল্য") which all
    describe the same actual service.  For the "same sub_category, different
    boards" case (e.g. SSC name correction across Dhaka/Chittagong boards),
    the corpus already encodes each board in its own sub_category, so
    (cat, sub_cat) is sufficient.

    Returns (disambiguate, candidates) where candidates is a list of
    (tag, human-readable service name) — one per distinct (cat, sub_cat).
    """
    if intent in ("factual_wiki", "factual_mixed"):
        return False, []

    if not _intent_is_short(sub_queries, raw_query, cfg.disambig_short_query_token_cap):
        return False, []

    candidate_entries = list(source_map.items())
    if not candidate_entries:
        return False, []

    distinct: Dict[Tuple[str, str], Tuple[str, str, str, float]] = {}
    for tag, meta in candidate_entries:
        key = (
            meta.get("category", "") or "",
            meta.get("sub_category", "") or "",
        )
        score = float(meta.get("score", 0.0))
        existing = distinct.get(key)
        if existing is not None:
            continue
        label_parts: List[str] = []
        for k in ("service", "sub_category", "category"):
            v = meta.get(k, "") or ""
            if v:
                label_parts.append(v)
        label = " — ".join(label_parts) if label_parts else "(unspecified)"
        distinct[key] = (tag, label, score)

    if len(distinct) < cfg.disambig_min_distinct_services:
        return False, []

    # Sort by cosine desc and cap the candidate count so the user isn't
    # drowned in a 15-item list for ultra-generic queries.
    ordered = sorted(
        distinct.values(),
        key=lambda t: -t[2],
    )
    cap = max(cfg.disambig_min_distinct_services, cfg.disambig_candidate_cap)
    ordered = ordered[:cap]

    candidates = [(tag, label) for tag, label, score in ordered]
    return True, candidates


# ============================================================
# Composer message assembly
# ============================================================

def _build_composer_user_message(
    raw_user_query: str,
    sub_queries: List[str],
    source_map: Dict[str, Dict[str, Any]],
    disambiguate: bool,
    disambig_candidates: List[Tuple[str, str]],
) -> str:
    """Render the composer's user message: <context> + optional <disambiguate>
    + <user_query>."""
    ctx_parts: List[str] = ["<context>"]
    if not source_map:
        ctx_parts.append("(no passages retrieved)")
    else:
        for tag, meta in source_map.items():
            hb: List[str] = []
            for key, label in (
                ("category", "বিভাগ"),
                ("sub_category", "উপ-বিভাগ"),
                ("service", "সেবা"),
                ("topic", "বিষয়"),
            ):
                v = meta.get(key, "") or ""
                if v:
                    hb.append(f"{label}: {v}")
            # Source-type label for unified corpus
            chunk_type = meta.get("chunk_type", "")
            if chunk_type == "wiki":
                hb.append("উৎস: উইকিপিডিয়া")
            elif chunk_type == "govt_service":
                hb.append("উৎস: সরকারি সেবা")
            header = " | ".join(hb) if hb else "—"
            ctx_parts.append("")
            ctx_parts.append(f"[{tag}] ({header})")
            ctx_parts.append(meta.get("text", ""))
    ctx_parts.append("</context>")

    sections = ["\n".join(ctx_parts)]

    if disambiguate and disambig_candidates:
        cand_lines = "\n".join(
            f"  - [{tag}] {label}" for tag, label in disambig_candidates
        )
        sections.append(
            "<disambiguate>\n"
            "The user's question is short and could match MULTIPLE distinct "
            "services/categories below. Your task: list them briefly with "
            "their [S#] tags and ask which one the user means. Do NOT pick "
            "one arbitrarily, do NOT refuse, and do NOT answer the question.\n\n"
            "Candidates to ask about:\n"
            f"{cand_lines}\n"
            "</disambiguate>"
        )

    sections.append(
        "<user_query>\n" + raw_user_query + "\n</user_query>"
    )

    if len(sub_queries) > 1 and not disambiguate:
        sub_lines = "\n".join(
            f"  {i}. {q}" for i, q in enumerate(sub_queries, start=1)
        )
        sections.append(
            "Note: the question contains multiple sub-questions:\n"
            f"{sub_lines}\n"
            "Address each in a short paragraph in order."
        )

    return "\n\n".join(sections)


def _sanitize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop <thinking>…</thinking> blocks from prior assistant turns."""
    out: List[Dict[str, str]] = []
    for msg in history or []:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "assistant":
            content = re.sub(r"<thinking>.*?</thinking>\s*", "", content, flags=re.DOTALL)
        out.append({"role": role, "content": content})
    return out


# ============================================================
# The pipeline
# ============================================================

async def run_factual_pipeline(
    raw_query: str,
    router_result: RouterResult,
    history: List[Dict[str, str]],
    primary_client: AsyncOpenAI,
    primary_model: str,
    secondary_client: AsyncOpenAI,
    secondary_model: str,
    cfg: PipelineConfig,
    http_client: Optional[httpx.AsyncClient] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run Stages 2 → 4 for a factual query.

    Caller handles Stages 0 (sanitize) and 1 (router), and routes
    chitchat/political-refuse intents to their static handlers. This function
    requires router_result.intent in {factual_govt, factual_wiki, factual_mixed}.

    Yields events; the caller forwards them to the API/UI. Channel filtering
    happens at the API boundary.
    """
    assert router_result.intent in ("factual_govt", "factual_wiki", "factual_mixed"), (
        "run_factual_pipeline received non-factual intent; route correctly upstream"
    )
    sub_queries = normalize_sub_queries(router_result.sub_queries_bengali or [raw_query])
    yield _evt("pipeline_start", n_subs=len(sub_queries))

    # ------------------------------------------------------------
    # Stage 2 — Single Jiggasha POST (instruction + embed + Qdrant + threshold)
    # ------------------------------------------------------------
    own_http = http_client is None
    http = http_client or httpx.AsyncClient(timeout=cfg.jiggasha_timeout)
    try:
        try:
            jres = await _call_jiggasha_multi(
                http, cfg.jiggasha_endpoint, sub_queries, cfg,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Jiggasha /search failed: %s", e)
            yield _evt(
                "retrieval_failed", channel="debug",
                error=str(e),
            )
            yield _evt("answer_chunk", channel="both", content=cfg.refusal_text_bn)
            yield _evt(
                "final_answer", channel="both",
                content=cfg.refusal_text_bn, source_map={},
                reason="jiggasha_failed",
            )
            yield _evt("answer_complete", channel="both")
            return
    finally:
        if own_http:
            await http.aclose()

    passages = jres.get("results", []) or []
    instruction = jres.get("instructions")  # list of instructions
    retrieval_elapsed_ms = jres.get("elapsed_ms")
    jiggasha_errors = jres.get("errors")

    yield _evt(
        "retrieval_done",
        passages=passages,
        passages_returned=len(passages),
        instructions=instruction,
        elapsed_ms=retrieval_elapsed_ms,
        jiggasha_errors=jiggasha_errors,
    )

    if not passages:
        yield _evt("answer_chunk", channel="both", content=cfg.refusal_text_bn)
        yield _evt(
            "final_answer", channel="both",
            content=cfg.refusal_text_bn, source_map={},
            reason="no_passages",
        )
        yield _evt("answer_complete", channel="both")
        return

    # ------------------------------------------------------------
    # ReAct retrieval loop — judge sufficiency, refine if needed
    # ------------------------------------------------------------
    if cfg.max_react_iterations > 0 and secondary_client is not None:
        for iteration in range(cfg.max_react_iterations):
            sufficient, refined = await _judge_retrieval_sufficiency(
                raw_query, passages, secondary_client, secondary_model,
            )
            if sufficient or not refined:
                break
            # Fresh client per iteration (original client already closed).
            react_http = httpx.AsyncClient(timeout=cfg.jiggasha_timeout)
            try:
                jres_refined = await _call_jiggasha_multi(
                    react_http, cfg.jiggasha_endpoint, [refined], cfg,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("ReAct retrieval failed on iteration %d: %s", iteration + 1, e)
                break
            finally:
                await react_http.aclose()
            new_passages = jres_refined.get("results", []) or []
            existing_ids = {p["passage_id"] for p in passages if p.get("passage_id") is not None}
            added = 0
            for p in new_passages:
                pid = p.get("passage_id")
                if pid is not None and pid not in existing_ids:
                    passages.append(p)
                    existing_ids.add(pid)
                    added += 1
            yield _evt(
                "retrieval_refined",
                iteration=iteration + 1,
                refined_query=refined,
                new_passages=added,
            )
            if added == 0:
                break

    source_map = _build_source_map_from_passages(passages)
    yield _evt(
        "source_map_allocated",
        n_sources=len(source_map),
        tags=list(source_map.keys()),
    )

    # Document-type guard: if the user explicitly asked for a specific
    # document type (e.g., marriage certificate) and NONE of the retrieved
    # passages match that type, refuse instead of offering irrelevant
    # disambiguation options.
    if not check_document_type_match(raw_query, source_map):
        yield _evt(
            "document_type_mismatch", channel="debug",
            note="retrieved passages do not match user's explicit document type",
        )
        yield _evt("answer_chunk", channel="both", content=cfg.refusal_text_bn)
        yield _evt(
            "final_answer", channel="both",
            content=cfg.refusal_text_bn, source_map={},
            reason="document_type_mismatch",
        )
        yield _evt("answer_complete", channel="both")
        return

    disambiguate, disambig_candidates = _detect_disambiguation(
        source_map, raw_query, sub_queries, cfg,
        intent=router_result.intent,
    )
    if disambiguate:
        yield _evt(
            "disambiguate_required",
            n_candidates=len(disambig_candidates),
            tags=[t for t, _ in disambig_candidates],
        )

    # ------------------------------------------------------------
    # Stage 3 — Compose (streaming primary LLM)
    # ------------------------------------------------------------
    # vLLM only accepts a single leading system message. Fold the per-turn
    # time-reminder into the trailing suffix of the system prompt so the
    # long static prefix still hits the prompt cache; only the tail varies.
    system_content = (
        get_composer_prompt(agent_name=cfg.agent_name)
        + "\n\n"
        + build_time_reminder()
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(_sanitize_history(history))
    messages.append({
        "role": "user",
        "content": _build_composer_user_message(
            raw_query, sub_queries, source_map,
            disambiguate, disambig_candidates,
        ),
    })

    yield _evt(
        "composer_start",
        model=primary_model,
        prompt_chars=sum(len(m.get("content", "")) for m in messages),
        n_messages=len(messages),
    )

    answer_acc: List[str] = []
    composer_usage: Optional[Dict[str, int]] = None
    parser = ThinkingParser()
    try:
        stream = await primary_client.chat.completions.create(
            model=primary_model,
            messages=messages,
            stream=True,
            temperature=cfg.composer_temperature,
            top_p=cfg.composer_top_p,
            max_tokens=cfg.composer_max_tokens,
            # vLLM honors this and emits a final delta carrying `usage`.
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            # `usage` shows up on the LAST chunk (delta is empty or absent).
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                try:
                    composer_usage = {
                        "prompt": int(getattr(chunk_usage, "prompt_tokens", 0) or 0),
                        "completion": int(
                            getattr(chunk_usage, "completion_tokens", 0) or 0
                        ),
                    }
                except Exception:  # noqa: BLE001
                    pass
            try:
                delta = chunk.choices[0].delta
                text = (getattr(delta, "content", None) or "")
            except (IndexError, AttributeError):
                continue
            if not text:
                continue
            for channel, piece in parser.feed(text):
                if channel == "answer":
                    answer_acc.append(piece)
                    yield _evt("answer_chunk", channel="both", content=piece)
                else:
                    yield _evt("reasoning_chunk", content=piece)
        for channel, piece in parser.flush():
            if channel == "answer":
                answer_acc.append(piece)
                yield _evt("answer_chunk", channel="both", content=piece)
            else:
                yield _evt("reasoning_chunk", content=piece)
    except Exception as e:  # noqa: BLE001
        logger.error("composer stream failed: %s", e, exc_info=True)
        yield _evt(
            "error", channel="both",
            content=cfg.refusal_text_bn, detail=str(e),
        )
        yield _evt("answer_complete", channel="both")
        return

    raw_answer = "".join(answer_acc).strip()
    yield _evt(
        "composer_done",
        chars=len(raw_answer),
        token_usage=composer_usage,
    )

    if not raw_answer:
        yield _evt(
            "final_answer", channel="both",
            content=cfg.refusal_text_bn, source_map=source_map,
            reason="composer_empty",
        )
        yield _evt("answer_complete", channel="both")
        return

    # ------------------------------------------------------------
    # Stage 4 — Post-flight
    # ------------------------------------------------------------
    final_answer, post_events, sources_block = await _post_flight(
        raw_answer=raw_answer,
        source_map=source_map,
        cfg=cfg,
        secondary_client=secondary_client,
        secondary_model=secondary_model,
        # Disambiguation responses are short and structured; skip NLI on them.
        skip_verify=disambiguate,
    )
    for ev in post_events:
        yield ev

    # Stream the canonical Sources block as answer_chunk so the UI
    # shows it incrementally instead of only inside final_answer.
    if sources_block:
        yield _evt("answer_chunk", channel="both", content="\n\n" + sources_block)

    yield _evt(
        "final_answer", channel="both",
        content=final_answer,
        source_map={
            tag: {k: v for k, v in meta.items() if k != "text"}
            for tag, meta in source_map.items()
        },
    )
    yield _evt("answer_complete", channel="both")


# ============================================================
# Stage 4 — Post-flight (strip + verify + Sources block)
# ============================================================

_CITE_TAG_RE = re.compile(r"\[S\d+\]")


def _strip_composer_sources_block(text: str) -> str:
    """Defensively cut any composer-emitted Sources/সূত্র block.

    If the body before the block has NO inline [S#] tags but the block
    itself listed tags, harvest those tags and append them to the body's
    last paragraph before stripping — otherwise the answer would lose all
    citations and `_post_flight` would force the static refusal.
    """
    if not text:
        return text
    patterns = [
        # --- separator style
        r"\n+---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b",
        r"(?:^|\n+)---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b",
        # **bold header style
        r"\n+\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*",
        r"(?:^|\n+)\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*",
        # plain header style
        r"\n+(?:সূত্র|উৎস|Sources)\s*[:：]",
        r"(?:^|\n+)(?:সূত্র|উৎস|Sources)\s*[:：]",
        # bullet-list-of-tags at end (composer sometimes emits bare list)
        r"\n+(?:\s*[-*]\s*\[S\d+\][^\n]*\n*)+$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        body = text[: m.start()].rstrip()
        trailing = text[m.start():]
        # If body has no inline cites but the trailing block does, harvest
        # those tags onto the last non-empty line of the body so the answer
        # survives the post-flight no-citations check.
        if not _CITE_TAG_RE.search(body):
            trailing_tags = list(dict.fromkeys(_CITE_TAG_RE.findall(trailing)))
            if trailing_tags:
                tag_suffix = " " + " ".join(trailing_tags)
                # Attach to the last non-empty line (paragraph break safe).
                lines = body.rstrip().split("\n")
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip():
                        lines[i] = lines[i].rstrip() + tag_suffix
                        break
                body = "\n".join(lines)
        text = body
    return text


# Phrases the composer uses to acknowledge "this specific thing is not in
# the corpus" — i.e. the start of mode (B) at the sentence level.
_PARTIAL_GAP_RE = re.compile(
    r"("
    r"(?:নির্দিষ্ট\s+[^।\n]{0,40}\s+)?উল্লেখ\s+নেই"
    r"|উল্লিখিত\s+নয়"
    r"|(?:সঠিক|নির্দিষ্ট)\s+তথ্য\s+পাওয়া\s+যায়নি"
    r"|(?:তথ্য\s+)?প্রসঙ্গে\s+(?:নেই|উল্লেখ\s+নেই)"
    r"|নির্দিষ্ট\s+তথ্য\s+নেই"
    r")",
    flags=re.IGNORECASE,
)

# The "but here's the general procedure" lead-in that should NEVER appear
# in a (B) response. When it follows a partial-gap phrase, it's mode-mixing.
_MODE_MIX_PARAGRAPH_RE = re.compile(
    r"(?:তবে|তথাপি|তবু|যদিও)[^।\n]*"
    r"(?:সাধারণ(?:ভাবে|ত)?|সাধারণ\s+পদ্ধতি|সাধারণ\s+আইনানুগ|"
    r"নিচে\s+দেওয়া\s+হলো|নিচে\s+দেয়া\s+হলো)",
    flags=re.IGNORECASE,
)


_B_HEADER_RE = re.compile(
    r"\n\s*এই\s+নির্দিষ্ট\s+বিষয়ে\s+সঠিক\s+তথ্য\s+পাওয়া\s+যায়নি"
)


def _strip_mode_mix_paragraph(text: str) -> Tuple[str, bool]:
    """If the answer admits a gap and then prepends "but here's the general
    procedure" anyway, strip everything from the bridge phrase up to either
    the (B) bullet header or end of text.

    This is a deterministic safety net for the composer's mode-mix bug —
    the prompt alone does not reliably suppress it at temp 0.1.

    Returns (cleaned_text, stripped_bool).
    """
    if not text:
        return text, False
    gap_match = _PARTIAL_GAP_RE.search(text)
    if not gap_match:
        return text, False
    tail = text[gap_match.end():]
    mix_match = _MODE_MIX_PARAGRAPH_RE.search(tail)
    if not mix_match:
        return text, False

    bridge_start = gap_match.end() + mix_match.start()
    rest_after_bridge = text[bridge_start:]

    # Cut up to the (B) header if present; otherwise drop to end of text.
    b_match = _B_HEADER_RE.search(rest_after_bridge)
    if b_match:
        end_rel = b_match.start()
        cleaned = (
            text[:bridge_start].rstrip()
            + "\n\n"
            + rest_after_bridge[end_rel:].lstrip()
        ).strip()
    else:
        cleaned = text[:bridge_start].rstrip()
    return cleaned, True


async def _post_flight(
    raw_answer: str,
    source_map: Dict[str, Dict[str, Any]],
    cfg: PipelineConfig,
    secondary_client: AsyncOpenAI,
    secondary_model: str,
    skip_verify: bool = False,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Strip composer Sources block, strip unknown tags, NLI verify, append
    canonical Sources block.

    Never raises; degrades to unverified/unsourced on failures.
    """
    events: List[Dict[str, Any]] = []

    raw_answer = _strip_composer_sources_block(raw_answer)
    raw_answer, mode_mix_stripped = _strip_mode_mix_paragraph(raw_answer)
    if mode_mix_stripped:
        events.append(_evt(
            "mode_mix_stripped",
            note="removed 'general procedure' paragraph after partial-gap caveat",
        ))
    cleaned, dropped = strip_unknown_tags(raw_answer, source_map)
    if dropped:
        for tag in dropped:
            events.append(_evt(
                "unsupported_claim",
                tag=tag, verdict="tag_not_in_source_map", action="stripped",
            ))

    used_tags = extract_citation_tags(cleaned)
    if not used_tags:
        return cfg.refusal_text_bn, events, ""

    if not skip_verify and cfg.verifier_enabled and secondary_client is not None:
        pairs = extract_citations(cleaned)
        pairs = [(t, s) for (t, s) in pairs if t in source_map]
        if pairs:
            events.append(_evt("verification_start", count=len(pairs)))
            try:
                verdicts, nli_usage = await verify_claims(
                    pairs=pairs,
                    source_map=source_map,
                    secondary_client=secondary_client,
                    secondary_model=secondary_model,
                    timeout=cfg.verifier_timeout,
                )
                cleaned, policy_events = apply_policy(
                    answer=cleaned,
                    pairs=pairs,
                    verdicts=list(verdicts),
                    policy=cfg.verifier_policy,
                    refusal_text=cfg.refusal_text_bn,
                )
                events.append(_evt(
                    "verification_result",
                    pairs=len(pairs),
                    verdicts=list(verdicts),
                    token_usage=nli_usage,
                ))
                for pe in policy_events:
                    if "channel" not in pe:
                        pe["channel"] = "debug"
                    events.append(pe)
            except Exception as e:  # noqa: BLE001
                logger.warning("Verifier pipeline failed (%s); keeping unverified.", e)
                events.append(_evt(
                    "verification_result",
                    action="degraded_failed", error=str(e),
                ))

    if cleaned.strip() == cfg.refusal_text_bn.strip():
        return cfg.refusal_text_bn, events, ""

    # Final safety net: if the composer STILL emitted a sources block that
    # survived stripping, cut it before appending the canonical one.
    cleaned = _strip_composer_sources_block(cleaned).rstrip()

    used_tags_after = extract_citation_tags(cleaned)
    sources_block = build_sources_block(source_map, used_tags_after)
    final = cleaned.rstrip() + (("\n\n" + sources_block) if sources_block else "")
    return final, events, sources_block
