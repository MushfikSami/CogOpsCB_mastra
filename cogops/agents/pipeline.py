"""
cogops/agents/pipeline.py

Deterministic factual-query pipeline. Stages:

    Stage 0  sanitize                  (pure code, no LLM)             — caller
    Stage 1  router                     (1 secondary-LLM call)          — caller
    Stage 2  Jiggasha multi-query rerank (1 HTTP POST, server-side LLM) — here
    Stage 3  compose                    (1 primary-LLM streaming call)  — here
    Stage 4  post-flight                (NLI verify + Sources block)    — here

Stage 2 collapses what used to be the chatbot-side "parallel retrieve +
LLM rerank" into a single POST to Jiggasha (`{sub_queries, rerank: true}`).
Jiggasha returns:
    {
      sub_queries: [...],
      passages: [{passage_id, text, category, ..., score}, ...],
      rerank: {"1": [[pid, cls], ...], "2": [...]},
      degraded: false
    }
where `cls` is 0=yes (directly answers) or 1=weak (tangential backfill).

This module is intentionally framework-light: depends only on
  - cogops.pipeline.router (RouterResult)
  - cogops.prompts.composer / time_reminder
  - cogops.verifier.citations / nli / policy
  - cogops.utils.thinking_parser
  - openai.AsyncOpenAI + httpx
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from cogops.pipeline.normalize import normalize_sub_queries
from cogops.pipeline.query_expand import check_document_type_match, expand_sub_queries
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

    # Jiggasha (multi-query rerank path)
    jiggasha_endpoint: str = "http://localhost:10000/search"
    jiggasha_timeout: float = 45.0
    jiggasha_rerank: bool = True
    top_k_per_sub: int = 20
    candidate_cap_global: int = 30
    keep_cap: int = 24
    weak_per_sub_cap: int = 3
    fallback_cosine_min: float = 0.50

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
    disambig_short_query_token_cap: int = 6
    disambig_candidate_cap: int = 6   # max candidates shown to the user

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
# Stage 2 — Single Jiggasha call (server-side LLM rerank)
# ============================================================

async def _call_jiggasha(
    http: httpx.AsyncClient,
    endpoint: str,
    sub_queries: List[str],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """POST one /search with the multi-query rerank shape.

    Retries on transient 5xx / network errors. Jiggasha applies its own
    concurrency cap; under load it may briefly 503 before the semaphore
    drains. Two retries with exponential backoff are enough to mask that.
    Raises on the final failure.
    """
    payload = {
        "sub_queries": sub_queries,
        "top_k_per_sub": cfg.top_k_per_sub,
        "rerank": cfg.jiggasha_rerank,
        "candidate_cap_global": cfg.candidate_cap_global,
        "keep_cap": cfg.keep_cap,
        "weak_per_sub_cap": cfg.weak_per_sub_cap,
        "fallback_cosine_min": cfg.fallback_cosine_min,
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


def _allocate_source_map_from_rerank(
    passages: List[Dict[str, Any]],
    rerank: Dict[str, List[List[int]]],
    sub_queries: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build the `[S#] → passage meta` map preserving Jiggasha's order.

    For each passage the per-sub `verdict_by_sub` records which sub-questions
    flagged it `yes` (0) or `weak` (1). The aggregate `verdict` is `yes` if
    ANY sub voted yes, else `weak`. `sub_indices` is the union of sub indices
    that voted for it (0-based).
    """
    # Invert rerank: pid -> {sub_idx: cls}
    pid_to_subs: Dict[int, Dict[int, int]] = {}
    for sub_key, entries in (rerank or {}).items():
        try:
            sub_idx = int(sub_key) - 1
        except (TypeError, ValueError):
            continue
        if sub_idx < 0:
            continue
        for entry in entries or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            try:
                pid = int(entry[0])
                cls = int(entry[1])
            except (TypeError, ValueError):
                continue
            pid_to_subs.setdefault(pid, {})[sub_idx] = cls

    source_map: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(passages, start=1):
        try:
            pid = int(p.get("passage_id", 0))
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        verdict_by_sub = pid_to_subs.get(pid, {})
        if any(v == 0 for v in verdict_by_sub.values()):
            verdict = "yes"
        elif verdict_by_sub:
            verdict = "weak"
        else:
            # In a well-behaved response every kept passage carries a verdict
            # for at least one sub; if Jiggasha skipped one, treat as weak.
            verdict = "weak"
        sub_indices = sorted(verdict_by_sub.keys())
        tag = f"S{i}"
        source_map[tag] = {
            "passage_id": pid,
            "text": p.get("text", ""),
            "category": p.get("category", "") or "",
            "sub_category": p.get("sub_category", "") or "",
            "service": p.get("service", "") or "",
            "topic": p.get("topic", "") or "",
            "score": float(p.get("score", 0.0)),
            "verdict": verdict,
            "verdict_by_sub": dict(verdict_by_sub),
            "sub_indices": sub_indices,
            "sub_queries": [
                sub_queries[j] for j in sub_indices if 0 <= j < len(sub_queries)
            ],
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
) -> Tuple[bool, List[Tuple[str, str]]]:
    """Decide whether the answer should ask for clarification.

    Triggers when:
      - the question has a SINGLE sub-question (multi-sub queries are already
        split by the user; per-sub disambiguation doesn't apply);
      - the normalized intent (sub_queries[0], or raw_query as fallback) is
        short — gauged in tokens, not characters; and
      - ≥ cfg.disambig_min_distinct_services distinct (category, sub_category)
        tuples appear among yes-or-weak kept passages.

    Weak counts here because for short/generic queries the LLM reranker often
    returns few yes verdicts but many weak ones spanning services — exactly
    when disambiguation matters most. The distinct key is (category,
    sub_category) and deliberately NOT `service`: the corpus's `service`
    field encodes per-aspect labels (e.g. "চারিত্রিক সনদ: ভাষা" vs
    "চারিত্রিক সনদ: সেবারমূল্য") which all describe the same actual service.
    For the "same sub_category, different boards" case (e.g. SSC name
    correction across Dhaka/Chittagong boards), the corpus already encodes
    each board in its own sub_category, so (cat, sub_cat) is sufficient.

    Returns (disambiguate, candidates) where candidates is a list of
    (tag, human-readable service name) — one per distinct (cat, sub_cat).
    Yes-verdict tags are preferred over weak when both exist for the same key.
    """
    if not _intent_is_short(sub_queries, raw_query, cfg.disambig_short_query_token_cap):
        return False, []

    candidate_entries: List[Tuple[str, Dict[str, Any]]] = [
        (tag, meta) for tag, meta in source_map.items()
        if meta.get("verdict") in ("yes", "weak")
    ]
    if not candidate_entries:
        return False, []

    distinct: Dict[Tuple[str, str], Tuple[str, str, str, float]] = {}
    for tag, meta in candidate_entries:
        key = (
            meta.get("category", "") or "",
            meta.get("sub_category", "") or "",
        )
        verdict = meta.get("verdict") or "weak"
        score = float(meta.get("score", 0.0))
        existing = distinct.get(key)
        if existing is not None and existing[2] == "yes":
            continue
        if existing is not None and verdict != "yes":
            continue
        label_parts: List[str] = []
        for k in ("service", "sub_category", "category"):
            v = meta.get(k, "") or ""
            if v:
                label_parts.append(v)
        label = " — ".join(label_parts) if label_parts else "(unspecified)"
        distinct[key] = (tag, label, verdict, score)

    if len(distinct) < cfg.disambig_min_distinct_services:
        return False, []

    # Sort by (yes-first, cosine desc) and cap the candidate count so the
    # user isn't drowned in a 15-item list for ultra-generic queries.
    ordered = sorted(
        distinct.values(),
        key=lambda t: (0 if t[2] == "yes" else 1, -t[3]),
    )
    cap = max(cfg.disambig_min_distinct_services, cfg.disambig_candidate_cap)
    ordered = ordered[:cap]

    candidates = [(tag, label) for tag, label, _, _ in ordered]
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
    requires router_result.intent == "factual_govt".

    Yields events; the caller forwards them to the API/UI. Channel filtering
    happens at the API boundary.
    """
    assert router_result.intent == "factual_govt", (
        "run_factual_pipeline received non-factual intent; route correctly upstream"
    )
    sub_queries = normalize_sub_queries(router_result.sub_queries_bengali or [raw_query])
    sub_queries = expand_sub_queries(sub_queries)
    yield _evt("pipeline_start", n_subs=len(sub_queries))

    # ------------------------------------------------------------
    # Stage 2 — Single Jiggasha POST (embed + Qdrant + LLM rerank)
    # ------------------------------------------------------------
    own_http = http_client is None
    http = http_client or httpx.AsyncClient(timeout=cfg.jiggasha_timeout)
    try:
        try:
            jres = await _call_jiggasha(
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

    passages = jres.get("passages", []) or []
    rerank = jres.get("rerank", {}) or {}
    degraded = bool(jres.get("degraded", False))
    rerank_usage = jres.get("rerank_usage")
    rerank_elapsed_ms = jres.get("elapsed_ms")

    yield _evt(
        "retrieval_done",
        passages=passages,
        rerank=rerank,
        passages_returned=len(passages),
        degraded=degraded,
        per_sub_counts={k: len(v) for k, v in rerank.items()},
        token_usage=rerank_usage,
        elapsed_ms=rerank_elapsed_ms,
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

    source_map = _allocate_source_map_from_rerank(passages, rerank, sub_queries)
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
        r"\n+---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b",
        r"\n+\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*",
        r"\n+(?:সূত্র|উৎস|Sources)\s*[:：]",
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
    the prompt alone does not reliably suppress it at temp 0.1, especially
    when the rerank has marked multiple adjacent passages as yes.

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

    used_tags_after = extract_citation_tags(cleaned)
    sources_block = build_sources_block(source_map, used_tags_after)
    final = cleaned.rstrip() + (("\n\n" + sources_block) if sources_block else "")
    return final, events, sources_block
