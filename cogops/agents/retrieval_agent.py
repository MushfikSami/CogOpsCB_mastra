"""
cogops/agents/retrieval_agent.py

Layer 3 — RetrievalAgent.

ReAct-style loop:
  1. Call Jiggasha /search (single query per call)
  2. RetrievalJudge evaluates sufficiency
  3. If insufficient → generate refined_query → search again
  4. Max iterations = config.max_react_iterations

Merge results across all sub-queries: dedupe by passage_id, keep highest
rerank_score, build unified source_map.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from cogops.prompts.time_reminder import build_time_reminder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    endpoint: str = "http://localhost:10000/search"
    timeout: float = 45.0
    top_k_fetch: int = 50
    use_instruction: bool = True
    cosine_threshold: Optional[float] = None
    token_budget: Optional[int] = None  # forwarded to Jiggasha; enforced there
    rerank_threshold: Optional[float] = None
    max_react_iterations: int = 0
    merge_global_cap: int = 50


@dataclass
class RetrievalResult:
    passages: List[Dict[str, Any]]
    source_map: Dict[str, Dict[str, Any]]
    instructions: List[str]
    elapsed_ms: int
    timing_ms: Dict[str, int]
    token_usage: Dict[str, int]
    errors: Optional[List[str]]


# ------------------------------------------------------------------
# Jiggasha client
# ------------------------------------------------------------------

async def _call_jiggasha(
    http: httpx.AsyncClient,
    endpoint: str,
    query: str,
    cfg: RetrievalConfig,
) -> Dict[str, Any]:
    """POST one /search to Jiggasha with retries."""
    payload = {
        "query": query,
        "top_k": cfg.top_k_fetch,
        "use_instruction": cfg.use_instruction,
        "cosine_threshold": cfg.cosine_threshold,
        "token_budget": cfg.token_budget,
    }
    # Reranking removed — Jiggasha service no longer uses a reranker.
    if cfg.rerank_threshold is not None:
        payload["rerank_threshold"] = cfg.rerank_threshold
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await http.post(endpoint, json=payload)
            if resp.status_code >= 500 and attempt < 2:
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
    if last_exc:
        raise last_exc
    raise RuntimeError("jiggasha call failed without exception")


async def _call_jiggasha_multi(
    http: httpx.AsyncClient,
    endpoint: str,
    queries: List[str],
    cfg: RetrievalConfig,
) -> Dict[str, Any]:
    """Call Jiggasha once per query in parallel, merge & deduplicate."""
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

        max_elapsed = max(max_elapsed, res.get("elapsed_ms", 0) or 0)
        tm = res.get("timing_ms") or {}
        for k in timing_agg:
            v = tm.get(k)
            if isinstance(v, (int, float)):
                timing_agg[k].append(int(v))

        tu = res.get("token_usage") or {}
        for k in token_agg:
            v = tu.get(k)
            if isinstance(v, (int, float)):
                token_agg[k] += int(v)

        all_instructions.append(res.get("instruction"))

        for p in (res.get("results") or []):
            pid = p.get("passage_id")
            if pid is None:
                continue
            existing = merged_passages.get(pid)
            if existing is None:
                merged_passages[pid] = dict(p)
            else:
                new_score = p.get("rerank_score")
                old_score = existing.get("rerank_score")
                if new_score is not None and old_score is not None:
                    if new_score > old_score:
                        merged_passages[pid] = dict(p)
                elif new_score is not None:
                    merged_passages[pid] = dict(p)
                elif p.get("score", 0.0) > existing.get("score", 0.0):
                    merged_passages[pid] = dict(p)

    if not merged_passages and first_exc is not None:
        raise first_exc

    def _sort_key(p: Dict[str, Any]) -> Tuple[float, float]:
        rs = p.get("rerank_score")
        return (-(rs if rs is not None else 0.0), -(p.get("score", 0.0)))

    sorted_passages = sorted(merged_passages.values(), key=_sort_key)

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


# ------------------------------------------------------------------
# RetrievalJudge
# ------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a retrieval judge for a Bangladesh government-services chatbot.

You will be given a user query and up to 5 retrieved passages. Decide whether
the passages are SUFFICIENT to directly answer the query.

"Sufficient" means at least one passage explicitly covers the user's exact
subject (procedure, fee, eligibility, contact, office, etc.), not just a
topically related area.

Output ONLY a JSON object with this exact shape:
  {"sufficiency": "sufficient"}
or
  {"sufficiency": "insufficient", "refined_query": "<improved formal Bengali query>"}
or
  {"sufficiency": "partial", "refined_query": "<improved formal Bengali query>"}

If insufficient or partial, provide a refined_query that is more specific and
formal. Keep it concise (under 20 words)."""


class RetrievalJudge:
    """Evaluate whether retrieved passages are sufficient to answer a query."""

    def __init__(
        self,
        secondary_client: Optional[AsyncOpenAI],
        secondary_model: str,
        timeout: float = 5.0,
    ):
        self.client = secondary_client
        self.model = secondary_model
        self.timeout = timeout

    async def judge(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[str]]:
        """Return (sufficiency, refined_query_or_none).

        sufficiency is one of: "sufficient", "partial", "insufficient".
        On failure, returns ("sufficient", None) — fail-open.
        """
        if not passages or not self.client:
            return "insufficient", query

        summary_lines: List[str] = []
        for i, p in enumerate(passages[:5], start=1):
            text = (p.get("text") or "")[:300]
            summary_lines.append(f"[{i}] {text}")
        passage_block = "\n\n".join(summary_lines)

        user_msg = (
            f"Query: {query}\n\n"
            f"Retrieved passages:\n{passage_block}\n\n"
            "Judge sufficiency and provide refined query if needed."
        )

        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "assistant", "content": build_time_reminder()},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=128,
                ),
                timeout=self.timeout,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            sufficiency = str(data.get("sufficiency", "sufficient")).lower()
            refined = data.get("refined_query")
            if isinstance(refined, str):
                refined = refined.strip()
                if not refined:
                    refined = None
            else:
                refined = None
            return sufficiency, refined
        except Exception as e:  # noqa: BLE001
            logger.warning("RetrievalJudge failed (%s); assuming sufficient.", e)
            return "sufficient", None


# ------------------------------------------------------------------
# Source map builder
# ------------------------------------------------------------------

def build_source_map(passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build [S#] → passage meta map. All passages get verdict='yes'."""
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
            "rerank_score": float(p.get("rerank_score", 0.0)) if p.get("rerank_score") is not None else None,
            "verdict": "yes",
            "tool": "jiggasha",
        }
    return source_map


# ------------------------------------------------------------------
# RetrievalAgent
# ------------------------------------------------------------------

class RetrievalAgent:
    """Layer 3 — retrieve passages with iterative ReAct refinement."""

    def __init__(
        self,
        cfg: RetrievalConfig,
        secondary_client: Optional[AsyncOpenAI],
        secondary_model: str,
    ):
        self.cfg = cfg
        self.judge = RetrievalJudge(secondary_client, secondary_model)

    async def retrieve(
        self,
        queries: List[str],
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> RetrievalResult:
        """Retrieve passages for all queries, with optional ReAct refinement.

        Args:
            queries: list of formalized standalone queries.
            http_client: optional shared httpx client.

        Returns:
            RetrievalResult with merged passages and source_map.
        """
        own_http = http_client is None
        http = http_client or httpx.AsyncClient(timeout=self.cfg.timeout)
        try:
            # Initial retrieval for all queries in parallel
            jres = await _call_jiggasha_multi(
                http, self.cfg.endpoint, queries, self.cfg,
            )
        finally:
            if own_http:
                await http.aclose()

        passages = jres.get("results", []) or []

        # ReAct loop: judge sufficiency, refine if needed
        if self.cfg.max_react_iterations > 0:
            for iteration in range(self.cfg.max_react_iterations):
                # Judge overall sufficiency against the first query (primary intent)
                primary_query = queries[0] if queries else ""
                sufficiency, refined = await self.judge.judge(primary_query, passages)
                if sufficiency == "sufficient" or not refined:
                    break

                react_http = httpx.AsyncClient(timeout=self.cfg.timeout)
                try:
                    jres_refined = await _call_jiggasha_multi(
                        react_http, self.cfg.endpoint, [refined], self.cfg,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("ReAct retrieval failed iter %d: %s", iteration + 1, e)
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

                logger.info(
                    "ReAct iter %d: sufficiency=%s added=%d",
                    iteration + 1, sufficiency, added,
                )
                if added == 0:
                    break

        # Global cap
        if len(passages) > self.cfg.merge_global_cap:
            passages = passages[: self.cfg.merge_global_cap]

        source_map = build_source_map(passages)

        return RetrievalResult(
            passages=passages,
            source_map=source_map,
            instructions=jres.get("instructions") or [],
            elapsed_ms=jres.get("elapsed_ms", 0),
            timing_ms=jres.get("timing_ms") or {},
            token_usage=jres.get("token_usage") or {},
            errors=jres.get("errors"),
        )
