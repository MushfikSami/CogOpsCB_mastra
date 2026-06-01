#!/usr/bin/env python3
"""FastAPI service for Bengali passage retrieval.

POST /search has two content-driven modes:

  Legacy single-query:
      { "query": "...", "top_k": 20 }
    →
      { "query": "...", "results": [...], "hits_total": N }

  Multi-query instruction-based retrieval:
      { "sub_queries": ["...", "..."],
        "use_instruction": true,
        "cosine_threshold": 0.70,
        "token_budget": 28000 }
    →
      { "sub_queries": [...],
        "passages": [...],
        "instruction": "...",
        "elapsed_ms": 123 }

Usage:
    python3 service.py                    # start on port 10000
    python3 service.py --port 8080
"""
import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from embedder import Embedder
from instruction import generate_instruction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(path: str = "config.yml") -> dict:
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env(key: str | None) -> str:
    if not key:
        return ""
    if key not in os.environ:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get(key, "")


# ============================================================ #
# Module-level globals (rebuilt in main() to honor --config)
# ============================================================ #

_cfg = _load_config()
app = FastAPI(title="Jiggasha Bengali Passage Search")

# Embedder + Qdrant + secondary LLM clients are built lazily so test code
# can import the module without booting the world.

embedder: Optional[Embedder] = None
qdrant_client = None
secondary_client = None
secondary_model: str = ""
retrieval_defaults: Dict[str, Any] = {}


def _build_clients() -> None:
    """(Re)build embedder, Qdrant client, secondary LLM client from _cfg."""
    global embedder, qdrant_client, secondary_client, secondary_model, retrieval_defaults

    embedder_url = _resolve_env(_cfg["embedder"].get("url_env"))
    embedder_api_key = _resolve_env(_cfg["embedder"].get("api_key_env"))
    embedder_model = _resolve_env(_cfg["embedder"].get("model_env")) or _cfg["embedder"].get("model", "")
    embedder = Embedder(
        url=embedder_url,
        api_key=embedder_api_key,
        model=embedder_model,
        batch_size=_cfg["embedder"].get("batch_size", 64),
    )

    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            _resolve_env(_cfg["qdrant"].get("url_env")),
            timeout=_cfg["qdrant"].get("timeout", 30),
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to init Qdrant client: %s", e)
        qdrant_client = None

    retcfg = _cfg.get("retrieval", {}) or {}
    retrieval_defaults = {
        "use_instruction": bool(retcfg.get("use_instruction", False)),
        "static_instruction": (retcfg.get("static_instruction") or "").strip(),
        "cosine_threshold": float(retcfg.get("cosine_threshold", 0.70)),
        "token_budget": int(retcfg.get("token_budget", 28000)),
        "top_k_fetch": int(retcfg.get("top_k_fetch", 50)),
        "instruction_temperature": float(retcfg.get("instruction_temperature", 0.2)),
        "instruction_max_tokens": int(retcfg.get("instruction_max_tokens", 128)),
        "instruction_timeout": float(retcfg.get("instruction_timeout_seconds", 5.0)),
    }

    # Secondary LLM config lives under the legacy "rerank" key for backward
    # compatibility with existing deployments.
    rcfg = _cfg.get("rerank", {}) or {}
    secondary_url = _resolve_env(rcfg.get("base_url_env", "SECONDARY_BASE_URL"))
    secondary_key = _resolve_env(rcfg.get("api_key_env", "SECONDARY_API_KEY"))
    secondary_model = _resolve_env(rcfg.get("model_env", "SECONDARY_MODEL_NAME"))

    if secondary_url and secondary_key:
        try:
            from openai import AsyncOpenAI
            secondary_client = AsyncOpenAI(
                base_url=secondary_url,
                api_key=secondary_key,
            )
            logger.info(
                "secondary LLM ready at %s (model=%s)",
                secondary_url, secondary_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to init secondary LLM client: %s", e)
            secondary_client = None
    else:
        secondary_client = None
        logger.warning(
            "no secondary LLM configured; dynamic instruction generation will fail",
        )


_build_clients()


# ============================================================ #
# Pydantic request / response models
# ============================================================ #

class SearchRequest(BaseModel):
    """One model covers both modes; presence of `sub_queries` selects multi-query."""
    # Legacy single-query
    query: Optional[str] = None
    retrieval_instruction: Optional[str] = None
    top_k: int = 20

    # Multi-query instruction-based retrieval
    sub_queries: Optional[List[str]] = None
    top_k_per_sub: int = 20
    chunk_type: Optional[str] = None   # "wiki" | "govt_service" | null

    # Instruction-based retrieval knobs
    use_instruction: bool = False
    cosine_threshold: Optional[float] = None
    token_budget: Optional[int] = None


# ============================================================ #
# Shared helpers
# ============================================================ #

def _qdrant_topk(query_vec: List[float], top_k: int, chunk_type: Optional[str] = None) -> List[Any]:
    """Sync Qdrant call. Caller is responsible for offloading from event loop."""
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialised")
    query_filter = None
    if chunk_type:
        from qdrant_client import models
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type),
                )
            ]
        )
    return qdrant_client.query_points(
        collection_name=_cfg["qdrant"]["collection"],
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
    ).points


def _hit_to_passage(hit: Any) -> Dict[str, Any]:
    payload = hit.payload or {}

    # passage_id: old schema has integer; new unified schema may not.
    # Fall back to a deterministic hash of the Qdrant point ID.
    pid_raw = payload.get("passage_id")
    if pid_raw is not None:
        passage_id = int(pid_raw)
    else:
        id_str = str(hit.id)
        try:
            passage_id = int(id_str)
        except ValueError:
            # UUID or other string id: use last 8 hex chars as int
            hex_part = id_str.replace("-", "")[-8:]
            passage_id = int(hex_part, 16)

    # Schema mapping: unified (bnwiki_chunks) → legacy (pipeline expects)
    category = payload.get("category") or payload.get("page_title", "")
    sub_category = payload.get("sub_category") or payload.get("section", "")
    service = payload.get("service") or payload.get("subsection", "")
    topic = payload.get("topic") or payload.get("page_title", "")

    return {
        "passage_id": passage_id,
        "text": payload.get("text", ""),
        "category": category,
        "sub_category": sub_category,
        "service": service,
        "topic": topic,
        "chunk_type": payload.get("chunk_type", ""),
        "llm_token_count": int(payload.get("llm_token_count", 0)),
        "score": float(hit.score) if hasattr(hit, "score") and hit.score is not None else 0.0,
    }


# ============================================================ #
# Instruction + threshold helpers
# ============================================================ #

async def _build_retrieval_instruction(req: SearchRequest) -> Optional[str]:
    """Return an instruction string, or None to fall back to raw query.

    Priority:
      1. Caller-supplied instruction (req.retrieval_instruction)
      2. Static instruction from config (zero latency, zero failure)
      3. Dynamic LLM-generated instruction (fallback when static is empty)
    """
    # 1. Caller-supplied instruction takes highest precedence.
    if req.retrieval_instruction:
        return req.retrieval_instruction.strip()

    # 2. Config / request flag must be set.
    if not req.use_instruction and not retrieval_defaults.get("use_instruction"):
        return None

    # 3. Static instruction from config — zero latency, reliable.
    static = retrieval_defaults.get("static_instruction", "")
    if static:
        return static

    # 4. Dynamic fallback — only when static is not configured.
    query_text = req.query or ""
    if not query_text and req.sub_queries:
        query_text = req.sub_queries[0]
    query_text = query_text.strip()
    if not query_text:
        return None

    instruction = await generate_instruction(
        query=query_text,
        secondary_client=secondary_client,
        secondary_model=secondary_model,
        temperature=retrieval_defaults.get("instruction_temperature", 0.2),
        max_tokens=retrieval_defaults.get("instruction_max_tokens", 128),
        timeout=retrieval_defaults.get("instruction_timeout", 5.0),
    )
    return instruction


def _apply_threshold_and_budget(
    candidates: List[Dict[str, Any]],
    threshold: float,
    budget: int,
    score_gap: float = 0.10,
) -> List[Dict[str, Any]]:
    """Filter by cosine threshold, greedily fit into token budget, and
    truncate at score cliffs to avoid tangentially related passages.

    Falls back to the top-3 raw candidates when nothing passes the threshold.
    """
    if not candidates:
        return []

    filtered = [c for c in candidates if c.get("score", 0.0) >= threshold]
    if not filtered:
        # Fallback: return highest-scoring raw candidates so the pipeline
        # never receives an empty set because the threshold was too aggressive.
        fallback = sorted(candidates, key=lambda c: -c.get("score", 0.0))[:3]
        logger.warning(
            "threshold: no candidates ≥ %.2f (best=%.3f); falling back to top-%d raw",
            threshold,
            candidates[0].get("score", 0.0) if candidates else 0.0,
            len(fallback),
        )
        return fallback

    filtered.sort(key=lambda c: -c.get("score", 0.0))

    # Score-gap truncation: if there is a sudden drop in cosine score,
    # truncate the list to avoid including tangentially related passages.
    # Only apply when at least 3 passages precede the gap (avoids
    # over-truncation for sparse results).
    if len(filtered) > 1 and score_gap > 0:
        kept_by_score: List[Dict[str, Any]] = [filtered[0]]
        for c in filtered[1:]:
            prev_score = kept_by_score[-1].get("score", 0.0)
            curr_score = c.get("score", 0.0)
            if len(kept_by_score) >= 3 and (prev_score - curr_score) >= score_gap:
                logger.info(
                    "score_gap: truncating at %d passages (%.3f → %.3f gap=%.3f)",
                    len(kept_by_score), prev_score, curr_score, prev_score - curr_score,
                )
                break
            kept_by_score.append(c)
        filtered = kept_by_score

    result: List[Dict[str, Any]] = []
    total_tokens = 0
    for c in filtered:
        tok = int(c.get("llm_token_count", 0))
        if tok <= 0:
            # Rough approximation for Bengali text when token count is missing.
            tok = max(1, len(c.get("text", "")) // 3)
        if total_tokens + tok > budget:
            if not result:
                # First passage alone exceeds budget — include it anyway
                # (better than returning nothing).
                result.append(c)
            break
        result.append(c)
        total_tokens += tok

    return result


# ============================================================ #
# /health
# ============================================================ #

@app.get("/health")
def health():
    status: Dict[str, Any] = {}
    try:
        if qdrant_client is None:
            raise RuntimeError("client not initialised")
        cnt = qdrant_client.count(_cfg["qdrant"]["collection"]).count
        status["qdrant"] = {"status": "ok", "points": cnt}
    except Exception as e:  # noqa: BLE001
        status["qdrant"] = {"status": "error", "detail": str(e)}
    try:
        if embedder is None:
            raise RuntimeError("embedder not initialised")
        embedder.embed("health")
        status["embedder"] = {"status": "ok"}
    except Exception as e:  # noqa: BLE001
        status["embedder"] = {"status": "error", "detail": str(e)}
    status["secondary_llm"] = {
        "status": "ok" if secondary_client is not None else "unavailable",
        "model": secondary_model,
    }
    return status


# ============================================================ #
# /search — content-driven dispatch
# ============================================================ #

@app.post("/search")
async def search(req: SearchRequest):
    """Either legacy single-query or multi-query instruction-based retrieval."""
    if req.sub_queries and len(req.sub_queries) > 0:
        return await _search_multi(req)
    if req.query is not None:
        return await _search_legacy(req)
    raise HTTPException(
        status_code=400,
        detail="Either `query` (legacy) or `sub_queries` (multi-query) must be provided.",
    )


async def _search_legacy(req: SearchRequest) -> Dict[str, Any]:
    """Single-query path. Supports raw cosine or instruction+threshold+budget."""
    t0 = time.time()
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialised")

    instruction = await _build_retrieval_instruction(req)
    query_text = req.query or ""
    embed_text = f"{instruction}\n{query_text}" if instruction else query_text

    try:
        query_vec = await asyncio.to_thread(embedder.embed, embed_text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Embedding failed: {e}")

    top_k = req.top_k or _cfg["qdrant"].get("top_k", 20)
    # When using instructions we fetch more candidates so the threshold has
    # enough data to work with.
    fetch_k = retrieval_defaults.get("top_k_fetch", 50) if instruction else top_k
    try:
        hits = await asyncio.to_thread(_qdrant_topk, query_vec, fetch_k, req.chunk_type)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant search failed: {e}")

    results = [_hit_to_passage(h) for h in (hits or [])]

    if instruction:
        threshold = (
            req.cosine_threshold
            if req.cosine_threshold is not None
            else retrieval_defaults.get("cosine_threshold", 0.70)
        )
        budget = (
            req.token_budget
            if req.token_budget is not None
            else retrieval_defaults.get("token_budget", 28000)
        )
        results = _apply_threshold_and_budget(results, threshold, budget)

    elapsed = time.time() - t0
    top_score = results[0]["score"] if results else 0.0
    logger.info(
        "search[legacy]: query=%r instruction=%s hits=%d top_score=%.3f elapsed=%dms",
        (req.query or "")[:50],
        "yes" if instruction else "no",
        len(results),
        top_score,
        int(elapsed * 1000),
    )
    return {
        "query": req.query,
        "results": results,
        "hits_total": len(results),
    }


async def _embed_and_query(
    sub_idx: int,
    query: str,
    top_k: int,
    chunk_type: Optional[str] = None,
    instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Embed + Qdrant top-K for one sub-query. Annotates each hit with sub_idx."""
    if embedder is None:
        raise RuntimeError("Embedder not initialised")
    embed_text = f"{instruction}\n{query}" if instruction else query
    vec = await asyncio.to_thread(embedder.embed, embed_text)
    hits = await asyncio.to_thread(_qdrant_topk, vec, top_k, chunk_type)
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        p = _hit_to_passage(h)
        p["_sub_idx"] = sub_idx
        out.append(p)
    return out


def _merge_candidates_raw(
    per_sub: List[List[Dict[str, Any]]],
    global_cap: int,
) -> List[Dict[str, Any]]:
    """Dedupe by passage_id, keep highest cosine, track sub provenance."""
    merged: Dict[int, Dict[str, Any]] = {}
    for hits in per_sub:
        for p in hits:
            pid = int(p.get("passage_id", 0))
            if pid <= 0:
                continue
            sub_idx = int(p.get("_sub_idx", 0))
            score = float(p.get("score", 0.0))
            existing = merged.get(pid)
            if existing is None:
                merged[pid] = {
                    "passage_id": pid,
                    "text": p.get("text", ""),
                    "category": p.get("category", ""),
                    "sub_category": p.get("sub_category", ""),
                    "service": p.get("service", ""),
                    "topic": p.get("topic", ""),
                    "chunk_type": p.get("chunk_type", ""),
                    "llm_token_count": int(p.get("llm_token_count", 0)),
                    "score": score,
                    "_sub_indices": [sub_idx],
                }
            else:
                if sub_idx not in existing["_sub_indices"]:
                    existing["_sub_indices"].append(sub_idx)
                if score > existing["score"]:
                    existing["score"] = score
    out = sorted(merged.values(), key=lambda c: -c["score"])
    return out[:global_cap]


async def _search_multi(req: SearchRequest) -> Dict[str, Any]:
    """Multi-query path: parallel embed+Qdrant, merge, threshold+budget filter."""
    sub_queries = [s for s in (req.sub_queries or []) if isinstance(s, str) and s.strip()]
    if not sub_queries:
        raise HTTPException(status_code=400, detail="`sub_queries` is empty.")
    if len(sub_queries) > 8:
        raise HTTPException(status_code=400, detail="`sub_queries` exceeds limit (8).")

    if embedder is None or qdrant_client is None:
        raise HTTPException(status_code=503, detail="Service backends not ready")

    t0 = time.time()

    instruction = await _build_retrieval_instruction(req)
    use_instruction_mode = instruction is not None

    if use_instruction_mode:
        # Fetch more candidates so threshold has material to work with.
        fetch_k = retrieval_defaults.get("top_k_fetch", 50)
    else:
        fetch_k = req.top_k_per_sub or _cfg["qdrant"].get("top_k", 20)

    try:
        chunk_type = req.chunk_type
        per_sub = await asyncio.gather(*[
            _embed_and_query(i, q, fetch_k, chunk_type, instruction)
            for i, q in enumerate(sub_queries)
        ])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {e}")

    global_cap = retrieval_defaults.get("top_k_fetch", 50)
    candidates = _merge_candidates_raw(per_sub, global_cap)

    if not candidates:
        return {
            "sub_queries": sub_queries,
            "passages": [],
            "instruction": instruction,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    if use_instruction_mode:
        threshold = (
            req.cosine_threshold
            if req.cosine_threshold is not None
            else retrieval_defaults.get("cosine_threshold", 0.70)
        )
        budget = (
            req.token_budget
            if req.token_budget is not None
            else retrieval_defaults.get("token_budget", 28000)
        )
        kept = _apply_threshold_and_budget(candidates, threshold, budget)
    else:
        # Raw cosine mode: no threshold, just return merged top results.
        kept = candidates

    passages_out = [_candidate_to_passage(c) for c in kept]

    elapsed = time.time() - t0
    total_tokens = sum(c.get("llm_token_count", 0) for c in kept)
    logger.info(
        "search[multi]: subs=%d instruction=%s candidates=%d kept=%d "
        "threshold=%s budget=%s tokens_used=%d elapsed=%dms",
        len(sub_queries), "yes" if instruction else "no",
        len(candidates), len(kept),
        f"{threshold:.2f}" if use_instruction_mode else "n/a",
        f"{budget}" if use_instruction_mode else "n/a",
        total_tokens, int(elapsed * 1000),
    )
    return {
        "sub_queries": sub_queries,
        "passages": passages_out,
        "instruction": instruction,
        "elapsed_ms": int(elapsed * 1000),
    }


def _candidate_to_passage(c: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a candidate dict to the passage response shape."""
    return {
        "passage_id": c.get("passage_id", 0),
        "text": c.get("text", ""),
        "category": c.get("category", ""),
        "sub_category": c.get("sub_category", ""),
        "service": c.get("service", ""),
        "topic": c.get("topic", ""),
        "chunk_type": c.get("chunk_type", ""),
        "llm_token_count": c.get("llm_token_count", 0),
        "score": c.get("score", 0.0),
    }


# ============================================================ #
# CLI
# ============================================================ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Config file path")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    args = parser.parse_args()

    global _cfg
    _cfg = _load_config(args.config)
    _build_clients()

    port = args.port if args.port is not None else _cfg.get("port", 10000)
    host = args.host if args.host is not None else _cfg.get("host", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
