#!/usr/bin/env python3
"""FastAPI service for Bengali passage retrieval.

POST /search has two content-driven modes:

  Legacy single-query:
      { "query": "...", "top_k": 20 }
    →
      { "query": "...", "results": [...], "hits_total": N }

  Multi-query rerank:
      { "sub_queries": ["...", "..."], "top_k_per_sub": 20,
        "rerank": true, "candidate_cap_global": 30,
        "keep_cap": 24, "weak_per_sub_cap": 3,
        "fallback_cosine_min": 0.50 }
    →
      { "sub_queries": [...],
        "passages": [...],
        "rerank": {"1": [[pid, cls], ...], "2": [...], ...},
        "degraded": false }

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
from rerank import RerankCandidate, RerankResult, run_rerank

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
rerank_defaults: Dict[str, Any] = {}


def _build_clients() -> None:
    """(Re)build embedder, Qdrant client, secondary LLM client from _cfg."""
    global embedder, qdrant_client, secondary_client, secondary_model, rerank_defaults

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

    rcfg = _cfg.get("rerank", {}) or {}
    rerank_defaults = {
        "timeout": float(rcfg.get("timeout_seconds", 30)),
        "keep_cap": int(rcfg.get("keep_cap", 24)),
        "weak_per_sub_cap": int(rcfg.get("weak_per_sub_cap", 3)),
        "candidate_cap_global": int(rcfg.get("candidate_cap_global", 30)),
        "fallback_cosine_min": float(rcfg.get("fallback_cosine_min", 0.50)),
    }

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
                "rerank: secondary LLM ready at %s (model=%s)",
                secondary_url, secondary_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to init secondary LLM client: %s", e)
            secondary_client = None
    else:
        secondary_client = None
        logger.warning(
            "rerank: no secondary LLM configured; multi-query path will use cosine safety net",
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

    # Multi-query rerank
    sub_queries: Optional[List[str]] = None
    top_k_per_sub: int = 20
    rerank: bool = False
    candidate_cap_global: Optional[int] = None
    keep_cap: Optional[int] = None
    weak_per_sub_cap: Optional[int] = None
    fallback_cosine_min: Optional[float] = None
    rerank_timeout: Optional[float] = None
    chunk_type: Optional[str] = None   # "wiki" | "govt_service" | null


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
        "score": float(hit.score) if hasattr(hit, "score") and hit.score is not None else 0.0,
    }


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
    """Either legacy single-query or multi-query rerank, by presence of `sub_queries`."""
    if req.sub_queries and len(req.sub_queries) > 0:
        return await _search_multi(req)
    if req.query is not None:
        return await _search_legacy(req)
    raise HTTPException(
        status_code=400,
        detail="Either `query` (legacy) or `sub_queries` (multi-query) must be provided.",
    )


async def _search_legacy(req: SearchRequest) -> Dict[str, Any]:
    """Single-query path. Returns raw cosine top-K (no filtering)."""
    t0 = time.time()
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialised")

    try:
        query_vec = await asyncio.to_thread(embedder.embed, req.query or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Embedding failed: {e}")

    top_k = req.top_k or _cfg["qdrant"].get("top_k", 20)
    try:
        hits = await asyncio.to_thread(_qdrant_topk, query_vec, top_k, req.chunk_type)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant search failed: {e}")

    results = [_hit_to_passage(h) for h in (hits or [])]
    elapsed = time.time() - t0
    top_score = results[0]["score"] if results else 0.0
    logger.info(
        "search[legacy]: query=%r hits=%d top_score=%.3f elapsed=%dms",
        (req.query or "")[:50], len(results), top_score, int(elapsed * 1000),
    )
    return {
        "query": req.query,
        "results": results,
        "hits_total": len(results),
    }


async def _embed_and_query(
    sub_idx: int, query: str, top_k: int, chunk_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Embed + Qdrant top-K for one sub-query. Annotates each hit with sub_idx."""
    if embedder is None:
        raise RuntimeError("Embedder not initialised")
    vec = await asyncio.to_thread(embedder.embed, query)
    hits = await asyncio.to_thread(_qdrant_topk, vec, top_k, chunk_type)
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        p = _hit_to_passage(h)
        p["_sub_idx"] = sub_idx
        out.append(p)
    return out


def _merge_candidates(
    per_sub: List[List[Dict[str, Any]]],
    global_cap: int,
) -> List[RerankCandidate]:
    """Dedupe by passage_id, keep highest cosine, track sub provenance."""
    merged: Dict[int, RerankCandidate] = {}
    for hits in per_sub:
        for p in hits:
            pid = int(p.get("passage_id", 0))
            if pid <= 0:
                continue
            sub_idx = int(p.get("_sub_idx", 0))
            score = float(p.get("score", 0.0))
            existing = merged.get(pid)
            if existing is None:
                merged[pid] = RerankCandidate(
                    passage_id=pid,
                    text=p.get("text", ""),
                    score=score,
                    category=p.get("category", ""),
                    sub_category=p.get("sub_category", ""),
                    service=p.get("service", ""),
                    topic=p.get("topic", ""),
                    chunk_type=p.get("chunk_type", ""),
                    sub_indices=[sub_idx],
                )
            else:
                if sub_idx not in existing.sub_indices:
                    existing.sub_indices.append(sub_idx)
                if score > existing.score:
                    existing.score = score
    out = sorted(merged.values(), key=lambda c: -c.score)
    return out[:global_cap]


async def _search_multi(req: SearchRequest) -> Dict[str, Any]:
    """Multi-query rerank path: parallel embed+Qdrant, merge, LLM rerank."""
    sub_queries = [s for s in (req.sub_queries or []) if isinstance(s, str) and s.strip()]
    if not sub_queries:
        raise HTTPException(status_code=400, detail="`sub_queries` is empty.")
    if len(sub_queries) > 8:
        raise HTTPException(status_code=400, detail="`sub_queries` exceeds limit (8).")

    if embedder is None or qdrant_client is None:
        raise HTTPException(status_code=503, detail="Service backends not ready")

    t0 = time.time()
    top_k = req.top_k_per_sub or _cfg["qdrant"].get("top_k", 20)
    try:
        chunk_type = req.chunk_type
        per_sub = await asyncio.gather(*[
            _embed_and_query(i, q, top_k, chunk_type) for i, q in enumerate(sub_queries)
        ])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {e}")

    global_cap = req.candidate_cap_global or rerank_defaults["candidate_cap_global"]
    candidates = _merge_candidates(per_sub, global_cap)

    if not candidates:
        return {
            "sub_queries": sub_queries,
            "passages": [],
            "rerank": {str(i + 1): [] for i in range(len(sub_queries))},
            "degraded": False,
        }

    if not req.rerank:
        # Caller wants raw merged candidates with per-sub provenance from
        # Qdrant only; no LLM call. Use the cosine safety net to shape the
        # response (everything tagged weak). Not flagged as degraded.
        from rerank import _cosine_safety_net  # type: ignore
        kept, per_q = _cosine_safety_net(
            candidates,
            n_subs=len(sub_queries),
            fallback_cosine_min=0.0,    # do not filter; the caller asked for raw
            keep_cap=req.keep_cap or rerank_defaults["keep_cap"],
        )
        passages_out = [_candidate_to_passage(c) for c in kept]
        return {
            "sub_queries": sub_queries,
            "passages": passages_out,
            "rerank": per_q,
            "degraded": False,
        }

    result: RerankResult = await run_rerank(
        sub_queries=sub_queries,
        candidates=candidates,
        secondary_client=secondary_client,
        secondary_model=secondary_model,
        timeout=req.rerank_timeout or rerank_defaults["timeout"],
        keep_cap=req.keep_cap or rerank_defaults["keep_cap"],
        weak_per_sub_cap=req.weak_per_sub_cap or rerank_defaults["weak_per_sub_cap"],
        fallback_cosine_min=(
            req.fallback_cosine_min
            if req.fallback_cosine_min is not None
            else rerank_defaults["fallback_cosine_min"]
        ),
    )

    passages_out = [_candidate_to_passage(c) for c in result.passages]

    elapsed = time.time() - t0
    yes_total = sum(
        1 for v in result.per_query.values() for _, cls in v if cls == 0
    )
    weak_total = sum(
        1 for v in result.per_query.values() for _, cls in v if cls == 1
    )
    logger.info(
        "search[multi]: subs=%d candidates=%d kept=%d (yes_votes=%d weak_votes=%d) degraded=%s elapsed=%dms",
        len(sub_queries), len(candidates), len(result.passages),
        yes_total, weak_total, result.degraded, int(elapsed * 1000),
    )
    return {
        "sub_queries": sub_queries,
        "passages": passages_out,
        "rerank": result.per_query,
        "degraded": result.degraded,
        "rerank_usage": result.usage,
        "elapsed_ms": int(elapsed * 1000),
    }


def _candidate_to_passage(c: RerankCandidate) -> Dict[str, Any]:
    return {
        "passage_id": c.passage_id,
        "text": c.text,
        "category": c.category,
        "sub_category": c.sub_category,
        "service": c.service,
        "topic": c.topic,
        "chunk_type": c.chunk_type,
        "score": c.score,
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
