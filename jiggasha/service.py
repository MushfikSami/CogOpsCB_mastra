#!/usr/bin/env python3
"""FastAPI service for Bengali passage retrieval.

POST /search (single-query only):
    { "query": "...",
      "use_instruction": true,
      "cosine_threshold": 0.70 }
  →
    { "query": "...",
      "results": [...],
      "hits_total": N,
      "instruction": "...",
      "elapsed_ms": 1234,
      "timing_ms": {...},
      "token_usage": {...} }

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
        "fallback_instruction": (retcfg.get("fallback_instruction") or "").strip(),
        "cosine_threshold": float(retcfg.get("cosine_threshold", 0.70)),
        "token_budget": int(retcfg.get("token_budget", 16000)),
        "top_k_fetch": int(retcfg.get("top_k_fetch", 50)),
        "instruction_temperature": float(retcfg.get("instruction_temperature", 0.2)),
        "instruction_max_tokens": int(retcfg.get("instruction_max_tokens", 128)),
        "instruction_timeout": float(retcfg.get("instruction_timeout_seconds", 5.0)),

    }

    # Secondary LLM config (used for dynamic instruction generation).
    secondary_cfg = _cfg.get("secondary_llm") or {}
    secondary_url = _resolve_env(secondary_cfg.get("base_url_env", "SECONDARY_BASE_URL"))
    secondary_key = _resolve_env(secondary_cfg.get("api_key_env", "SECONDARY_API_KEY"))
    secondary_model = _resolve_env(secondary_cfg.get("model_env", "SECONDARY_MODEL_NAME"))

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
    """Single-query retrieval request."""
    query: Optional[str] = None
    retrieval_instruction: Optional[str] = None
    top_k: int = 20

    # Instruction-based retrieval knobs
    use_instruction: Optional[bool] = None   # None = use config default; True/False = override
    cosine_threshold: Optional[float] = None
    token_budget: Optional[int] = None  # max cumulative tokens to return; default 28000


# ============================================================ #
# Shared helpers
# ============================================================ #

def _qdrant_topk(query_vec: List[float], top_k: int) -> List[Any]:
    """Sync Qdrant call. Caller is responsible for offloading from event loop."""
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialised")
    return qdrant_client.query_points(
        collection_name=_cfg["qdrant"]["collection"],
        query=query_vec,
        limit=top_k,
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
      3. Dynamic LLM-generated instruction (best quality)
      4. Fallback instruction from config (when LLM fails)
    """
    # 1. Caller-supplied instruction takes highest precedence.
    if req.retrieval_instruction:
        return req.retrieval_instruction.strip()

    # 2. Determine whether instruction mode is enabled.
    if req.use_instruction is False:
        return None
    if req.use_instruction is not True and not retrieval_defaults.get("use_instruction"):
        return None

    # 3. Static instruction from config — zero latency, reliable.
    static = retrieval_defaults.get("static_instruction", "")
    if static:
        return static

    # 4. Dynamic instruction from secondary LLM.
    query_text = (req.query or "").strip()
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
    if instruction:
        return instruction

    # 5. Fallback instruction when LLM fails / times out.
    fallback = retrieval_defaults.get("fallback_instruction", "")
    if fallback:
        logger.warning(
            "instruction: LLM failed; using fallback instruction for query=%r",
            query_text[:60],
        )
        return fallback
    return None


def _apply_token_budget(
    candidates: List[Dict[str, Any]],
    token_budget: int,
) -> List[Dict[str, Any]]:
    """Enforce token budget on candidates already in priority order.

    Keeps candidates in order until cumulative token count would exceed budget.
    The first candidate is always kept even if it alone exceeds the budget.
    When llm_token_count is missing/0, approximates as len(text)//3.
    """
    if not candidates:
        return []

    kept: List[Dict[str, Any]] = []
    total = 0
    for c in candidates:
        tok = int(c.get("llm_token_count", 0))
        if tok == 0:
            tok = max(1, len(c.get("text", "")) // 3)
        if kept and total + tok > token_budget:
            logger.info(
                "budget: stopped after %d passages (%d/%d tokens); dropping %d more",
                len(kept), total, token_budget, len(candidates) - len(kept),
            )
            break
        kept.append(c)
        total += tok
    return kept


def _apply_threshold_and_budget(
    candidates: List[Dict[str, Any]],
    threshold: float,
    token_budget: int,
) -> List[Dict[str, Any]]:
    """Filter by cosine threshold, sort by score desc, then enforce token budget.

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
        filtered = fallback

    filtered.sort(key=lambda c: -c.get("score", 0.0))
    return _apply_token_budget(filtered, token_budget)


def _merge_candidates_raw(
    per_sub: List[List[Dict[str, Any]]],
    global_cap: int = 50,
) -> List[Dict[str, Any]]:
    """Merge candidates from multiple sub-queries, deduplicate by passage_id."""
    best: Dict[int, Dict[str, Any]] = {}
    for sub_idx, sub_list in enumerate(per_sub):
        for c in sub_list:
            pid = int(c.get("passage_id", 0))
            if pid <= 0:
                continue
            if pid not in best:
                best[pid] = dict(c)
                best[pid].setdefault("_sub_indices", []).append(sub_idx)
            elif c.get("score", 0.0) > best[pid].get("score", 0.0):
                old_indices = best[pid].get("_sub_indices", [])
                best[pid] = dict(c)
                best[pid]["_sub_indices"] = old_indices + [sub_idx]
            else:
                best[pid].setdefault("_sub_indices", []).append(sub_idx)

    merged = sorted(best.values(), key=lambda c: -c.get("score", 0.0))
    return merged[:global_cap]


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
# /search — single-query only
# ============================================================ #

@app.post("/search")
async def search(req: SearchRequest):
    """Single-query instruction-based or raw cosine retrieval."""
    if req.query is not None:
        return await _search_legacy(req)
    raise HTTPException(
        status_code=400,
        detail="`query` must be provided.",
    )


async def _search_legacy(req: SearchRequest) -> Dict[str, Any]:
    """Single-query path: embed → Qdrant → threshold → token budget."""
    t0 = time.time()
    timing: Dict[str, int] = {}
    token_usage: Dict[str, int] = {
        "instruction_prompt": 0,
        "instruction_completion": 0,
    }

    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialised")

    t_inst0 = time.time()
    instruction = await _build_retrieval_instruction(req)
    timing["instruction"] = int((time.time() - t_inst0) * 1000)

    query_text = req.query or ""
    embed_text = f"{instruction}\n{query_text}" if instruction else query_text

    t_emb0 = time.time()
    try:
        query_vec = await asyncio.to_thread(embedder.embed, embed_text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Embedding failed: {e}")
    timing["embedding"] = int((time.time() - t_emb0) * 1000)

    top_k = req.top_k or _cfg["qdrant"].get("top_k", 20)
    fetch_k = retrieval_defaults.get("top_k_fetch", 50) if instruction else top_k

    t_qdr0 = time.time()
    try:
        hits = await asyncio.to_thread(_qdrant_topk, query_vec, fetch_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant search failed: {e}")
    timing["qdrant"] = int((time.time() - t_qdr0) * 1000)

    results = [_hit_to_passage(h) for h in (hits or [])]

    if instruction:
        threshold = (
            req.cosine_threshold
            if req.cosine_threshold is not None
            else retrieval_defaults.get("cosine_threshold", 0.70)
        )
        results = _apply_threshold_and_budget(results, threshold, token_budget=999_999_999)

    # Apply token budget universally (instruction mode or raw cosine)
    token_budget = (
        req.token_budget
        if req.token_budget is not None
        else retrieval_defaults.get("token_budget", 28000)
    )
    results = _apply_token_budget(results, token_budget)

    elapsed = time.time() - t0
    top_score = results[0]["score"] if results else 0.0
    logger.info(
        "search[legacy]: query=%r instruction=%s hits=%d top_cosine=%.3f elapsed=%dms",
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
        "instruction": instruction,
        "elapsed_ms": int(elapsed * 1000),
        "timing_ms": timing,
        "token_usage": token_usage,
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
