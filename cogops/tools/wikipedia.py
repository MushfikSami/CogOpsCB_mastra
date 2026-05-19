"""
cogops/tools/wikipedia.py

Wikipedia retrieval tool. Skeleton for the plug-in architecture — proves that
adding a new retrieval source is a one-module change plus one config line in
`tools.enabled`. POSTs to a Wikipedia search service (env: WIKI_ENDPOINT),
allocates S# tags via ctx.source_map (shared with all other retrieval tools),
and returns the same (content_for_model, sources) contract as Jiggasha.

This tool is OFF by default — not listed in configs/config.yml `tools.enabled`.
Flip it on when ready.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx

from cogops.tools.registry import ToolContext

logger = logging.getLogger(__name__)


NAME = "search_wikipedia"

DESCRIPTION = (
    "Retrieve Wikipedia passages for general-knowledge or background questions "
    "(history, geography, biography, broader context) that are NOT covered by "
    "Bangladesh government-service procedures. Do NOT use this for fees, "
    "procedures, application steps, or office contact info — use the gov-services "
    "retrieval tool for those. Bengali query preferred; English/transliteration acceptable."
)

SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Wikipedia search query (Bengali or English).",
                },
                "lang": {
                    "type": "string",
                    "enum": ["bn", "en"],
                    "default": "bn",
                    "description": "Wikipedia language edition. 'bn' for Bengali, 'en' for English.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

NO_RELEVANT_RESULTS = (
    "NO_RELEVANT_RESULTS: No Wikipedia passage cleared the score threshold for this query. "
    "Per protocol, respond with the standard refusal — do not attempt to answer."
)

_CONFIG: Dict[str, Any] = {
    "endpoint": None,
    "endpoint_env": "WIKI_ENDPOINT",
    "timeout_seconds": 8.0,
    "top_k": 5,
    "min_score": 0.30,
}


def configure(cfg: Dict[str, Any]) -> None:
    for key in ("endpoint", "endpoint_env", "timeout_seconds", "top_k", "min_score"):
        if key in cfg:
            _CONFIG[key] = cfg[key]


def _resolve_endpoint() -> str:
    if _CONFIG.get("endpoint"):
        return _CONFIG["endpoint"]
    env_key = _CONFIG.get("endpoint_env", "WIKI_ENDPOINT")
    val = os.environ.get(env_key, "")
    if not val:
        raise RuntimeError(
            f"Wikipedia endpoint not configured. Set env var '{env_key}' "
            f"or pass tools.wikipedia.endpoint in config."
        )
    return val


def _format_passage(tag: str, payload: Dict[str, Any]) -> str:
    title = payload.get("title", "")
    text = payload.get("text", "").strip()
    header = f"শিরোনাম: {title}" if title else "—"
    return f"[{tag}] ({header})\n{text}"


async def handler(
    query: str,
    ctx: ToolContext,
    lang: str = "bn",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve Wikipedia passages and tag them with S# citations.

    NOTE: This is a skeleton. The exact request/response shape depends on the
    deployed Wikipedia search service — adjust the POST body and result-field
    extraction to match. Defaults assume a {results: [{title, text, score, ...}]}
    shape similar to Jiggasha.
    """
    endpoint = _resolve_endpoint()
    timeout = float(_CONFIG.get("timeout_seconds", 8.0))
    top_k = int(_CONFIG.get("top_k", 5))
    min_score = float(_CONFIG.get("min_score", 0.30))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json={"query": query, "lang": lang, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.error("Wikipedia request failed: %s", e)
        return (
            f"ERROR: Wikipedia retrieval failed ({e.__class__.__name__}). "
            "Per protocol, respond with the standard refusal.",
            [],
        )

    raw_results = data.get("results", []) or []
    surviving = [r for r in raw_results if float(r.get("score", 0.0)) >= min_score]

    if not surviving:
        return NO_RELEVANT_RESULTS, []

    formatted_blocks: List[str] = []
    telemetry_sources: List[Dict[str, Any]] = []

    for r in surviving:
        payload = {
            "tool": NAME,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "text": r.get("text", ""),
            "score": float(r.get("score", 0.0)),
            "lang": lang,
        }
        tag = ctx.allocate_source_tag(payload)
        formatted_blocks.append(_format_passage(tag, payload))
        telemetry_sources.append({"tag": tag, **payload})

    return "\n\n".join(formatted_blocks), telemetry_sources
