"""
cogops/tools/websearch.py

Public-web search skeleton. Off by default. Wire in a SearXNG / Tavily /
Brave-Search-style backend by setting WEBSEARCH_ENDPOINT and (optionally)
WEBSEARCH_API_KEY in the environment.

Per the system prompt's tool-selection hierarchy, this tool should be the
LAST resort — the LLM is instructed to prefer gov-services and Wikipedia first.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx

from cogops.tools.registry import ToolContext

logger = logging.getLogger(__name__)


NAME = "search_web"

DESCRIPTION = (
    "Search the public web for current events or rapidly-changing facts that "
    "are not in the government-services database or Wikipedia. Use sparingly "
    "and ONLY when the other retrieval tools return NO_RELEVANT_RESULTS. "
    "Web results are less authoritative than government sources — treat citations "
    "from this tool with extra skepticism. Bengali or English query."
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
                    "description": "The web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of result snippets to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

NO_RELEVANT_RESULTS = (
    "NO_RELEVANT_RESULTS: No web search result cleared the score threshold for this query. "
    "Per protocol, respond with the standard refusal."
)

_CONFIG: Dict[str, Any] = {
    "endpoint": None,
    "endpoint_env": "WEBSEARCH_ENDPOINT",
    "api_key_env": "WEBSEARCH_API_KEY",
    "timeout_seconds": 10.0,
    "max_results": 5,
    "min_score": 0.20,
}


def configure(cfg: Dict[str, Any]) -> None:
    for key in ("endpoint", "endpoint_env", "api_key_env",
                "timeout_seconds", "max_results", "min_score"):
        if key in cfg:
            _CONFIG[key] = cfg[key]


def _resolve_endpoint() -> str:
    if _CONFIG.get("endpoint"):
        return _CONFIG["endpoint"]
    env_key = _CONFIG.get("endpoint_env", "WEBSEARCH_ENDPOINT")
    val = os.environ.get(env_key, "")
    if not val:
        raise RuntimeError(
            f"Web-search endpoint not configured. Set env var '{env_key}' "
            f"or pass tools.websearch.endpoint in config."
        )
    return val


def _format_passage(tag: str, payload: Dict[str, Any]) -> str:
    title = payload.get("title", "")
    url = payload.get("url", "")
    text = payload.get("text", "").strip()
    header_bits = []
    if title:
        header_bits.append(f"শিরোনাম: {title}")
    if url:
        header_bits.append(f"URL: {url}")
    header = " | ".join(header_bits) if header_bits else "—"
    return f"[{tag}] ({header})\n{text}"


async def handler(
    query: str,
    ctx: ToolContext,
    max_results: int = 5,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Public web search skeleton. Adjust to the actual backend's request/response shape."""
    endpoint = _resolve_endpoint()
    timeout = float(_CONFIG.get("timeout_seconds", 10.0))
    min_score = float(_CONFIG.get("min_score", 0.20))
    api_key = os.environ.get(_CONFIG.get("api_key_env", ""), "") or None

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                endpoint,
                json={"query": query, "max_results": max_results},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.error("Web search request failed: %s", e)
        return (
            f"ERROR: Web search failed ({e.__class__.__name__}). "
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
            "text": r.get("text", "") or r.get("snippet", ""),
            "score": float(r.get("score", 0.0)),
        }
        tag = ctx.allocate_source_tag(payload)
        formatted_blocks.append(_format_passage(tag, payload))
        telemetry_sources.append({"tag": tag, **payload})

    return "\n\n".join(formatted_blocks), telemetry_sources
