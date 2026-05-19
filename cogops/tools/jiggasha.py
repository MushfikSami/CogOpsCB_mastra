"""
cogops/tools/jiggasha.py

Bangladesh government-services retrieval tool. POSTs the user's query to the
Jiggasha service (Qdrant + reranker), filters by minimum reranker score,
allocates monotonic S# tags from the per-turn ToolContext.source_map, and
returns a Bengali-friendly formatted block plus a structured sources list
for debug telemetry.

When no passage clears the score threshold, returns the sentinel string
NO_RELEVANT_RESULTS — the system prompt instructs the LLM to refuse on this.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx

from cogops.tools.registry import ToolContext

logger = logging.getLogger(__name__)


NAME = "search_gov_services"

DESCRIPTION = (
    "Retrieve authoritative Bangladesh government-service passages "
    "(fees, procedures, eligibility, office contacts, document checklists, "
    "ministry/department information). Use this for ANY factual question about "
    "Bangladesh government services — NID, passport, BRTA, tax, land records, "
    "certificates, licenses, citizen services. The query should be in Bengali "
    "(English proper nouns are acceptable). Returns numbered passages with "
    "[S#] tags that you MUST cite verbatim in your answer."
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
                    "description": (
                        "The Bengali question to retrieve passages for. "
                        "Pass the user's exact question or a focused reformulation."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

NO_RELEVANT_RESULTS = (
    "NO_RELEVANT_RESULTS: No Jiggasha passage cleared the score threshold for this query. "
    "Per protocol, respond with the standard Bengali refusal — do not attempt to answer."
)

_CONFIG: Dict[str, Any] = {
    "endpoint": None,        # resolved at call-time from env or configure()
    "endpoint_env": "JIGGASHA_ENDPOINT",
    "timeout_seconds": 8.0,
    "top_k": 5,
    "min_score": 0.35,
}


def configure(cfg: Dict[str, Any]) -> None:
    """Optional config hook called by build_tool_registry() with per-tool config."""
    for key in ("endpoint", "endpoint_env", "timeout_seconds", "top_k", "min_score"):
        if key in cfg:
            _CONFIG[key] = cfg[key]


def _resolve_endpoint() -> str:
    """Resolve the Jiggasha endpoint URL. Explicit cfg > env var > error."""
    if _CONFIG.get("endpoint"):
        return _CONFIG["endpoint"]
    env_key = _CONFIG.get("endpoint_env", "JIGGASHA_ENDPOINT")
    val = os.environ.get(env_key, "")
    if not val:
        raise RuntimeError(
            f"Jiggasha endpoint not configured. Set env var '{env_key}' "
            f"or pass tools.jiggasha.endpoint in config."
        )
    return val


def _format_passage_for_model(tag: str, payload: Dict[str, Any]) -> str:
    """Render a single passage block the model sees, with its [S#] tag embedded."""
    category = payload.get("category", "")
    sub_category = payload.get("sub_category", "")
    service = payload.get("service", "")
    topic = payload.get("topic", "")
    text = payload.get("text", "").strip()

    header_bits = []
    if category:
        header_bits.append(f"বিভাগ: {category}")
    if sub_category:
        header_bits.append(f"উপ-বিভাগ: {sub_category}")
    if service:
        header_bits.append(f"সেবা: {service}")
    if topic:
        header_bits.append(f"বিষয়: {topic}")
    header = " | ".join(header_bits) if header_bits else "—"

    return f"[{tag}] ({header})\n{text}"


async def handler(query: str, ctx: ToolContext) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve government-service passages and tag them with S# citations.

    Returns:
        (content_for_model, sources)
        content_for_model: Bengali-formatted passage blocks with [S#] tags,
            or NO_RELEVANT_RESULTS sentinel if nothing clears the threshold.
        sources: list of dicts mirroring source_map entries (for telemetry).
    """
    endpoint = _resolve_endpoint()
    timeout = float(_CONFIG.get("timeout_seconds", 8.0))
    top_k = int(_CONFIG.get("top_k", 5))
    min_score = float(_CONFIG.get("min_score", 0.35))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json={"query": query, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.error("Jiggasha request failed: %s", e)
        return (
            f"ERROR: Jiggasha retrieval failed ({e.__class__.__name__}). "
            "Per protocol, respond with the standard Bengali refusal.",
            [],
        )

    raw_results = data.get("results", []) or []
    surviving = [r for r in raw_results if float(r.get("score", 0.0)) >= min_score]

    if not surviving:
        logger.info(
            "Jiggasha: no result cleared min_score=%.2f for query=%r (got %d raw)",
            min_score, query[:80], len(raw_results),
        )
        return NO_RELEVANT_RESULTS, []

    formatted_blocks: List[str] = []
    telemetry_sources: List[Dict[str, Any]] = []

    for r in surviving:
        payload = {
            "tool": NAME,
            "passage_id": r.get("passage_id"),
            "category": r.get("category", ""),
            "sub_category": r.get("sub_category", ""),
            "service": r.get("service", ""),
            "topic": r.get("topic", ""),
            "text": r.get("text", ""),
            "score": float(r.get("score", 0.0)),
        }
        tag = ctx.allocate_source_tag(payload)
        formatted_blocks.append(_format_passage_for_model(tag, payload))
        telemetry_sources.append({"tag": tag, **payload})

    content = "\n\n".join(formatted_blocks)
    logger.info(
        "Jiggasha: returned %d passage(s) for query=%r (filtered from %d)",
        len(surviving), query[:80], len(raw_results),
    )
    return content, telemetry_sources
