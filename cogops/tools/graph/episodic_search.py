"""
cogops/tools/graph/episodic_search.py

Search raw passage data in Episodic nodes by text content or metadata fields.
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from cogops.config.loader import load_config, get_tool_config

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)


async def episodic_search(search_term: str, max_results: int = 10) -> str:
    """Search raw passage data in Episodic nodes."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver
    cfg = get_tool_config(CONFIG, 'episodic_search')
    max_results = max_results or cfg.get('max_results', 10)

    md = f"## Episodic Search: '{search_term}'\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            f"MATCH (ep:Episodic) "
            f"WHERE toLower(ep.content) CONTAINS toLower($term) "
            f"RETURN ep.content "
            f"ORDER BY ep.created_at DESC "
            f"LIMIT $maxResults",
            term=search_term,
            maxResults=max_results
        )
        records = await result.data()

    if not records:
        md += f"No passages found for '{search_term}'.\n"
        return md

    import json
    md += f"Found {len(records)} passages:\n\n"
    for i, r in enumerate(records, 1):
        raw = r.get("ep.content", "{}")
        try:
            json_content = json.loads(raw) if isinstance(raw, str) else raw
            text = json_content.get("text", "")[:300]
            category = json_content.get("category", "")
        except Exception:
            text = str(raw)[:300]
            category = ""

        md += f"### Passage {i}\n"
        if category:
            md += f"**Category:** {category}\n"
        md += f"**Text:** {text}...\n\n"

    return md


episodic_search_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "episodic_search",
            "description": "Search raw passage data in Episodic nodes by text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search term."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 10)."
                    }
                },
                "required": ["search_term"]
            }
        }
    }
]

episodic_search_tools_map = {
    "episodic_search": episodic_search
}
