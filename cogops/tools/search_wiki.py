"""
cogops/tools/search_wiki.py

Search tool for Bangladesh-focused Wikipedia database.
Simplified interface: natural language query only (no formal_query/keyword_split).
URL extraction and site health checking preserved from original.
"""

import os
import re
import logging
import httpx
from typing import Tuple, List, Union

from cogops.utils.url_media_extractor import extract_urls_media
from cogops.utils.site_checker import check_urls, replace_webpage_urls_in_text

logger = logging.getLogger(__name__)


def _get_endpoint() -> str:
    from cogops.config.loader import load_config
    try:
        cfg = load_config()
        env_name = cfg.get("wiki", {}).get("endpoint_env", "WIKI_ENDPOINT")
        default = cfg.get("wiki", {}).get("endpoint_default", "http://172.22.11.241:9220/search")
        return os.getenv(env_name, default)
    except Exception:
        return os.getenv("WIKI_ENDPOINT", "http://172.22.11.241:9220/search")


def _get_timeout() -> float:
    from cogops.config.loader import load_config
    try:
        cfg = load_config()
        return cfg.get("wiki", {}).get("timeout", 30)
    except Exception:
        return 30.0


async def search_wiki(query: str, **_injectable) -> Tuple[Union[str, List[str]], List[str]]:
    """
    Search the Bangladesh-focused Wikipedia database for general knowledge.

    Args:
        query: Natural language query in Bengali or English.
        **_injectable: Server-side injected params — unused.

    Returns:
        (formatted_text_for_model, sources_list)
    """
    if not query or not query.strip():
        return "No query provided.", []

    timeout = _get_timeout()

    try:
        payload = {
            "formal_query": query.strip(),
            "keyword_string": query.strip(),
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(_get_endpoint(), json=payload)

        if resp.status_code != 200:
            return f"Wiki search failed (HTTP {resp.status_code}): {resp.text[:200]}", []

        data = resp.json()
        combined = data.get("combined_context", "")
        results = data.get("results", [])

        if not combined and not results:
            return "No relevant information found in Wikipedia.", []

        # Format context
        context_parts = [f"Query: {query.strip()}"]
        if combined:
            context_parts.append(f"Context:\n{combined}")

        # Collect sources and media links
        sources = []
        title_to_url: dict[str, str] = {}
        if results:
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                pub = r.get("published_at", "")
                sources.append(
                    f"{i}. [{title}]({url})" + (f" (updated: {pub})" if pub else "")
                )
                if url:
                    title_to_url[title] = url

        # Inject Media Links per section
        if combined and title_to_url:
            try:
                sections = re.split(r'(?=\n### )', combined.strip())
                new_context_parts: list[str] = []
                seen_images: set[str] = set()

                for section in sections:
                    section = section.strip()
                    if not section:
                        continue

                    section_title_match = re.match(r'###\s*(.+)', section)
                    section_title = section_title_match.group(1).strip() if section_title_match else ""

                    # Find page URL for this section
                    page_url = title_to_url.get(section_title)
                    if not page_url:
                        cleaned = re.sub(r'\s*\(বাংলাদেশ\)\s*$', '', section_title)
                        page_url = title_to_url.get(cleaned)
                    if not page_url:
                        for rtitle, rurl in title_to_url.items():
                            if rtitle in section:
                                page_url = rurl
                                break

                    # Extract and check URLs
                    if page_url:
                        section_extracted = extract_urls_media(section + "\n" + page_url, source="wikipedia")
                        if section_extracted:
                            media_items = await check_urls(section_extracted)
                            section = replace_webpage_urls_in_text(section, section_extracted, media_items)
                            for m in media_items:
                                status = m.get("status", "?")
                                u = m.get("url", "")
                                if not u.startswith("http://") and not u.startswith("https://"):
                                    continue
                                key = u.lower()
                                if key in seen_images:
                                    continue
                                seen_images.add(key)
                                typ = m.get("type", "?")
                                extra = f" (redirects to {m['redirect_to']})" if "redirect_to" in m else ""
                                section += f"\n\n## Media Links\n- [{typ}] {u} — **{status}**{extra}"

                    new_context_parts.append(section)

                context_parts = [f"Query: {query.strip()}"]
                if new_context_parts:
                    context_parts.append("Context:\n" + "\n".join(new_context_parts))
            except Exception as e:
                logger.debug("URL extraction failed: %s", e)

        return context_parts, sources

    except httpx.TimeoutException:
        logger.warning("Wiki search timed out: %s", query[:50])
        return "Wiki search timed out.", []
    except Exception as e:
        logger.error("search_wiki error: %s", e)
        return f"Wiki search error: {e}", []


# --- Lean Tool Schema ---
search_wiki_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": (
                "Search the Bangladesh-focused Wikipedia database for general knowledge. "
                "Use for questions about Bangladesh (history, geography, politics, current events), "
                "world events, public figures, or general knowledge not specific to government procedures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query in Bengali or English.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

search_wiki_tools_map = {
    "search_wiki": search_wiki,
}
