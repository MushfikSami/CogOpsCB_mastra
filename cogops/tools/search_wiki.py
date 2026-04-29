import os
import logging
import httpx
from typing import Tuple, List, Union

from cogops.utils.url_media_extractor import extract_urls_media
from cogops.utils.site_checker import check_urls, replace_webpage_urls_in_text

logger = logging.getLogger(__name__)

WIKI_ENDPOINT = os.getenv(
    "WIKIPEDIA_DATA__QUERY_END_POINT",
    "http://172.22.11.241:9220/search",
)


async def search_wiki(formal_query: str, keyword_string: str) -> Tuple[Union[str, List[str]], List[str]]:
    """
    Search the Bangladesh-focused Wikipedia database for general knowledge passages.

    Args:
        formal_query: The question in formal Bengali, expressing the exact information sought.
            Example: "বর্তমানে বাংলাদেশ সরকারে রাষ্ট্রপতি হিসেবে যিনি দায়িত্ব পালন করছেন তার নাম কি ?"
        keyword_string: Space-separated Bengali keywords (3-8 words) extracted from the query.
            Example: "বর্তমান রাষ্ট্রপতি বাংলাদেশ সরকার দায়িত্ব"

    Returns:
        A tuple containing:
        - formatted text with combined_context (the answer/observation)
        - list of source references (metadata for logs)
    """
    if not formal_query or not keyword_string:
        return "No query or keywords provided.",[]

    try:
        url = _get_endpoint()
        payload = {
            "formal_query": formal_query,
            "keyword_string": keyword_string,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            return f"Wiki search failed (HTTP {resp.status_code}): {resp.text}",[]

        data = resp.json()
        combined = data.get("combined_context", "")
        results = data.get("results",[])

        if not combined and not results:
            return "No relevant results found.",[]

        # Format: show combined_context first (what the agent sees as observation)
        context = [f"Query: {formal_query}"]
        if combined:
            context.append(f"Context:\n{combined}")

        # Collect result URLs for sources
        sources = []
        # Map section titles (after ###) to page URLs
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

        # Split combined_context by ### headers and inject Media Links per section.
        if combined and title_to_url:
            try:
                import re as _re
                sections = _re.split(r'(?=\n### )', combined.strip())
                all_page_urls = list(title_to_url.values())

                new_context_parts: list[str] = []
                seen_images: set[str] = set()

                for section in sections:
                    section = section.strip()
                    if not section:
                        continue

                    # Extract page URL for this section by matching ### title
                    section_title_match = _re.match(r'###\s*(.+)', section)
                    section_title = section_title_match.group(1).strip() if section_title_match else ""
                    # Try exact match first, then partial match
                    page_url = title_to_url.get(section_title)
                    if not page_url:
                        # Try stripping common disambiguation suffixes
                        cleaned = _re.sub(r'\s*\(বাংলাদেশ\)\s*$', '', section_title)
                        page_url = title_to_url.get(cleaned)
                    if not page_url:
                        # Try substring: does any result title appear in the section text?
                        for rtitle, rurl in title_to_url.items():
                            if rtitle in section:
                                page_url = rurl
                                break

                    # Extract wiki file refs from this section only
                    section_extracted: list[dict] = []
                    if page_url:
                        entry_text = section + "\n" + page_url
                        section_extracted = extract_urls_media(entry_text, source="wikipedia")

                    # Check status for this section's extracted items
                    if section_extracted:
                        media_items = await check_urls(section_extracted)
                        # Replace webpage URLs in section with verified versions
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
                            extra = ""
                            if "redirect_to" in m:
                                extra = f" (redirects to {m['redirect_to']})"
                            media_line = f"- [{typ}] {u} — **{status}**{extra}"
                            section += "\n\n## Media Links\n" + media_line

                    new_context_parts.append(section)

                # Rebuild context with per-section Media Links
                context = [f"Query: {formal_query}"]
                if new_context_parts:
                    context.append("Context:\n" + "\n".join(new_context_parts))
            except Exception as e:
                logger.debug("URL extraction failed: %s", e)

        return context, sources

    except Exception as e:
        logger.error(f"search_wiki error: {e}")
        return f"Wiki search error: {e}",[]


def _get_endpoint() -> str:
    return os.getenv(
        "WIKIPEDIA_DATA__QUERY_END_POINT",
        "http://172.22.11.241:9220/search",
    )


search_wiki_tools_list =[
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": (
                "Search the Bangladesh-focused Wikipedia database for general knowledge. "
                "Use for questions about Bangladesh (history, geography, politics, policy, "
                "current events), world events, public figures, or general knowledge not "
                "specific to government procedures. This is the fallback when "
                "search_knowledge returns no results for a government service query, and "
                "the primary choice for non-government general knowledge questions.\n\n"
                "Parameters Guidelines:\n"
                "When a user asks a colloquial, informal, or English-mixed question, you MUST translate it into formal Wikipedia terminology.\n"
                "For example, 'আইসিটি মিনিস্ট্রি' should be translated to the formal 'তথ্য ও যোগাযোগ প্রযুক্তি মন্ত্রনালয়'.\n\n"
                "Furthermore, you must infer implicit context and use standard Wikipedia phrasing.\n"
                "For example, if the user asks: 'বাংলাদেশের প্রেসিডেন্ট কে?'\n"
                "- formal_query: Write the exact question representing the absolute information sought, using formal words and tones aware of Wikipedia structures. "
                "Example translation: 'বর্তমানে বাংলাদেশ সরকারে রাষ্ট্রপতি হিসেবে যিনি দায়িত্ব পালন করছেন তার নাম কি ?'\n"
                "- keyword_string: Provide a space-separated list of words related to the query. "
                "Since the question does not specify a past date, it means the CURRENT president, so we include 'বর্তমান'. "
                "Also, Wikipedia pages often use phrases like 'তিনি বর্তমানে অমুক পদে দায়িত্ব পালন করছেন', so we include the word 'দায়িত্ব'. "
                "Example translation: 'বর্তমান রাষ্ট্রপতি বাংলাদেশ সরকার দায়িত্ব'\n\n"
                "Note: Pipe-separated keywords also work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formal_query": {
                        "type": "string",
                        "description": (
                            "The exact question representing the absolute information sought. "
                            "Write in proper formal Bengali (বাংলা) mimicking the tone and nature of Wikipedia pages. "
                            "Example: 'বর্তমানে বাংলাদেশ সরকারে রাষ্ট্রপতি হিসেবে যিনি দায়িত্ব পালন করছেন তার নাম কি ?'"
                        ),
                    },
                    "keyword_string": {
                        "type": "string",
                        "description": (
                            "Space-separated Bengali keywords (3-8 words) related to the query. "
                            "Include implicit contextual words (like 'বর্তমান') and typical Wikipedia terminology (like 'দায়িত্ব'). "
                            "Example: 'বর্তমান রাষ্ট্রপতি বাংলাদেশ সরকার দায়িত্ব'"
                        ),
                    },
                },
                "required": ["formal_query", "keyword_string"],
            },
        },
    }
]

search_wiki_tools_map = {
    "search_wiki": search_wiki,
}