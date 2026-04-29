import os
import logging
import httpx
from typing import Tuple, List, Union

from cogops.utils.url_media_extractor import extract_urls_media
from cogops.utils.site_checker import check_urls

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

        # Collect result URLs for Media Links extraction and sources
        sources = []
        result_urls = []
        if results:
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                pub = r.get("published_at", "")
                sources.append(
                    f"{i}. [{title}]({url})" + (f" (updated: {pub})" if pub else "")
                )
                if url:
                    result_urls.append(url)

        # Extract URLs and check status, append as "Media Links" to context
        # Include result URLs in text so wiki file refs can be resolved
        all_text = "\n".join(context + result_urls)
        try:
            extracted = extract_urls_media(all_text, source="wikipedia")
            if extracted:
                media_items = await check_urls(extracted)
                media_lines = ["\n## Media Links"]
                for m in media_items:
                    status = m.get("status", "?")
                    u = m.get("url", "")
                    # Skip items that aren't real URLs
                    if not u.startswith("http://") and not u.startswith("https://"):
                        continue
                    typ = m.get("type", "?")
                    extra = ""
                    if "redirect_to" in m:
                        extra = f" (redirects to {m['redirect_to']})"
                    media_lines.append(f"- [{typ}] {u} — **{status}**{extra}")
                context.append("\n".join(media_lines))
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