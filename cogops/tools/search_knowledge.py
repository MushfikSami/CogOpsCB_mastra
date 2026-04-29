import os
import logging
import httpx
from typing import Tuple, List, Union

from cogops.utils.url_media_extractor import extract_urls_media
from cogops.utils.site_checker import check_urls, replace_webpage_urls_in_text

from cogops.config.loader import load_config

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = os.getenv(
    "JIGGASHA_DATA__QUERY_END_POINT",
    "http://172.22.11.241:9210/search",
)

# Cached config for top_k default
_knowledge_config: dict | None = None


def _get_knowledge_config() -> dict:
    global _knowledge_config
    if _knowledge_config is None:
        _knowledge_config = load_config()
    return _knowledge_config


def _default_top_k() -> int:
    return (
        _get_knowledge_config()
        .get("knowledge_search", {})
        .get("top_k_default", 20)
    )


async def search_knowledge(formal_query: str, keyword_string: str) -> Tuple[Union[str, List[str]], List[str]]:
    """
    Search the Bangladesh government service database (Jiggasha) for relevant passages.

    Args:
        formal_query: The exact question in formal Bengali (বাংলা), expressing the
            information needed. Use proper Bengali vocabulary as on an official form.
            Example: "এসএসসি সার্টিফিকেটে নিজের নাম পরিবর্তন করতে ঢাকা শিক্ষা বোর্ডে কত টাকা জমা দিতে হয় ?"
        keyword_string: Space-separated Bengali keywords (3-8 words) extracted from the query.
            These are the key terms that appear in the database text.
            Example: "এসএসসি সার্টিফিকেট নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি"

    Returns:
        A tuple containing:
        - formatted text with combined_context (the answer/observation)
        - list of source node paths (metadata for logs)
    """
    if not formal_query or not keyword_string:
        return "No query or keywords provided.",[]

    top_k = _default_top_k()

    try:
        url = _get_endpoint()
        payload = {
            "formal_query": formal_query,
            "keyword_string": keyword_string,
            "top_k": top_k,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            return f"Search failed (HTTP {resp.status_code}): {resp.text}",[]

        data = resp.json()
        combined = data.get("combined_context", "")
        results = data.get("results",[])

        if not combined and not results:
            return "No relevant results found.",[]

        # Format: show combined_context first (what the agent sees as observation)
        context = [f"Query: {formal_query}"]
        if combined:
            context.append(f"Retrieved Context:\n{combined}")

        # Add results metadata for reference
        sources = []
        if results:
            sources =[r.get("node", "") for r in results]

        # Extract URLs and check status, append as "Media Links" to context
        all_text = "\n".join(context)
        try:
            extracted = extract_urls_media(all_text, source="jiggasha")
            if extracted:
                media_items = await check_urls(extracted)
                # Replace webpage URLs in context with verified versions
                all_text = replace_webpage_urls_in_text(all_text, extracted, media_items)
                # Rebuild context parts from the replaced text.
                # Context was: [Query: ..., Retrieved Context:\ncombined...]
                parts = all_text.split("\n", 1)
                query_part = parts[0] if parts[0].startswith("Query:") else f"Query: {formal_query}"
                context = [query_part, parts[1]] if len(parts) > 1 else [query_part]
                media_lines = ["\n## Media Links"]
                for m in media_items:
                    status = m.get("status", "?")
                    u = m.get("url", "")
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
        logger.error(f"search_knowledge error: {e}")
        return f"Search error: {e}",[]


def _get_endpoint() -> str:
    return os.getenv(
        "JIGGASHA_DATA__QUERY_END_POINT",
        "http://172.22.11.241:9210/search",
    )


search_knowledge_tools_list =[
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the Bangladesh government service database (Jiggasha) for relevant "
                "passages. Use for ANY question about Bangladesh government services, procedures, "
                "fees, document requirements, offices, boards, departments, or regulations. "
                "Covers 30+ services: education (শিক্ষা), passports (পাসপোর্ট), "
                "NID (জাতীয় পরিচয়পত্র), birth/death registration (জন্ম/মৃত্যু নিবন্ধন), "
                "land (ভূমি), trade licenses (ট্রেড লাইসেন্স), vehicles (যানবাহন), "
                "utilities (ইউটিলিটি), pensions (পেনশন), disaster management (দূর্যোগ "
                "ব্যবস্থাপনা), social safety (সামাজিক সুরক্ষা), law and security (আইন ও "
                "নিরাপত্তা), health (স্বাস্থ্য) and more.\n\n"
                "Parameters Guidelines:\n"
                "When you want an answe which comes from an informal or colloquial question, you MUST translate it into official government terminology.\n"
                "For example, if the user asks: 'এস এস সি তে আমার নাম ভুল আসছে । সার্টিফিকেটের নাম উলটা পালটা আসছে । আমি ঢাকা বোর্ডে পরীক্ষা দিছিলাম । এখন নাম চেঞ্জ করা লাগবে । টেকা টুকা লাগবে নাকি আবার ?'\n"
                "or there were several turns - like "
                "Or if there were several turns - like:\n"
                "User: সার্টফিকেটে আমার নাম ভুল আসছে\n"
                "AI: আপনি কোন সার্টিফিকেট এর ব্যাপারে বলছেন অনুগ্রহ করে পরিষ্কার করুন\n"
                "User: ssc\n"
                "AI: এসএসসি সার্টিফিকেটের নাম সংশোধন প্রক্রিয়া শিক্ষা বোর্ডের সাথে পরিবর্তিত হতে পারে। আপনি কোন শিক্ষা বোর্ড থেকে এই নাম সংশোধন সম্পর্কে জানতে চান?\n"
                "User: ঢাকা বোর্ড , ওরা টাকা কত নেয় ?\n\n"
                "Here, now that you have clarification, you can use this tool:\n"
                "- formal_query: Write the exact question you need answered in formal Bengali (বাংলা). "
                "Use proper Bengali vocabulary — as you would write on an official government form. "
                "It represents the absolute information that you are looking for. "
                "Example translation: 'এসএসসি সার্টিফিকেটে নিজের নাম পরিবর্তন করতে ঢাকা শিক্ষা বোর্ডে কত টাকা জমা দিতে হয় ?'\n"
                "- keyword_string: Space-separated Bengali keywords extracted from your "
                "query. These are the key terms that would appear in the government database text (mostly .gov.bd). "
                "Example translation: 'এসএসসি সার্টিফিকেট নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি'\n\n"
                "If this tool returns no results call search_wiki as your next action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formal_query": {
                        "type": "string",
                        "description": (
                            "The exact question representing the absolute information sought. "
                            "Write in proper formal Bengali (বাংলা) as it would appear on a government document. "
                            "Example: 'এসএসসি সার্টিফিকেটে নিজের নাম পরিবর্তন করতে ঢাকা শিক্ষা বোর্ডে কত টাকা জমা দিতে হয় ?'"
                        ),
                    },
                    "keyword_string": {
                        "type": "string",
                        "description": (
                            "Space-separated Bengali keywords (3-8 words) related to the query. "
                            "These words should be terms that exist in the government database text as they "
                            "appear together in a related document. "
                            "Example: 'এসএসসি সার্টিফিকেট নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি'"
                        ),
                    },
                },
                "required": ["formal_query", "keyword_string"],
            },
        },
    }
]

search_knowledge_tools_map = {
    "search_knowledge": search_knowledge,
}