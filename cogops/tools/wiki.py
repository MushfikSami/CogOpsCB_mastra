"""
cogops/tools/wiki.py

Wikipedia search fallback. Used when Graphiti/graph tools return no data
for a factual query that might exist in general-knowledge sources. All
queries are forced into the Bangladesh context by appending "বাংলাদেশ"
so the results remain relevant to this agent's domain.

Three tools are exposed to the model:
- wikipedia_search          → list top-N page titles (snippet + url + last-edited)
- wikipedia_get_summary     → intro paragraph of a specific page
- wikipedia_get_full_content → full plaintext of a specific page (capped)

All three are async wrappers around synchronous `requests` calls; the
reasoning loop runs them via asyncio.to_thread when needed.
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from cogops.config.loader import load_config, get_tool_config

load_dotenv()

logger = logging.getLogger(__name__)

_CONFIG = load_config()

_DEFAULT_BASE_URL = "https://bn.wikipedia.org/w/api.php"
_DEFAULT_SEARCH_LIMIT = int(
    _CONFIG.get("wiki", {}).get("search_limit", 5)
)
_DEFAULT_CONTENT_CAP = int(
    _CONFIG.get("wiki", {}).get("content_char_cap", 5000)
)
_STALE_YEARS = int(_CONFIG.get("wiki", {}).get("stale_years", 2))
_HTTP_TIMEOUT_SECONDS = int(_CONFIG.get("wiki", {}).get("http_timeout_seconds", 8))

_BASE_URL = os.getenv("WIKI_SEARCH_URL", _DEFAULT_BASE_URL)
_CONTACT_EMAIL = os.getenv("WIKI_SEARCH_EMAIL", "")
_USER_AGENT = (
    f"CogOpsCB-BDWikiSearch/1.0 "
    f"({_CONTACT_EMAIL or 'no-contact@example.invalid'})"
)
_HEADERS = {"User-Agent": _USER_AGENT}


# --- Helpers -----------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<.*?>")


def _strip_html(raw: str) -> str:
    return _HTML_TAG_RE.sub("", raw or "").strip()


def _enforce_bangladesh_context(query: str) -> str:
    """Always append the Bengali word for Bangladesh so results stay in scope."""
    q = (query or "").strip()
    if not q:
        return "বাংলাদেশ"
    if "বাংলাদেশ" in q or "bangladesh" in q.lower():
        return q
    return f'{q} "বাংলাদেশ"'


def _format_timestamp(wiki_ts: str) -> Dict[str, Any]:
    """Parse Wikipedia ISO timestamps; flag pages older than _STALE_YEARS.

    Returns dict: {"human": str, "is_stale": bool, "age_days": int|None}
    """
    if not wiki_ts:
        return {"human": "অজানা", "is_stale": False, "age_days": None}
    try:
        dt = datetime.strptime(wiki_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return {"human": wiki_ts, "is_stale": False, "age_days": None}
    now = datetime.now(timezone.utc)
    age_days = (now - dt).days
    is_stale = age_days > (_STALE_YEARS * 365)
    return {
        "human": dt.strftime("%Y-%m-%d"),
        "is_stale": is_stale,
        "age_days": age_days,
    }


def _format_age_note(ts: Dict[str, Any]) -> str:
    """Append a stale-warning suffix when the page hasn't been updated recently."""
    if ts.get("is_stale"):
        return (
            f" ⚠️ (সর্বশেষ সম্পাদনা {ts['human']}; "
            f"{_STALE_YEARS} বছরের বেশি পুরনো — তথ্য যাচাই করুন)"
        )
    if ts.get("human") and ts["human"] != "অজানা":
        return f" (সর্বশেষ সম্পাদনা: {ts['human']})"
    return ""


def _do_get(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(
            _BASE_URL,
            params=params,
            headers=_HEADERS,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Wikipedia API request failed: {e}")
        return None


def _wikipedia_search_sync(query: str, top: int) -> str:
    """Return a markdown-formatted list of the top-N Wikipedia hits."""
    cfg = get_tool_config(_CONFIG, "wiki") or {}
    limit = max(1, min(int(top or cfg.get("search_limit", _DEFAULT_SEARCH_LIMIT)), 10))

    forced_query = _enforce_bangladesh_context(query)
    data = _do_get({
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": forced_query,
        "srlimit": limit,
        "utf8": 1,
    })
    if data is None:
        return "Wikipedia অনুসন্ধানে ত্রুটি হয়েছে। পরে চেষ্টা করুন।"

    results = data.get("query", {}).get("search", []) or []
    if not results:
        return (
            f"কোনো Wikipedia ফলাফল পাওয়া যায়নি '{query}'-এর জন্য "
            f"(বাংলাদেশ প্রসঙ্গে)।"
        )

    out: List[str] = [f"## Wikipedia: '{query}' ({len(results)} results)\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = _strip_html(r.get("snippet", ""))
        url = f"https://bn.wikipedia.org/wiki/{title.replace(' ', '_')}"
        ts = _format_timestamp(r.get("timestamp", ""))
        out.append(
            f"**{i}. {title}**{_format_age_note(ts)}\n"
            f"   URL: {url}\n"
            f"   Snippet: {snippet}\n"
        )
    return "\n".join(out)


def _wikipedia_get_summary_sync(page_title: str) -> str:
    if not page_title or not page_title.strip():
        return "page_title খালি। একটি বৈধ Wikipedia পৃষ্ঠার শিরোনাম দিন।"

    data = _do_get({
        "action": "query",
        "format": "json",
        "prop": "extracts|revisions",
        "rvprop": "timestamp",
        "exintro": 1,
        "explaintext": 1,
        "titles": page_title,
        "utf8": 1,
    })
    if data is None:
        return f"Wikipedia সারাংশ আনয়নে ত্রুটি: '{page_title}'।"

    pages = data.get("query", {}).get("pages", {}) or {}
    for page_id, info in pages.items():
        if str(page_id) == "-1" or info.get("missing") is not None:
            return f"Wikipedia-তে '{page_title}' নামে কোনো পৃষ্ঠা নেই।"
        content = info.get("extract") or ""
        revs = info.get("revisions") or [{}]
        ts = _format_timestamp(revs[0].get("timestamp", ""))
        if not content.strip():
            return (
                f"'{page_title}' পৃষ্ঠাটি পাওয়া গেছে কিন্তু কোনো "
                f"সারাংশ পাঠ নেই।{_format_age_note(ts)}"
            )
        return (
            f"## {page_title}{_format_age_note(ts)}\n\n"
            f"{content.strip()}"
        )
    return f"Wikipedia সারাংশ পাওয়া যায়নি: '{page_title}'।"


def _wikipedia_get_full_content_sync(page_title: str) -> str:
    if not page_title or not page_title.strip():
        return "page_title খালি। একটি বৈধ Wikipedia পৃষ্ঠার শিরোনাম দিন।"

    data = _do_get({
        "action": "query",
        "format": "json",
        "prop": "extracts|revisions",
        "rvprop": "timestamp",
        "explaintext": 1,
        "titles": page_title,
        "utf8": 1,
    })
    if data is None:
        return f"Wikipedia পূর্ণ পৃষ্ঠা আনয়নে ত্রুটি: '{page_title}'।"

    pages = data.get("query", {}).get("pages", {}) or {}
    cap = _DEFAULT_CONTENT_CAP
    for page_id, info in pages.items():
        if str(page_id) == "-1" or info.get("missing") is not None:
            return f"Wikipedia-তে '{page_title}' নামে কোনো পৃষ্ঠা নেই।"
        content = info.get("extract") or ""
        revs = info.get("revisions") or [{}]
        ts = _format_timestamp(revs[0].get("timestamp", ""))
        if not content.strip():
            return (
                f"'{page_title}' পৃষ্ঠাটি পাওয়া গেছে কিন্তু কোনো "
                f"মূল পাঠ্য নেই।{_format_age_note(ts)}"
            )
        truncated = content[:cap]
        suffix = "" if len(content) <= cap else f"\n\n…[কাটছাঁট করা হয়েছে — মূল পৃষ্ঠায় আরও আছে]"
        return (
            f"## {page_title} — পূর্ণ পাঠ্য{_format_age_note(ts)}\n\n"
            f"{truncated.strip()}{suffix}"
        )
    return f"Wikipedia পূর্ণ পাঠ্য পাওয়া যায়নি: '{page_title}'।"


# --- Async wrappers exposed to the registry ----------------------------

async def wikipedia_search(query: str, top: int = 5) -> str:
    return await asyncio.to_thread(_wikipedia_search_sync, query, top)


async def wikipedia_get_summary(page_title: str) -> str:
    return await asyncio.to_thread(_wikipedia_get_summary_sync, page_title)


async def wikipedia_get_full_content(page_title: str) -> str:
    return await asyncio.to_thread(_wikipedia_get_full_content_sync, page_title)


# --- Schemas -----------------------------------------------------------

wikipedia_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Fallback: search Bangla Wikipedia for the user's query "
                "(Bangladesh context is automatically enforced). Use this "
                "ONLY after graph tools (graph_search / entity_search / "
                "episodic_search) have been tried and returned no useful "
                "result. Returns the top-N page titles with snippets, URLs, "
                "and last-edited dates. Always start with `top=1` and call "
                "`wikipedia_get_summary` on the first result; only try the "
                "next result if the summary isn't relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search term in Bengali (preferred) or English. "
                            "The tool auto-appends the Bangladesh context."
                        ),
                    },
                    "top": {
                        "type": "integer",
                        "description": (
                            "Number of results to return (1–10). Start at 1; "
                            "increase only if earlier summaries aren't useful."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get_summary",
            "description": (
                "Fetch the intro paragraph and last-updated date of a specific "
                "Wikipedia page (use the exact title from wikipedia_search). "
                "If the summary contains the answer, synthesize and reply. If "
                "it doesn't, call `wikipedia_get_full_content` for the full page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_title": {
                        "type": "string",
                        "description": "Exact Wikipedia page title.",
                    },
                },
                "required": ["page_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get_full_content",
            "description": (
                "Fetch the full plaintext of a specific Wikipedia page "
                "(capped) when the summary wasn't sufficient. Use sparingly; "
                "output may be large (post-tool refine will condense it)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_title": {
                        "type": "string",
                        "description": "Exact Wikipedia page title.",
                    },
                },
                "required": ["page_title"],
            },
        },
    },
]

wikipedia_tools_map = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_summary": wikipedia_get_summary,
    "wikipedia_get_full_content": wikipedia_get_full_content,
}
