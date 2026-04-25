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

import chromadb
import requests
from dotenv import load_dotenv

from cogops.config.loader import load_config, get_tool_config
from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig

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
_CHROMA_TOP = int(_CONFIG.get("wiki", {}).get("chroma_top", 5))

# ChromaDB / Triton settings for title-suggest fallback
_CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "localhost")
_CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", "8443"))
_WIKI_TITLE_COLLECTION = os.getenv("WIKI_TITLE_VECTOR_DB", "WikiTitles")
_TRITON_URL = os.getenv("TRITON_URL", "localhost:6000")
_TRITON_MODEL = os.getenv("TRITON_MODEL_NAME", "gemma_embedding")
_TRITON_TOKENIZER = os.getenv("TRITON_TOKENIZER", "onnx-community/embeddinggemma-300m-ONNX")

_QUERY_PREFIX = "task: search result | query: "

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


# --- ChromaDB-based title suggestion -----------------------------------


def _wikipedia_title_suggest_sync(query: str, top: int) -> str:
    """Find Wikipedia page titles via ChromaDB semantic search.

    Embeds the query with the Triton embedder (using the same query prefix
    as ingestion), then queries the WikiTitles collection.
    """
    cfg = get_tool_config(_CONFIG, "wiki") or {}
    n_results = max(1, min(top or cfg.get("chroma_top", _CHROMA_TOP), 10))

    # Build embedder — uses the same Gemma model as ingestion
    triton_url = _TRITON_URL
    if not triton_url.startswith(("http://", "https://")):
        triton_url = f"http://{triton_url}"
    embedder_cfg = TritonEmbedderConfig(
        url=triton_url,
        model_name=_TRITON_MODEL,
        tokenizer_path=_TRITON_TOKENIZER,
        max_batch_size=8,
    )
    embedder = TritonEmbedder(config=embedder_cfg)

    try:
        embedding = embedder.create_sync(_QUERY_PREFIX + (query or ""))
    except Exception as e:
        return f"ট্রাইটন এম্বেডিং ত্রুটি: {e}"

    # Connect to ChromaDB
    try:
        chroma_client = chromadb.HttpClient(
            host=_CHROMA_DB_HOST, port=_CHROMA_DB_PORT
        )
        chroma_client.heartbeat()
    except Exception as e:
        return f"ক্রোমাডিবি সংযোগ ত্রুটি: {e}"

    try:
        collection = chroma_client.get_collection(_WIKI_TITLE_COLLECTION)
    except Exception as e:
        return f"'{_WIKI_TITLE_COLLECTION}' সংগ্রহ পাওয়া যায়নি। {e}"

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas"],
        )
    except Exception as e:
        return f"ক্রোমাডিবি অনুসন্ধান ত্রুটি: {e}"

    titles = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not titles:
        return (
            f"'{query}'-এর জন্য কোনো মিল পাওয়া যায়নি Wikipedia শিরোনামের "
            f"ভেক্টর ডেটাবেজে। অনুগ্রহ করে ভিন্ন কিওয়ার্ড ব্যবহার করুন।"
        )

    out: List[str] = [f"## ভেক্টর মিল: '{query}' ({len(titles)} শিরোনাম)\n"]
    for i, (title, dist, meta) in enumerate(zip(titles, distances, metadatas), 1):
        out.append(
            f"**{i}. {title}**\n"
            f"   দূরত্ব: {dist:.4f}\n"
        )
        if meta:
            source = meta.get("source", "")
            if source:
                out.append(f"   উৎস: {source}\n")
    return "\n".join(out)


# --- Async wrappers exposed to the registry ----------------------------

async def wikipedia_search(query: str, top: int = 5) -> str:
    return await asyncio.to_thread(_wikipedia_search_sync, query, top)


async def wikipedia_get_summary(page_title: str) -> str:
    return await asyncio.to_thread(_wikipedia_get_summary_sync, page_title)


async def wikipedia_get_full_content(page_title: str) -> str:
    return await asyncio.to_thread(_wikipedia_get_full_content_sync, page_title)


async def wikipedia_title_suggest(query: str, top: int = 5) -> str:
    return await asyncio.to_thread(_wikipedia_title_suggest_sync, query, top)


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
                "Fetch the intro paragraph of a Wikipedia page. "
                "**Never call this as your first tool in a turn.** "
                "You must first call `wikipedia_search` to get the exact page title, "
                "then use that title here. Never use a URL or raw title from the "
                "user's message — always go through `wikipedia_search` first."
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
    {
        "type": "function",
        "function": {
            "name": "wikipedia_title_suggest",
            "description": (
                "Fallback: find relevant Wikipedia page titles using semantic "
                "similarity against a pre-indexed vector database. Use ONLY when "
                "`wikipedia_search` returns no results. Convert the user's query "
                "into Bengali keywords before calling. Returns a ranked list of "
                "matching Wikipedia page titles. After getting results, call "
                "`wikipedia_get_summary` on the top result; if the summary "
                "doesn't answer, try the next title from this list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search keywords in Bengali (preferred). Convert "
                            "the user's query into Bengali keywords before searching."
                        ),
                    },
                    "top": {
                        "type": "integer",
                        "description": (
                            "Number of titles to return (1–10). Start at 3–5."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]

wikipedia_tools_map = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_summary": wikipedia_get_summary,
    "wikipedia_get_full_content": wikipedia_get_full_content,
    "wikipedia_title_suggest": wikipedia_title_suggest,
}
