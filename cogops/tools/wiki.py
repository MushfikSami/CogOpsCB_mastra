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
    """Return a markdown-formatted list of Wikipedia page-title suggestions.

    Uses the OpenSearch (autocomplete) endpoint, which returns titles only —
    no snippets, no timestamps. The titles are intended to be passed to
    `wikipedia_get_summary` / `wikipedia_get_full_content`.

    OpenSearch response shape: [query, [titles], [descriptions], [links]]
    """
    cfg = get_tool_config(_CONFIG, "wiki") or {}
    limit = max(1, min(int(top or cfg.get("search_limit", _DEFAULT_SEARCH_LIMIT)), 10))

    if not query or not query.strip():
        return "অনুসন্ধান প্রশ্ন খালি। একটি বৈধ অনুসন্ধান প্রশ্ন দিন।"

    data = _do_get({
        "action": "opensearch",
        "format": "json",
        "search": query,
        "limit": limit,
        "utf8": 1,
    })
    if data is None:
        return "Wikipedia অনুসন্ধানে ত্রুটি হয়েছে। পরে চেষ্টা করুন।"

    if not isinstance(data, list) or len(data) < 4:
        return f"Wikipedia OpenSearch থেকে অপ্রত্যাশিত সাড়া: '{query}'।"

    titles = data[1] or []
    descriptions = data[2] or []
    links = data[3] or []

    if not titles:
        # OpenSearch found nothing — fall back to semantic ChromaDB lookup.
        return _chroma_title_fallback_sync(query, limit)

    out: List[str] = [f"## Wikipedia শিরোনাম প্রস্তাব: '{query}' ({len(titles)})\n"]
    for i, title in enumerate(titles, 1):
        desc = descriptions[i - 1] if i - 1 < len(descriptions) else ""
        url = links[i - 1] if i - 1 < len(links) else \
            f"https://bn.wikipedia.org/wiki/{title.replace(' ', '_')}"
        line = f"**{i}. {title}**\n   URL: {url}\n"
        if desc:
            line += f"   বিবরণ: {desc}\n"
        out.append(line)
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


# --- ChromaDB semantic fallback (used internally by wikipedia_search) --


def _chroma_title_fallback_sync(query: str, top: int) -> str:
    """Internal: semantic fallback when OpenSearch returns no titles.

    The WikiTitles collection stores page titles as documents, no metadata
    (see ingestion/wiki_title_vectorizer/chroma_config.yml — metadata_columns
    is empty and content_column is `page_title`). We embed the query with the
    same Gemma Triton embedder used at ingest time and read back
    `documents` + `distances` only.
    """
    cfg = get_tool_config(_CONFIG, "wiki") or {}
    n_results = max(1, min(int(top or cfg.get("chroma_top", _CHROMA_TOP)), 10))

    # tritonclient.http.InferenceServerClient expects "host:port" — no scheme.
    triton_url = _TRITON_URL
    for prefix in ("http://", "https://"):
        if triton_url.startswith(prefix):
            triton_url = triton_url[len(prefix):]
            break
    try:
        embedder_cfg = TritonEmbedderConfig(
            url=triton_url,
            model_name=_TRITON_MODEL,
            tokenizer_path=_TRITON_TOKENIZER,
            max_batch_size=8,
        )
        embedder = TritonEmbedder(config=embedder_cfg)
        embedding = embedder.create_sync(_QUERY_PREFIX + (query or ""))
    except Exception as e:
        return f"ট্রাইটন এম্বেডিং ত্রুটি: {type(e).__name__}: {e}"

    # chromadb.HttpClient expects a bare host — strip any scheme from env.
    chroma_host = _CHROMA_DB_HOST
    for prefix in ("http://", "https://"):
        if chroma_host.startswith(prefix):
            chroma_host = chroma_host[len(prefix):]
            break
    try:
        chroma_client = chromadb.HttpClient(
            host=chroma_host, port=_CHROMA_DB_PORT
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
            include=["documents", "distances"],
        )
    except Exception as e:
        return f"ক্রোমাডিবি অনুসন্ধান ত্রুটি: {e}"

    titles = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not titles:
        return (
            f"'{query}'-এর জন্য কোনো Wikipedia শিরোনাম প্রস্তাব পাওয়া যায়নি "
            f"(OpenSearch ও ভেক্টর উভয় উৎসে)।"
        )

    out: List[str] = [
        f"## Wikipedia শিরোনাম প্রস্তাব (ভেক্টর ফলব্যাক): '{query}' ({len(titles)})\n"
    ]
    for i, (title, dist) in enumerate(zip(titles, distances), 1):
        url = f"https://bn.wikipedia.org/wiki/{str(title).replace(' ', '_')}"
        out.append(
            f"**{i}. {title}**\n"
            f"   URL: {url}\n"
            f"   দূরত্ব: {float(dist):.4f}\n"
        )
    return "\n".join(out)


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
                "Fallback: get Wikipedia page-title suggestions (autocomplete) "
                "for the query. Use ONLY after graph tools have been tried and "
                "returned no useful result. Returns a ranked list of suggested "
                "page titles (with URLs and short descriptions). The titles are "
                "meant to be passed to `wikipedia_get_summary`, then "
                "`wikipedia_get_full_content` if the summary is insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search term in Bengali (preferred) or English."
                        ),
                    },
                    "top": {
                        "type": "integer",
                        "description": (
                            "Number of title suggestions to return (1–10). "
                            "Start at 1; increase only if earlier summaries "
                            "aren't useful."
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
]

wikipedia_tools_map = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_summary": wikipedia_get_summary,
    "wikipedia_get_full_content": wikipedia_get_full_content,
}
