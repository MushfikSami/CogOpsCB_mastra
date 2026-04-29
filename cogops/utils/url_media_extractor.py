"""
cogops/utils/url_media_extractor.py

Extracts URLs and file references from search-tool context text.

Scans for:
  - HTTP/HTTPS URLs (e.g. https://example.com/page)
  - Wikipedia file references: [[File:name.jpg]], [[Image:name.png]]
  - Generic file extensions in text: .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx, .zip, .tar, .gz

Returns a de-duplicated list of:
  [{"url": "...", "type": "webpage"|"image"|"pdf"|"..."|"wiki_file"}, ...]
"""

import os
import re
from typing import List, Dict, Any, Literal
from urllib.parse import quote

# Well-known file extensions → type label
_FILE_EXTENSIONS: Dict[str, str] = {
    ".pdf": "pdf",
    ".doc": "document",
    ".docx": "document",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
    ".rar": "archive",
    ".7z": "archive",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".svg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".mp4": "video",
    ".avi": "video",
    ".mkv": "video",
    ".mov": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".ogg": "audio",
}

# Pattern for http/https URLs.
# Excludes whitespace, angle brackets, smart quotes, backticks, parentheses,
# brackets, pipes.  Dots ARE allowed (URLs contain them).  Trailing
# punctuation (.), ;, :, ) are cleaned via rstrip after extraction.
def _url_encode(s: str) -> str:
    """URL-encode a string, preserving Bengali characters as UTF-8 bytes."""
    return quote(s, safe="")


_URL_RE = re.compile(
    r"https?://[^\s<>'‘’“”\x60()\[\]{}|]+",
    re.IGNORECASE,
)

# Pattern for Wikipedia file references: [[File:x]], [[Image:x]], [[চিত্র:x]], [[ছবি:x]]
# Captures everything between "File:" and the closing "]]", then we strip
# display params (pipe + size/thumb/etc.) to get the bare filename.
_WIKI_FILE_RE = re.compile(
    r'\[\[(?:File|Image|চিত্র|ছবি)\s*:\s*([^\]]+?)\s*\]\]',
    re.IGNORECASE,
)

# Pattern for standalone file extensions in text (e.g. "document.pdf")
_STANDALONE_FILE_RE = re.compile(
    r'\b(\S+\.(?:pdf|docx?|xlsx?|pptx?|zip|tar|gz|rar|7z|jpe?g|png|gif|webp|svg|bmp|tiff|mp4|avi|mkv|mov|wav|mp3|ogg))',
    re.IGNORECASE,
)


def extract_urls_media(text: str, source: str = "wikipedia") -> List[Dict[str, Any]]:
    """
    Extract all URLs and file references from *text*.

    Args:
        text: Raw context string (combined_context, retrieved text, etc.)
        source: "wikipedia" or "jiggasha" — affects file-extension
                classification and URL construction for wiki file refs.

    Returns:
        Deduplicated list of {"url": str, "type": str} dicts, preserving
        discovery order (URLs first, then wiki files, then standalone files).
    """
    if not text:
        return []

    seen: set[str] = set()
    results: List[Dict[str, Any]] = []

    def _add(url: str, typ: str) -> None:
        key = url.lower()
        if key in seen:
            return
        seen.add(key)
        results.append({"url": url, "type": typ})

    # 1. HTTP/HTTPS URLs
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(").,;:")
        _add(url, "webpage")

    # 2. Wikipedia file references: [[File:Foo.jpg|15px]], [[চিত্র:Foo.jpg|15px]]
    #    Captures the raw content between "File:" and "]]", then strips
    #    display params (pipe + size/thumb) to get the bare filename.
    for match in _WIKI_FILE_RE.finditer(text):
        raw = match.group(1)
        filename = raw.split("|")[0].strip()
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        typ = _FILE_EXTENSIONS.get(ext, "wiki_file")

        # Look for a Wikipedia page URL anywhere in the text.
        # The combined_context can be 1500+ chars, so a ±500 window won't
        # always catch the source URLs at the bottom.
        nearby = text
        page_url_match = _URL_RE.search(nearby)
        if page_url_match and "wikipedia" in page_url_match.group(0).lower():
            page_url = page_url_match.group(0)
            # Encode filename: spaces → _, Bengali → URL-encoded
            safe_name = filename.replace(" ", "_")
            safe_name = _url_encode(safe_name)

            if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff"):
                if source == "wikipedia":
                    # Build Wikipedia page URL with media anchor:
                    # https://bn.wikipedia.org/wiki/Mohammed_Shahabuddin#/media/চিত্র:Filename.jpg
                    media_anchor = _url_encode(f"চিত্র:{filename}")
                    # Insert #/media/ before the last path segment if no anchor exists
                    if "#" not in page_url:
                        wiki_url = f"{page_url}#/media/{media_anchor}"
                    else:
                        wiki_url = f"{page_url}#/media/{media_anchor}"
                    _add(wiki_url, typ)
                else:
                    # Jiggasha or unknown — try Wikimedia upload URL
                    prefix = safe_name[:2].lower() if safe_name else "xx"
                    first_dir = prefix[0] if prefix else "x"
                    second_dir = prefix[:2] if len(prefix) >= 2 else prefix[0]
                    upload_url = (
                        f"https://upload.wikimedia.org/wikipedia/commons/thumb/"
                        f"{first_dir}/{second_dir}/{safe_name}/120px-{safe_name}"
                    )
                    _add(upload_url, typ)
            else:
                _add(page_url, typ)
        else:
            _add(filename, typ)

    # 3. Standalone file extensions (e.g. "document.pdf" in text)
    # Skip matches that are inside [[...]] wiki file references (duplicates).
    for match in _STANDALONE_FILE_RE.finditer(text):
        # Check if this match is inside a [[...]] wiki reference
        start = match.start()
        # Count open/closed [[ ]] before this position
        before = text[:start]
        if before.count("[[") > before.count("]]"):
            continue
        fname = match.group(1)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()
        typ = _FILE_EXTENSIONS.get(ext, "document")
        _add(fname, typ)

    return results


def extract_urls_media_streamed(
    chunk_gen,
    source: str = "wikipedia",
) -> List[Dict[str, Any]]:
    """
    Extract URLs/media from a generator of text chunks (e.g. streaming tokens).
    Returns the same deduplicated list as `extract_urls_media`.

    This is useful when you want to start extracting before the full context
    has arrived — early URLs will appear in results immediately.
    """
    seen: set[str] = set()
    results: List[Dict[str, Any]] = []
    buf = ""
    # Hold back a bit so file refs spanning chunks are caught
    HOLDBACK = 20

    def _add(url: str, typ: str) -> None:
        key = url.lower()
        if key in seen:
            return
        seen.add(key)
        results.append({"url": url, "type": typ})

    for chunk in chunk_gen:
        buf += chunk
        if len(buf) > HOLDBACK:
            safe = buf[:-HOLDBACK]
            buf = buf[-HOLDBACK:]
            for m in _URL_RE.finditer(safe):
                url = m.group(0).rstrip(").,;:")
                _add(url, "webpage")
            for m in _WIKI_FILE_RE.finditer(safe):
                filename = m.group(1).split("|")[0].strip()
                _, ext = os.path.splitext(filename)
                typ = _FILE_EXTENSIONS.get(ext, "wiki_file")
                _add(filename, typ)
            for m in _STANDALONE_FILE_RE.finditer(safe):
                start = m.start()
                before = safe[:start]
                if before.count("[[") > before.count("]]"):
                    continue
                fname = m.group(1)
                _, ext = os.path.splitext(fname)
                typ = _FILE_EXTENSIONS.get(ext, "document")
                _add(fname, typ)

    # Final pass on remaining buffer
    for m in _URL_RE.finditer(buf):
        url = m.group(0).rstrip(").,;:")
        _add(url, "webpage")
    for m in _WIKI_FILE_RE.finditer(buf):
        filename = m.group(1).split("|")[0].strip()
        _, ext = os.path.splitext(filename)
        typ = _FILE_EXTENSIONS.get(ext, "wiki_file")
        _add(filename, typ)
    for m in _STANDALONE_FILE_RE.finditer(buf):
        start = m.start()
        before = buf[:start]
        if before.count("[[") > before.count("]]"):
            continue
        fname = m.group(1)
        _, ext = os.path.splitext(fname)
        typ = _FILE_EXTENSIONS.get(ext, "document")
        _add(fname, typ)

    return results
