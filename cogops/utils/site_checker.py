"""
cogops/utils/site_checker.py

Async URL health checker using aiohttp.

Given a list of {"url": str, "type": str}, checks each URL with a HEAD
request / GET fallback and attaches a "status" field: "alive", "redirect",
"not_found", "error: ..." .

Special handling:
- .gov.bd HTTPS URLs: SSL certs are expired/self-signed → bypass cert
  verification. Also falls back to HTTP if HTTPS fails.
- Wikipedia (.wikipedia.org) URLs: treated as "alive" without a network
  check (the backend search already confirmed the page exists).
"""

import asyncio
import logging
import ssl
from typing import Any, Dict, List, Optional

import aiohttp

from cogops.utils.url_media_extractor import _URL_RE

logger = logging.getLogger(__name__)

# Timeout per URL check (seconds)
_CHECK_TIMEOUT = 10

# Concurrency limit — don't hammer endpoints
_MAX_CONCURRENCY = 8

# SSL context that skips certificate verification (for .gov.bd sites with
# expired/self-signed certs).
_GOVBD_SSL_CTX = ssl.create_default_context()
_GOVBD_SSL_CTX.check_hostname = False
_GOVBD_SSL_CTX.verify_mode = ssl.CERT_NONE


def _is_govbd_https(url: str) -> bool:
    """Return True if this is an HTTPS URL targeting a .gov.bd domain."""
    return url.startswith("https://") and ".gov.bd" in url


def _is_govbd_http(url: str) -> bool:
    """Return True if this is an HTTP URL targeting a .gov.bd domain."""
    return url.startswith("http://") and ".gov.bd" in url


def _is_wikipedia(url: str) -> bool:
    """Return True if this is a Wikipedia URL."""
    return "wikipedia" in url


def _is_wikimedia(url: str) -> bool:
    """Return True if this is a Wikimedia Commons URL."""
    return "upload.wikimedia.org" in url


def _to_http(url: str) -> str:
    """Convert https://...gov.bd... to http://...gov.bd..."""
    return url.replace("https://", "http://", 1)


def replace_webpage_urls_in_text(
    text: str,
    original_items: List[Dict[str, Any]],
    checked_items: List[Dict[str, Any]],
) -> str:
    """
    For webpage-type media items, if the checked URL differs from the
    original URL in *text*, replace the text version with the verified URL.

    The checker may normalise https → http (e.g. for .gov.bd sites).
    Pass the same objects before and after check_urls.
    Status info is already in the Media Links section — only replace the URL.
    """
    if not original_items or not checked_items:
        return text

    url_map: dict[str, str] = {}
    for orig, chk in zip(original_items, checked_items):
        orig_url = orig.get("url", "")
        chk_url = chk.get("url", "")
        if not orig_url.startswith("http") or not chk_url.startswith("http"):
            continue
        if orig.get("type") != "webpage":
            continue
        if orig_url.lower() == chk_url.lower():
            continue
        url_map[orig_url.lower()] = chk_url

    if not url_map:
        return text

    def _replace(m):
        low = m.group(0).lower()
        return url_map.get(low, m.group(0))

    return _URL_RE.sub(_replace, text)


async def check_urls(
    items: List[Dict[str, Any]],
    timeout: int = _CHECK_TIMEOUT,
    max_concurrency: int = _MAX_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """
    Check whether each URL in *items* is alive.

    Args:
        items: List of dicts with at least "url" and "type" keys.
        timeout: Seconds per URL check.
        max_concurrency: Max simultaneous checks.

    Returns:
        Same list with "status" key added to each dict:
        - "alive"       — 2xx response
        - "redirect"    — 3xx response (location included)
        - "not_found"   — 404
        - "error: ..."  — other HTTP error or connection failure
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    results: List[Dict[str, Any]] = [dict(item) for item in items]

    # Build SSL-aware connector for .gov.bd URLs
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        ttl_dns_cache=300,
        ssl=_GOVBD_SSL_CTX,
    )

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
    }

    async with aiohttp.ClientSession(connector=connector, headers=HEADERS) as session:
        tasks = [
            _check_one(session, item, semaphore, timeout)
            for item in results
        ]
        await asyncio.gather(*tasks)

    return results


async def _check_one(
    session: aiohttp.ClientSession,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    timeout: int,
) -> None:
    """Check a single URL and mutate item in-place with status."""
    url = item.get("url", "")
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        item["status"] = "error: not a URL"
        return

    # Wikipedia / Wikimedia URLs: already confirmed alive by the search
    # backend. No need to HEAD-check.
    if _is_wikipedia(url) or _is_wikimedia(url):
        item["status"] = "alive"
        return

    # For .gov.bd HTTPS URLs, try HTTP as fallback (many gov sites only
    # serve HTTP or have expired certs).
    # For other HTTPS URLs, we'll try HTTP too if HEAD returns 403/404.
    alt_url = _to_http(url) if _is_govbd_https(url) else None
    non_govbd_https = url.startswith("https://") and not _is_govbd_https(url)

    async with semaphore:
        checked_urls = [url]
        if alt_url:
            checked_urls.append(alt_url)

        for attempt_url in checked_urls:
            try:
                async with session.head(
                    attempt_url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    allow_redirects=False,
                ) as resp:
                    if 200 <= resp.status < 300:
                        # For .gov.bd URLs, always return the HTTP version
                        if _is_govbd_https(attempt_url):
                            item["url"] = attempt_url.replace("https://", "http://", 1)
                        item["status"] = "alive"
                        break
                    elif 300 <= resp.status < 400:
                        location = resp.headers.get("Location", "")
                        if attempt_url == url:
                            item["status"] = "redirect"
                            if location:
                                if _is_govbd_https(location):
                                    location = location.replace("https://", "http://", 1)
                                item["redirect_to"] = location
                        else:
                            item["status"] = "redirect"
                            if location:
                                if _is_govbd_https(location):
                                    location = location.replace("https://", "http://", 1)
                                item["redirect_to"] = location
                        if _is_govbd_https(item["url"]):
                            item["url"] = item["url"].replace("https://", "http://", 1)
                        break
                    # For .gov.bd HTTPS URLs, non-2xx/3xx may be due to CDN/SSL
                    # blocking (e.g. Akamai 403). Fall through to try HTTP.
                    elif _is_govbd_https(attempt_url) and attempt_url == url:
                        logger.debug(
                            "URL %s → %d (CDN/SSL block?), trying HTTP fallback",
                            url, resp.status,
                        )
                        # Don't set error yet — fall through to alt_url
                    # For non-`.gov.bd` HTTPS URLs, 403/404 from HEAD — try HTTP.
                    elif non_govbd_https and attempt_url == url and resp.status in (403, 404):
                        logger.debug(
                            "URL %s → %d (HEAD blocked?), trying HTTP fallback",
                            url, resp.status,
                        )
                        # Fall through — alt_url will be tried after the loop.
                        if alt_url is None:
                            alt_url = _to_http(url)

            except asyncio.TimeoutError:
                if attempt_url == alt_url:
                    item["status"] = "error: timeout"
            except aiohttp.ClientError as e:
                if attempt_url == alt_url:
                    item["status"] = f"error: {type(e).__name__}: {e}"
                logger.debug("URL %s (%s) failed: %s", url, attempt_url, e)
            except Exception as e:
                if attempt_url == alt_url:
                    item["status"] = f"error: {type(e).__name__}: {e}"

        # If no status set yet (HTTPS HEAD blocked by CDN/SSL, or HTTP HEAD
        # returned non-2xx), try HTTP HEAD → GET fallback chain for .gov.bd.
        if "status" not in item and alt_url:
            # Step 1: Try HTTP HEAD
            try:
                async with session.head(
                    alt_url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    allow_redirects=False,
                ) as resp:
                    if 200 <= resp.status < 300:
                        item["url"] = alt_url
                        item["status"] = "alive"
                    elif 300 <= resp.status < 400:
                        location = resp.headers.get("Location", "")
                        item["url"] = alt_url
                        item["status"] = "redirect"
                        if location:
                            item["redirect_to"] = location
                    elif resp.status == 403:
                        item["url"] = alt_url
                        item["status"] = "unknown"
                    else:
                        # HEAD returned non-2xx/3xx — try GET as fallback.
                        # Many .gov.bd sites block HEAD but accept GET.
                        await _try_get(session, alt_url, item, resp.status, timeout, True)
            except asyncio.TimeoutError:
                item["url"] = alt_url
                item["status"] = "error: timeout"
            except aiohttp.ClientError as e:
                item["url"] = alt_url
                item["status"] = f"error: {type(e).__name__}: {e}"
            except Exception as e:
                item["url"] = alt_url
                item["status"] = f"error: {type(e).__name__}: {e}"
        elif "status" not in item and not alt_url:
            item["status"] = "error: no response"

        logger.debug("Checked %s → %s", item["url"], item["status"])


async def _try_get(
    session: aiohttp.ClientSession,
    url: str,
    item: Dict[str, Any],
    head_status: int,
    timeout: int,
    is_govbd: bool = False,
) -> None:
    """Try GET when HEAD failed. .gov.bd sites often block HEAD but accept GET."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as resp:
            if 200 <= resp.status < 300:
                item["url"] = url
                item["status"] = "alive"
            elif 300 <= resp.status < 400:
                location = resp.headers.get("Location", "")
                item["url"] = url
                item["status"] = "redirect"
                if location:
                    if is_govbd and location.startswith("https://"):
                        location = location.replace("https://", "http://", 1)
                    item["redirect_to"] = location
            elif resp.status == 403:
                item["url"] = url
                item["status"] = "unknown"
            else:
                item["url"] = url
                item["status"] = f"error: {resp.status}"
    except asyncio.TimeoutError:
        item["url"] = url
        item["status"] = "error: timeout"
    except aiohttp.ClientError as e:
        item["url"] = url
        item["status"] = f"error: {type(e).__name__}: {e}"
    except Exception as e:
        item["url"] = url
        item["status"] = f"error: {type(e).__name__}: {e}"
