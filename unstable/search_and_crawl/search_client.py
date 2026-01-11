# /search_client.py
import httpx
from typing import List, Dict, Any

# Import settings and secrets from the central config module
from config import SEARCH_CONFIG, SEARXNG_API_URL, SEARXNG_API_KEY

async def search_links(query: str) -> List[Dict[str, Any]]:
    """
    Performs an asynchronous search using settings from the central config.
    """
    headers = {
        "Authorization": f"Bearer {SEARXNG_API_KEY}",
        "Accept": "application/json",
    }
    params = {
        'q': query,
        'format': 'json',
        'count': SEARCH_CONFIG.get('num_results', 50),
    }

    async with httpx.AsyncClient() as client:
        try:
            timeout = SEARCH_CONFIG.get('timeout', 5.0)
            
            response = await client.get(
                SEARXNG_API_URL.rstrip('/') + "/search", # Ensure single slash
                headers=headers,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])

        except httpx.RequestError as e:
            print(f"An error occurred during the request to SearXNG: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred in search_links: {e}")
            return []