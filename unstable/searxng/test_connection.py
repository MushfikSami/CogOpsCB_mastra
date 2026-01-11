import os
import requests
import json

# --- Configuration ---
SEARXNG_API_URL = "http://localhost:9123/"
SEARXNG_API_KEY = "THE SECRET API KEY"

def search(query: str, num_results: int = 5):
    """
    Performs a search query against the private SearXNG API.
    """
    headers = {
        "Authorization": f"Bearer {SEARXNG_API_KEY}",
        "Accept": "application/json",  # Explicitly accept JSON
    }
    # Parameters to be sent in the URL query string
    params = {
        'q': query,
        'format': 'json',
        'count': num_results,
    }

    try:
        # Use GET request which is more common for search APIs
        response = requests.get(SEARXNG_API_URL + "search", headers=headers, params=params, timeout=6)

        # --- START OF DEBUGGING BLOCK ---
        print("\n--- DEBUG INFO ---")
        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers.get('Content-Type', 'N/A')}")
        print("--- Response Body (Raw) ---")
        print(response.text)
        print("--------------------------\n")
        # --- END OF DEBUGGING BLOCK ---

        # Now, attempt to parse the response
        if response.status_code == 200:
            results = response.json()
            return results.get("results", [])
        else:
            print(f"Error: Received non-200 status code {response.status_code}")
            return None

    except json.JSONDecodeError:
        print("❌ JSON Decode Error: The server did not return valid JSON.")
        print("   Check the raw response body above to see the actual server output (it might be an HTML error page).")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    search_query = "benefits of containerization vs virtualization"
    print(f"🔍 Searching for: '{search_query}'...")
    
    search_results = search(search_query)

    if search_results:
        print(f"\n✅ Found {len(search_results)} results:\n")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result.get('title', 'No Title')}")
            print(f"   URL: {result.get('url', 'No URL')}")
            print(f"   Snippet: {result.get('content', 'No snippet available.')[:150]}...\n")
    else:
        print("\n❌ Failed to retrieve search results.")