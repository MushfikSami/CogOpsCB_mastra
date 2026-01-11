# Private SearXNG API for SearchRAG

This project provides a secure, fast, and private search API using SearXNG, specifically configured to act as a backend for a Retrieval-Augmented Generation (RAG) pipeline.

The service runs in Docker, is accessible only within your local network, and is protected by a mandatory API key. It is optimized as a headless (API-only) service, disabling the web UI for better performance and security.

This guide includes a specific solution for servers where the `/home` partition is low on space, by intelligently storing Docker volume data on the main root (`/`) partition using symbolic links.

## Key Features

*   **API-Only (Headless):** The web interface is disabled to minimize resource usage and attack surface.
*   **API Key Authentication:** All requests must be authenticated with a secret API key.
*   **Local Network Access:** Designed to be consumed by other services (like a vLLM server) on the same private network.
*   **Conflict-Free Networking:** Uses a custom Docker network (`10.61.0.0/24`) to avoid common conflicts with VPNs and other services.
*   **Optimized for RAG:** Configured with a 5-second request timeout and high-quality search engines.
*   **Efficient Storage Management:** Instructions provided to handle low disk space on `/home` by storing data on the root partition.

## Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) installed.
*   [Docker Compose](https://docs.docker.com/compose/install/) installed.
*   [Python 3](https://www.python.org/downloads/) installed (for the Python testing example).
*   The `requests` library for Python (`pip install requests`).

---

## Deployment Guide (Step-by-Step)

### Step 1: Create Project Structure

First, create a directory for your project. We'll set up all the necessary files and subdirectories.

```bash
# Create the main project directory and enter it
mkdir my-searxng-api
cd my-searxng-api

# Create the placeholder for the configuration files
touch docker-compose.yml .env

# We will create the data directories in the next step
```

### Step 2: Set Up Data Storage on the Root Partition

Your server has limited space in `/home` but plenty in `/`. We will create the actual data directories in `/opt` and link them to our project folder. This is the most critical step for your specific server condition.

```bash
# 1. Create the real data directories in /opt
sudo mkdir -p /opt/docker-data/searxng
sudo mkdir -p /opt/docker-data/redis

# 2. Change ownership to your current user so Docker has permission to write
sudo chown -R $(whoami):$(whoami) /opt/docker-data

# 3. Navigate back to your project directory (if you're not already there)
cd ~/my-searxng-api # Adjust path if needed

# 4. Create symbolic links. These act as pointers from your project
#    folder to the actual data folders in /opt.
ln -s /opt/docker-data/searxng searxng-data
ln -s /opt/docker-data/redis redis-data
```
When you run `ls -l`, you should see the links clearly pointing to the `/opt` directories.

### Step 3: Create the `docker-compose.yml` File

This file orchestrates our SearXNG and Redis containers. It is configured to use the symbolic links we just created for data storage.

Paste the following content into `docker-compose.yml`:

```yaml
# docker-compose.yml
version: '3.8'

services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - "9123:8080"
    volumes:
      # Docker follows this symlink to store data in /opt/docker-data/searxng
      - ./searxng-data:/etc/searxng:rw
    environment:
      - SEARXNG_API_KEY=${SEARXNG_API_KEY}
    depends_on:
      - redis
    networks:
      - searxng-net
    logging: &logging_defaults
      driver: "json-file"
      options: { max-size: "10m", max-file: "3" }

  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    volumes:
      # Docker follows this symlink to store data in /opt/docker-data/redis
      - ./redis-data:/data
    networks:
      - searxng-net
    logging: *logging_defaults

networks:
  searxng-net:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: "10.61.0.0/24"
```

### Step 4: Set Your Secret API Key

For security, your API key is stored in a separate `.env` file. Docker Compose automatically reads this file and makes the variables available to the containers.

**Action:** Run the following command, replacing the placeholder with a strong, random key.

```bash
# You can generate a strong key with: openssl rand -hex 32
echo "SEARXNG_API_KEY=your_super_secret_api_key_here" > .env
```

### Step 5: Create the SearXNG `settings.yml`

This file configures the behavior of SearXNG. It enables API key authentication, sets the timeout, and defines which search engines to use.

**Action:** Create a file at `searxng-data/settings.yml` and paste the following. Remember to also replace the `secret_key` with a *different* random string.

```yaml
# ================================
# SearXNG Settings
# ================================

general:
  instance_name: "My Private SearXNG"
  default_locale: "bn"

doi_resolvers:
  doi.org: "https://doi.org/"
  oadoi.org: "https://oadoi.org/"
default_doi_resolver: "doi.org"

server:
  secret_key: "CHANGE_ME_TO_A_VERY_STRONG_RANDOM_KEY"
  limiter: false

limiter:
  rates: "240 per 1 minute"

http:
  # API Key support
  api_keys:
    - "!!ENV SEARXNG_API_KEY"

valkey:
  url: "redis://redis:6379/0"

search:
  formats:
    - json
  request_timeout: 5.0

# ================================
# Engines
# ================================
# Use default engines, but only enable the safe ones
use_default_engines: true

engines:
  - name: google
    engine: google
    shortcut: gg
    categories: ['general']
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    categories: ['general']
    disabled: false

  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    categories: ['general', 'science']
    disabled: false

```

### Step 6: Launch the Service

You are now ready to start the API server.

```bash
# Start the containers in detached (background) mode
docker-compose up -d

# Verify that the containers are running
docker-compose ps
```
You should see both the `searxng` and `redis` containers with a status of `Up`. To check the logs for any errors, run `docker-compose logs -f searxng`.

---

## Testing Guide (Step-by-Step)

### Prerequisites for Testing

1.  **Find your Server's IP:** Find the local IP address of the machine running Docker (e.g., `192.168.1.100`).
    *   On Linux/macOS: `ip a` or `ifconfig`
    *   On Windows: `ipconfig`
2.  **Get your API Key:** Use the key you placed in the `.env` file.

### Test 1: Using `curl` (Command Line)

This is the quickest way to confirm the API is working.

**Action:** Replace `<YOUR_SERVER_IP>` and `<YOUR_API_KEY>` with your actual values.

```bash
curl -X POST \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"q": "what is the speed of light", "count": 5}' \
  "http://<YOUR_SERVER_IP>:9123/"
```
You should receive a JSON object containing the search results.

### Test 2: Using a Python Script

This is a more practical example of how you would integrate the search API into your RAG application.

**Action:**
1. Make sure you have the `requests` library: `pip install requests`
2. Create a file named `test_search.py`.
3. Paste the code below, replacing the placeholder values for `SEARXNG_API_URL` and `SEARXNG_API_KEY`.

```python
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
```

**Run the script from your terminal:**

```bash
python test_search.py
```

You should see a formatted list of the top 5 search results, proving your API is fully functional and ready for integration.

## Managing the Service

*   **Stop the containers:** `docker-compose down`
*   **Restart the containers:** `docker-compose restart`
*   **View live logs:** `docker-compose logs -f`