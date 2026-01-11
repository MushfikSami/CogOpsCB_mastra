# /main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Import helper modules and the specific config section
from search_client import search_links
from crawler import crawl_urls_in_parallel
from config import API_CONFIG

app = FastAPI(
    title="Intelligent Crawler API",
    description="An API that searches for a query and crawls relevant government websites.",
    version="1.0.0",
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class CrawlResult(BaseModel):
    title: str
    url: str
    content: str

class ApiResponse(BaseModel):
    status: str
    results: List[CrawlResult]

# --- API Endpoints ---
@app.post("/search-and-crawl", response_model=ApiResponse)
async def search_and_crawl(request: QueryRequest):
    """
    Performs a search, filters URLs based on configuration, crawls the results,
    and returns cleaned, visible content.
    """
    print(f"Received query: '{request.query}'")
    
    try:
        search_results = await search_links(request.query)
        if not search_results:
            return {"status": "success", "results": []}

        # The check for filter_domain is now done on startup in config.py
        filter_domain = API_CONFIG['filter_domain']

        filtered_urls = [
            result.get("url")
            for result in search_results
            if result.get("url") and filter_domain in result.get("url")
        ]

        if not filtered_urls:
            print(f"No '{filter_domain}' URLs found in search results.")
            return {"status": "success", "results": []}
        
        excluded_extensions = tuple(API_CONFIG.get('excluded_file_extensions', []))
        crawlable_urls = [
            url for url in filtered_urls
            if not any(url.split('?')[0].lower().endswith(ext) for ext in excluded_extensions)
        ]

        if not crawlable_urls:
            print("No crawlable HTML URLs found after filtering for file extensions.")
            return {"status": "success", "results": []}
        
        print(f"Found {len(crawlable_urls)} crawlable URLs to process.")

        crawled_content = await crawl_urls_in_parallel(crawlable_urls)

        return {"status": "success", "results": crawled_content}

    except Exception as e:
        print(f"An error occurred in the main endpoint: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Intelligent Crawler API. Use the /docs endpoint for interaction."}