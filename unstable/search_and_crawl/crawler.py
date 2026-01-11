# /crawler.py
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from typing import List, Dict, Any
from bs4 import BeautifulSoup

# Import the specific configuration section from the central config module
from config import CRAWLER_CONFIG

async def crawl_urls_in_parallel(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Crawls URLs in parallel using the library's correct timeout configuration.
    """
    crawled_data = []

    # Get the timeout values from our corrected config
    browser_timeout_sec = CRAWLER_CONFIG.get('browser_page_timeout', 3)
    master_timeout_sec = CRAWLER_CONFIG.get('global_task_timeout', 4)

    # Layer 1: Configure the browser's page load timeout using the correct keyword.
    # The value is in milliseconds.
    run_config = CrawlerRunConfig(
        target_elements=CRAWLER_CONFIG.get('target_elements', []),
        excluded_tags=CRAWLER_CONFIG.get('excluded_tags', []),
        excluded_selector=', '.join(CRAWLER_CONFIG.get('excluded_selectors', [])),
        verbose=CRAWLER_CONFIG.get('verbose', False),
        # V-- THIS IS THE CORRECTED KEYWORD --V
        page_timeout=browser_timeout_sec * 1000  # e.g., 3 * 1000 = 3000ms
    )

    async with AsyncWebCrawler(verbose=CRAWLER_CONFIG.get('verbose', False)) as crawler:
        
        # Layer 2: The master timeout for the entire per-URL task.
        results = await crawler.arun_many(
            urls=urls,
            output_formats=['markdown'],
            config=run_config,
            timeout=master_timeout_sec
        )

        for result in results:
            if not result.success:
                print(f"[SKIPPED] URL failed or timed out: {result.url}")
                continue
            
            if not result.cleaned_html:
                continue

            soup = BeautifulSoup(result.cleaned_html, 'html.parser')

            for a_tag in soup.find_all('a'):
                a_tag.replace_with(a_tag.get_text())

            for tag in soup.find_all(['img', 'figure']):
                tag.decompose()

            clean_text = soup.get_text(separator='\n', strip=True)

            if clean_text:
                crawled_data.append({
                    "title": result.metadata.get("title", "No Title Found"),
                    "url": result.url,
                    "content": clean_text,
                })

    return crawled_data