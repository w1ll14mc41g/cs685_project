"""
Web Retrieval Module with Tavily API Integration

Implements web document retrieval using Tavily API (free tier: 1,000 searches/month)
with aggressive caching, rate limiting, and structured output.

See src/retrieval/README.md for detailed documentation, caching strategy, cost analysis,
and integration examples.

Usage:
    from src.retrieval.web_retrieval import search_web
    web_docs = search_web("climate change", k=3)
    # Results auto-cached; second call retrieves from cache instantly
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path("cache")
WEB_RESULTS_CACHE = CACHE_DIR / "web_results.json"


def _load_cache() -> dict:
    """Load cache from cache/web_results.json or return empty dict."""
    if WEB_RESULTS_CACHE.exists():
        with open(WEB_RESULTS_CACHE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Save cache to cache/web_results.json for persistence across runs."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(WEB_RESULTS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def _get_cache_key(query: str, k: int) -> str:
    """
    Generate cache key as MD5 hash of "query:k".
    
    Different k values produce different cache entries:
        _get_cache_key("climate change", k=3) → hash1
        _get_cache_key("climate change", k=5) → hash2  # Different!
    """
    key_str = f"{query}:{k}"
    return hashlib.md5(key_str.encode()).hexdigest()


def search_web(query: str, k: int = 3, max_retries: int = 3, initial_backoff: float = 1.0) -> list:
    """
    Retrieve top-k web results with caching and exponential backoff rate limiting.
    
    Args:
        query (str): Search query
        k (int): Number of results (default: 3)
        max_retries (int): Max retry attempts (default: 3)
        initial_backoff (float): Initial backoff seconds (default: 1.0)
    
    Returns:
        list[dict]: Web documents with keys: id, content, url, source_type, title, domain
    
    See src/retrieval/README.md for caching strategy, cost analysis, and examples.
    """
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not set in environment")
        return []
    
    # Check cache first (avoids API call if results already retrieved)
    cache = _load_cache()
    cache_key = _get_cache_key(query, k)
    
    if cache_key in cache:
        logger.info(f"Cache hit for query: {query[:50]}...")
        return cache[cache_key]
    
    # Attempt API call with exponential backoff for rate limiting
    client = TavilyClient(api_key=api_key)
    backoff = initial_backoff
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Tavily API call for query: {query[:50]}... (attempt {attempt + 1}/{max_retries})")
            
            # Call Tavily API with structured parameters
            response = client.search(query=query, max_results=k, include_answer=False)
            
            # Process and structure results to match local retrieval format
            web_docs = []
            if "results" in response:
                for i, result in enumerate(response["results"]):
                    # Create structured document matching tfidf_retrieval output format
                    # (id, content, url, source_type, title, domain)
                    doc = {
                        "id": hashlib.md5(f"{query}:{result.get('url', '')}".encode()).hexdigest(),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "source_type": "web",
                        "title": result.get("title", ""),
                        "domain": result.get("source", ""),
                    }
                    web_docs.append(doc)
            
            # Cache the results for future calls (prevents re-querying)
            cache[cache_key] = web_docs
            _save_cache(cache)
            
            logger.info(f"Retrieved {len(web_docs)} web documents")
            return web_docs
        
        except Exception as e:
            last_error = e
            if "rate_limit" in str(e).lower() or "429" in str(e):
                # Rate limit detected: retry with exponential backoff
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limited. Backing off for {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff: 1s → 2s → 4s
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
            else:
                # Non-rate-limit error: don't retry
                logger.error(f"Tavily API error: {e}")
                break
    
    # All retries exhausted or error occurred
    logger.error(f"Failed to retrieve web results after {max_retries} attempts: {last_error}")
    return []
