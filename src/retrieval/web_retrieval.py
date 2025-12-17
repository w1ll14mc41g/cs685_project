"""
Web Retrieval Module with Tavily API Integration

Implements web document retrieval using Tavily API (free tier: 1,000 searches/month)
with structured output and rate limiting (exponential backoff).

Usage:
    from src.retrieval.web_retrieval import search_web
    web_docs = search_web("climate change", k=3)
"""


import os
import time
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_output(docs: list, api_k: int = None) -> dict:
    """
    Convert raw Tavily docs into the final structure format.

    Args:
        docs (list): list of Tavily-structured documents
        api_k (int): The actual max_results value passed to Tavily API

    Returns:
        dict: formatted output with num_docs, results, and api_k
    """
    return {
        "num_docs": len(docs),
        "api_k": api_k,
        "results": docs
    }


def search_web(
    query: str,
    k: int = 3,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    query_id: str = None,
):
    """
    Retrieve top-k web results from Tavily API with exponential backoff.
    Automatically adjusts API max_results parameter if needed to retrieve at least k
    documents. Returns structured output including the actual api_k used.
    
    Args:
        query (str): Search query
        k (int): Desired number of results
        max_retries (int): Max retry attempts for rate limiting
        initial_backoff (float): Initial backoff in seconds
        query_id (str, optional): Optional query ID from theperspective dataset.
                                  If provided, returns full entry structure with id, query, and web_docs.
    
    Returns:
        dict: If query_id is provided, returns {"id": query_id, "query": query, "web_docs": {...}}
              Otherwise, returns {"num_docs": ..., "api_k": ..., "results": [...]} (web_docs structure)
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY missing.")
        web_docs = _format_output([], None)
        # If query_id is provided, return full entry structure matching web files format
        if query_id is not None:
            return {
                "id": query_id,
                "query": query,
                "web_docs": web_docs
            }
        return web_docs
    # Call Tavily with retry logic
    client = TavilyClient(api_key=api_key)
    api_k = k
    max_api_k = k + 10  # Max attempts to reach k results
    docs = []
    api_k_used = k

    while api_k <= max_api_k:
        backoff = initial_backoff
        for attempt in range(max_retries):
            try:
                logger.info(f"Tavily API call for '{query[:60]}' (api_k={api_k}, attempt {attempt+1}/{max_retries})")
                response = client.search(
                    query=query,
                    max_results=api_k,
                    include_answer=False
                )
                raw_results = response.get("results", [])
                docs = []
                for idx, result in enumerate(raw_results):
                    url = result.get("url", "")
                    docs.append({
                        "id": idx,
                        "content": result.get("content", "") or "",
                        "url": url,
                        "source_type": "web",
                        "title": result.get("title", "") or "",
                        "domain": result.get("source", "") or "",
                    })
                # Check if we got enough results
                if len(docs) >= k:
                    # Trim to exactly k (keep most relevant, which are first in list)
                    docs = docs[:k]
                    api_k_used = api_k
                    web_docs = _format_output(docs, api_k_used)
                    
                    # If query_id is provided, return full entry structure matching web files format
                    if query_id is not None:
                        return {
                            "id": query_id,
                            "query": query,
                            "web_docs": web_docs
                        }
                    return web_docs
                # Not enough results, try with higher api_k
                logger.warning(f"Got {len(docs)} results, need {k}. Retrying with api_k={api_k+1}")
                break
            except Exception as e:
                msg = str(e).lower()
                # Handle rate-limit
                if "429" in msg or "rate" in msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited. Backoff {backoff}s…")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        logger.error("Rate limit exceeded fully.")
                else:
                    logger.error(f"Tavily API error (non-retry): {e}")
                break
        # If we got here, either max_retries exceeded or we need to try higher api_k
        if len(docs) < k and api_k < max_api_k:
            api_k += 1
        else:
            break
    # If we still don't have enough, return what we got
    api_k_used = api_k if len(docs) >= k else None
    if len(docs) < k:
        logger.warning(f"Search completed with {len(docs)} results (requested {k})")
    web_docs = _format_output(docs, api_k_used)
    
    # If query_id is provided, return full entry structure matching web files format
    if query_id is not None:
        return {
            "id": query_id,
            "query": query,
            "web_docs": web_docs
        }
    return web_docs