"""
Web search service - search the web using Tavily API.
Tavily is purpose-built for AI apps: clean results, no rate limiting,
free tier (1000 searches/month), no scraping needed.
Get your free API key at: https://app.tavily.com
"""

from typing import Any
from loguru import logger
import httpx

from app.core.config import settings

TAVILY_API_URL = "https://api.tavily.com/search"


def search_web(query: str, max_results: int | None = None) -> list[dict[str, str]]:
    """Search the web using Tavily. Returns list of {title, url, snippet}."""
    if max_results is None:
        max_results = settings.web_search_max_results

    try:
        response = httpx.post(
            TAVILY_API_URL,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",       # "advanced" uses 2 credits, basic = 1
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        search_results = []
        for r in data.get("results", []):
            search_results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),   # Tavily already returns clean content
            })

        logger.info(f"Web search for '{query[:50]}...' returned {len(search_results)} results")
        return search_results

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def search_and_scrape(query: str, max_results: int | None = None) -> list[dict[str, Any]]:
    """
    Search the web and return enriched results.
    Tavily already returns clean extracted content, so no scraping needed.
    """
    results = search_web(query, max_results)
    enriched = []

    for result in results:
        # Tavily's snippet IS the extracted content — no need to scrape
        enriched.append({
            "title": result["title"],
            "url": result["url"],
            "snippet": result["snippet"],
            "full_content": result["snippet"],   # already clean text from Tavily
        })

    logger.info(f"Retrieved {len(enriched)} results for '{query[:50]}...'")

    # If we got very few results, try a broader query
    if len(enriched) < 2:
        logger.info("Few results returned, trying broader search")
        alt_results = search_web(query + " explained", max_results=5)
        seen_urls = {r["url"] for r in enriched}
        for result in alt_results:
            if result["url"] not in seen_urls:
                enriched.append({
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "full_content": result["snippet"],
                })

    return enriched


async def decompose_query(query: str) -> list[str]:
    """Use the LLM to break a complex query into sub-queries."""
    from app.services.llm import generate

    prompt = f"""Break the following question into 2-3 short, independent web search queries.
Return ONLY the search queries, one per line. No numbering, bullets, or explanations.
Keep queries short (under 8 words each).

Question: {query}

Search queries:"""

    response = await generate(prompt)
    sub_queries = [q.strip().strip("-•*1234567890.)") for q in response.strip().split("\n") if q.strip()]
    sub_queries = [q for q in sub_queries if 10 < len(q) < 100]
    sub_queries = sub_queries[:3]

    if not sub_queries:
        return [query]

    logger.info(f"Decomposed query into {len(sub_queries)} sub-queries: {sub_queries}")
    return sub_queries