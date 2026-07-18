"""
Search provider abstraction.

Provider chain (tried in order until one returns results):
  1. Serper    — Google results via serper.dev         (SERPER_API_KEY)
  2. Tavily    — AI-optimized search via tavily.com    (TAVILY_API_KEY)
  3. DuckDuckGo — unofficial, no key, last resort

If a provider has no API key configured it is skipped automatically.
If a provider has a key but fails (quota, network, etc.) the next is tried.
The rest of the application calls search() and never needs to know which
provider ran.
"""

import logging
import os
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Returns a list of results, each with at least:
          { "title": str, "url": str, "body": str }
        Returns [] on failure — never raises.
        """


# ---------------------------------------------------------------------------
# Provider 1 — Serper (Google results)
# ---------------------------------------------------------------------------

class SerperProvider(SearchProvider):
    """
    Uses serper.dev — Google Search results via REST API.
    Free tier: 2,500 searches. No credit card required.
    Env var: SERPER_API_KEY
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        api_key = os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            return []

        try:
            response = httpx.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "body": item.get("snippet", ""),
                })
            logger.info("Serper returned %d results for: %s", len(results), query)
            return results

        except Exception as exc:
            logger.error("Serper search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Provider 2 — Tavily
# ---------------------------------------------------------------------------

class TavilyProvider(SearchProvider):
    """
    Uses Tavily Search API — built for AI agents.
    Free tier: 1,000 searches/month. No credit card required.
    Env var: TAVILY_API_KEY
    Sign up: tavily.com
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return []

        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "body": item.get("content", ""),
                })
            logger.info("Tavily returned %d results for: %s", len(results), query)
            return results

        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Provider 3 — DuckDuckGo (last resort, no key needed)
# ---------------------------------------------------------------------------

class DuckDuckGoProvider(SearchProvider):
    """
    Unofficial DuckDuckGo library. No API key required.
    Unreliable on hosted environments — used only as last resort.
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            mapped = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                }
                for r in results
            ]
            logger.info("DuckDuckGo returned %d results for: %s", len(mapped), query)
            return mapped

        except Exception as exc:
            logger.error("DuckDuckGo search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class FallbackSearchProvider(SearchProvider):
    """
    Tries each provider in order. Returns the first non-empty result set.
    Skips providers with no API key configured.
    Falls through to DuckDuckGo as the final fallback.
    """

    def __init__(self):
        self._providers = [
            ("Serper",     SerperProvider()),
            ("Tavily",     TavilyProvider()),
            ("DuckDuckGo", DuckDuckGoProvider()),
        ]

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        for name, provider in self._providers:
            try:
                results = provider.search(query, max_results=max_results)
                if results:
                    logger.info("Search fulfilled by %s (%d results)", name, len(results))
                    return results
                else:
                    logger.warning("%s returned no results, trying next provider", name)
            except Exception as exc:
                logger.error("%s raised an exception: %s — trying next provider", name, exc)

        logger.error("All search providers exhausted — returning empty results")
        return []


# ---------------------------------------------------------------------------
# Module-level singleton — this is what the rest of the app uses
# ---------------------------------------------------------------------------

_provider = FallbackSearchProvider()


def search(query: str, max_results: int = 5) -> list[dict]:
    """Convenience function for the rest of the app. Never raises."""
    return _provider.search(query, max_results=max_results)
