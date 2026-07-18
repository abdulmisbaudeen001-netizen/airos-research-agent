"""
Search provider abstraction.

To swap providers: implement SearchProvider and update get_provider().
The rest of the application never imports a concrete provider directly.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Returns a list of results, each with at least:
          { "title": str, "url": str, "body": str }
        Returns [] on failure — never raises.
        """


class DuckDuckGoProvider(SearchProvider):
    """
    Default free provider. Uses the unofficial duckduckgo-search library.
    No API key required. May require maintenance if upstream changes.
    Pin the version in requirements.txt on the day of install.
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                }
                for r in results
            ]
        except Exception as exc:
            logger.error("DuckDuckGo search failed: %s", exc)
            return []


def get_provider() -> SearchProvider:
    """
    Returns the active search provider.
    To switch providers, change this function only.
    """
    return DuckDuckGoProvider()


# Module-level singleton
_provider = get_provider()


def search(query: str, max_results: int = 5) -> list[dict]:
    """Convenience function for the rest of the app."""
    return _provider.search(query, max_results=max_results)
