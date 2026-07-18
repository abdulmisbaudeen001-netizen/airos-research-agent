"""
Report Orchestrator.

Takes a classified intent, runs the appropriate tools (browser / search),
passes collected data to the LLM, and returns the final response string.
"""

import asyncio
import logging

import browser as browser_engine
import search as search_engine
import llm
from utils import normalize_url

logger = logging.getLogger(__name__)


async def generate(
    user_message: str,
    intent: dict,
    conversation_history: list[dict],
) -> str:
    """
    Main entry point. Returns the final text response for the user.
    """
    collected = {}

    tools = intent.get("tools", [])
    targets = intent.get("targets", [])
    execution = intent.get("execution", "single")
    query = intent.get("query", "")

    if not tools:
        # General question — no data collection needed
        collected["knowledge"] = {"type": "general"}
        return await llm.reason(user_message, collected, conversation_history)

    if execution == "parallel":
        collected = await _run_parallel(tools, targets, query)
    elif execution == "sequential":
        collected = await _run_sequential(tools, targets, query)
    else:
        # Single tool / target
        collected = await _run_single(tools, targets, query)

    return await llm.reason(user_message, collected, conversation_history)


# ---------------------------------------------------------------------------
# Execution strategies
# ---------------------------------------------------------------------------

async def _run_single(tools: list, targets: list, query: str) -> dict:
    collected = {}

    if "browser" in tools and targets:
        url = normalize_url(targets[0])
        logger.info("Browser: %s", url)
        data = await browser_engine.collect(url)
        collected["website"] = {"type": "browser", "data": data}

    elif "search" in tools:
        q = query or (targets[0] if targets else "")
        logger.info("Search: %s", q)
        results = search_engine.search(q)
        collected["search"] = {"type": "search", "results": results}

    return collected


async def _run_parallel(tools: list, targets: list, query: str) -> dict:
    """Run browser and/or search tasks concurrently."""
    tasks = {}

    for i, target in enumerate(targets):
        key = f"source_{i+1}"
        if target.startswith("http") or "." in target.split("/")[0]:
            # Looks like a URL/domain — use browser
            if "browser" in tools:
                url = normalize_url(target)
                tasks[key] = ("browser", url)
            else:
                tasks[key] = ("search", target)
        else:
            tasks[key] = ("search", target)

    # If no structured targets but we have a query, fall back to single search
    if not tasks and query:
        results = search_engine.search(query)
        return {"search": {"type": "search", "results": results}}

    async def run_task(key, task_type, target):
        if task_type == "browser":
            data = await browser_engine.collect(target)
            return key, {"type": "browser", "data": data}
        else:
            results = search_engine.search(target)
            return key, {"type": "search", "results": results}

    results = await asyncio.gather(
        *[run_task(k, t, tgt) for k, (t, tgt) in tasks.items()],
        return_exceptions=True,
    )

    collected = {}
    for item in results:
        if isinstance(item, Exception):
            logger.error("Parallel task failed: %s", item)
        else:
            key, data = item
            collected[key] = data

    return collected


async def _run_sequential(tools: list, targets: list, query: str) -> dict:
    """
    Run tools in order. Currently used for verification:
    browser first (if applicable), then search informed by what was found.
    """
    collected = {}

    if "browser" in tools and targets:
        url = normalize_url(targets[0])
        logger.info("Sequential browser: %s", url)
        data = await browser_engine.collect(url)
        collected["website"] = {"type": "browser", "data": data}

    if "search" in tools:
        # Use the explicit query, or fall back to the original user query
        q = query or (targets[1] if len(targets) > 1 else "")
        if q:
            logger.info("Sequential search: %s", q)
            results = search_engine.search(q)
            collected["search"] = {"type": "search", "results": results}

    return collected
