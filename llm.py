"""
LLM Engine — OpenRouter integration.

Every message flows through the LLM twice:
  1. classify_intent()  — LLM decides if it needs tools or can answer directly
  2. reason()           — LLM produces the final response
"""

import datetime
import json
import logging
import httpx

from config import OPENROUTER_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/airos-research-agent",
    "X-Title": "AIROS Research Agent",
}


def _now() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _date_context() -> str:
    now = _now()
    tomorrow = now + datetime.timedelta(days=1)
    yesterday = now - datetime.timedelta(days=1)
    return (
        f"Today is {now.strftime('%A, %B %d, %Y')}. "
        f"Yesterday was {yesterday.strftime('%A, %B %d, %Y')}. "
        f"Tomorrow is {tomorrow.strftime('%A, %B %d, %Y')}. "
        f"Current time is {now.strftime('%H:%M')} UTC."
    )


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _build_intent_prompt() -> str:
    return f"""You are an intent classifier for an AI research assistant.

{_date_context()}

Classify the user's message into exactly one intent and return ONLY valid JSON.
No markdown, no explanation, no extra text — just the JSON object.

Possible intents:
- "website_analysis": user wants to analyze a specific website
- "research": user wants current news, events, prices, recent happenings, or ANY information that may have changed recently and requires a live web search
- "general": questions the LLM can answer directly from its own knowledge — math, definitions, explanations, how things work, casual conversation

Critical routing rules:
- "What is today's date?" / "What day is it?" / "What is tomorrow?" → "general" — date is already known from context above, no search needed
- "What happened yesterday in X?" / "Latest news" / "What is going on in X?" / "Current price of X" / "Recent events" → "research" — needs live web search
- "Who is X?" where X is a well-known historical figure → "general"
- "Who is X?" where X may be a current/recent person or role → "research"
- Definitions, explanations, how things work → "general"
- Anything involving "latest", "current", "recent", "today's news", "what's happening", "yesterday" → "research"
- When unsure between "research" and "general" for a time-sensitive topic, choose "research"

Response schema:
{{"intent": "website_analysis" | "research" | "general", "execution": "single" | "parallel" | "sequential", "tools": ["browser"] | ["search"] | ["browser", "search"] | [], "targets": [], "query": "", "confidence": 0.0}}

Examples:

User: "What is today's date?"
{{"intent":"general","execution":"single","tools":[],"targets":[],"query":"","confidence":0.99}}

User: "What is tomorrow?"
{{"intent":"general","execution":"single","tools":[],"targets":[],"query":"","confidence":0.99}}

User: "What happened yesterday in New York?"
{{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"New York news yesterday","confidence":0.97}}

User: "What is FastAPI?"
{{"intent":"general","execution":"single","tools":[],"targets":[],"query":"","confidence":0.95}}

User: "What is going on in Nigeria?"
{{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"Nigeria latest news today","confidence":0.97}}

User: "Latest AI news"
{{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"latest AI news today","confidence":0.97}}

User: "Compare Railway and Render"
{{"intent":"research","execution":"parallel","tools":["search"],"targets":["Railway hosting","Render hosting"],"query":"","confidence":0.95}}

User: "Analyze github.com"
{{"intent":"website_analysis","execution":"single","tools":["browser"],"targets":["https://github.com"],"query":"","confidence":0.99}}

User: "Go to stripe.com and compare its pricing with PayPal"
{{"intent":"research","execution":"parallel","tools":["browser","search"],"targets":["https://stripe.com","PayPal pricing"],"query":"","confidence":0.93}}"""


async def classify_intent(message: str) -> dict:
    """
    Every message passes through here first.
    Falls back to general so the LLM at least attempts to answer.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _build_intent_prompt()},
            {"role": "user", "content": message},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(OPENROUTER_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()

        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        return {
            "intent": "general",
            "execution": "single",
            "tools": [],
            "targets": [],
            "query": message,
            "confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# Main reasoning call
# ---------------------------------------------------------------------------

def _build_reasoning_prompt() -> str:
    return f"""You are AIROS Research Agent, a smart AI assistant built to help users research anything.

{_date_context()}

You are the brain of this system. You have full permission to use your own knowledge, the date context above, and any search results or browser data provided to you.

How to respond:
- You already know today's date, yesterday's date, and tomorrow's date from the context above. Use them confidently and directly. Never say you cannot answer a date question.
- For simple questions — dates, definitions, explanations, casual conversation — answer directly and confidently from your own knowledge.
- For questions where search results or browser data were collected, use that data as your primary source and enrich it with your own knowledge where helpful.
- If search returned no results, still answer from your own knowledge. Never refuse a question just because search came back empty.
- Never say "I cannot access real-time information" or "I do not have access to the current date" — the date is provided above and you must use it.
- Never refuse to answer a question you have the knowledge to answer.

FORMATTING RULES — strictly follow these for every response:
- Use Telegram HTML formatting only. Never use markdown (* ** _ ` #).
- Start with a single bold title: <b>🔎 Topic Title</b>
- For section headers use: <b>🔹 Section Name</b>
- For bullet points use: • one short fact, maximum 10 words
- For sub-bullets use:   ↳ one supporting detail, one line only
- Bold key names or labels: <b>word</b>
- Separate every section with a blank line
- Never write paragraphs — every fact is its own bullet line
- Write like a news briefing: short, sharp, scannable
- Maximum 5 bullets per section

Website Report Structure (use only when analyzing a website):
<b>1. Purpose &amp; Overview</b>
<b>2. Target Audience</b>
<b>3. Core Features / Offerings</b>
<b>4. Technologies Detected</b>
<b>5. Navigation &amp; UX</b>
<b>6. Notable Network Activity</b>
<b>7. Strengths</b>
<b>8. Weaknesses / Gaps</b>
<b>9. Overall Assessment</b>"""


async def reason(user_message: str, collected_data: dict, conversation_history: list[dict]) -> str:
    """
    Final step — every response passes through here.
    LLM answers directly from knowledge + date context + any collected data.
    """
    data_summary = _format_collected_data(collected_data)

    messages = [
        {"role": "system", "content": _build_reasoning_prompt()},
    ]

    for entry in conversation_history[-6:]:
        messages.append(entry)

    messages.append({
        "role": "user",
        "content": f"User request: {user_message}\n\nCollected data:\n{data_summary}",
    })

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.3,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(OPENROUTER_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    except httpx.TimeoutException:
        return "The analysis took too long. Please try again."
    except Exception as exc:
        logger.error("LLM reasoning failed: %s", exc)
        return "Something went wrong. Please try again."


# ---------------------------------------------------------------------------
# Data formatter
# ---------------------------------------------------------------------------

def _format_collected_data(data: dict) -> str:
    if not data:
        return "No external data collected. Answer from your own knowledge and the date context in the system prompt."

    # General intent — no tools ran
    if list(data.keys()) == ["knowledge"] and data.get("knowledge", {}).get("type") == "general":
        return "No external data collected. Answer from your own knowledge and the date context in the system prompt."

    parts = []

    for source, payload in data.items():
        if source == "knowledge":
            continue

        parts.append(f"=== {source.upper()} ===")

        if payload.get("type") == "browser":
            d = payload["data"]
            if d.get("error"):
                parts.append(f"Browser error: {d['error']}")
                continue

            page = d.get("page", {})
            parts.append(f"URL: {page.get('final_url') or page.get('url')}")
            parts.append(f"Title: {page.get('title')}")
            parts.append(f"Description: {page.get('description')}")
            parts.append(f"Language: {page.get('language')}")

            content = d.get("content", {})
            headings = content.get("headings", [])[:10]
            if headings:
                parts.append("Headings: " + " | ".join(h["text"] for h in headings if h.get("text")))

            paragraphs = content.get("paragraphs", [])[:8]
            if paragraphs:
                parts.append("Key paragraphs:\n" + "\n".join(f"- {p[:200]}" for p in paragraphs))

            nav = content.get("navigation", [])[:10]
            if nav:
                parts.append("Navigation: " + " | ".join(n["text"] for n in nav if n.get("text")))

            tech = d.get("technology", {})
            all_tech = (
                tech.get("frameworks", []) +
                tech.get("libraries", []) +
                tech.get("analytics", []) +
                tech.get("cdn", [])
            )
            if all_tech:
                parts.append("Technologies: " + ", ".join(all_tech))

            network = d.get("network", {})
            failed = network.get("failed", [])
            if failed:
                parts.append(f"Failed requests: {len(failed)}")

            ws = network.get("websockets", [])
            if ws:
                parts.append(f"WebSockets detected: {len(ws)}")

            perf = d.get("browser", {}).get("performance", {})
            if perf:
                parts.append(f"Load time: {perf.get('load_time_seconds')}s, Requests: {perf.get('request_count')}")

        elif payload.get("type") == "search":
            results = payload.get("results", [])
            if not results:
                parts.append("Search returned no results. Use your own knowledge to answer.")
            else:
                for i, r in enumerate(results[:5], 1):
                    parts.append(f"{i}. {r.get('title')}\n   {r.get('url')}\n   {r.get('body', '')[:300]}")

    if not parts:
        return "No external data collected. Answer from your own knowledge and the date context in the system prompt."

    return "\n".join(parts)
