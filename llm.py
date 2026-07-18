"""
LLM Engine — OpenRouter integration.

All messages flow through the LLM:
  1. classify_intent()  — LLM decides what the user wants and where to route it
  2. reason()           — LLM reasons over all collected data and produces the final response
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


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _build_intent_prompt() -> str:
    now = datetime.datetime.utcnow()
    today = now.strftime("%A, %B %d, %Y")
    tomorrow = (now + datetime.timedelta(days=1)).strftime("%A, %B %d, %Y")
    return f"""You are an intent classifier for an AI research assistant.

Today is {today} (UTC). Tomorrow is {tomorrow}.

Classify the user's message into exactly one intent and return ONLY valid JSON.
No markdown, no explanation, no extra text — just the JSON object.

Possible intents:
- "website_analysis": user wants to analyze, inspect, or learn about a specific website or company's web presence
- "research": user wants current information, news, weather, events, dates, comparisons, or any facts that may require searching the web
- "general": user is asking a conceptual or educational question that can be answered purely from knowledge, with no need for current data

Rules for routing:
- ANY question about time, dates, days, current events, news, weather, prices, or people's current status → "research"
- Questions like "what is tomorrow", "what day is it", "what is happening in X" → "research"
- Questions about stable concepts, definitions, or history → "general"
- When in doubt, route to "research" not "general"

For "website_analysis" and "research", extract the primary target or query.
For comparisons, set execution to "parallel" and list multiple targets.
For verification (check a claim against sources), set execution to "sequential".

Response schema:
{{
  "intent": "website_analysis" | "research" | "general",
  "execution": "single" | "parallel" | "sequential",
  "tools": ["browser"] | ["search"] | ["browser", "search"] | [],
  "targets": ["<primary target>"],
  "query": "<search query if research intent>",
  "confidence": 0.0-1.0
}}

Examples:

User: "What is FastAPI?"
{{"intent":"general","execution":"single","tools":[],"targets":[],"query":"","confidence":0.95}}

User: "What is tomorrow?"
{{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"date tomorrow {tomorrow}","confidence":0.99}}

User: "What day is today?"
{{"intent":"general","execution":"single","tools":[],"targets":[],"query":"","confidence":0.99}}

User: "What is going on in Nigeria?"
{{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"Nigeria latest news today","confidence":0.97}}

User: "Latest news about AI agents"
{{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"latest AI agents news today","confidence":0.97}}

User: "Compare Railway and Render"
{{"intent":"research","execution":"parallel","tools":["search"],"targets":["Railway hosting","Render hosting"],"query":"","confidence":0.95}}

User: "Analyze github.com"
{{"intent":"website_analysis","execution":"single","tools":["browser"],"targets":["https://github.com"],"query":"","confidence":0.99}}

User: "Go to stripe.com and compare its pricing with PayPal"
{{"intent":"research","execution":"parallel","tools":["browser","search"],"targets":["https://stripe.com","PayPal pricing"],"query":"","confidence":0.93}}

User: "Is it true that Cloudflare went public in 2019?"
{{"intent":"research","execution":"sequential","tools":["search"],"targets":[],"query":"Cloudflare IPO date history","confidence":0.96}}"""


async def classify_intent(message: str) -> dict:
    """
    Every message passes through the LLM for intent classification.
    Falls back to research intent (not general) on any failure — safer default.
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
        # Default to research so the bot at least tries to search
        return {
            "intent": "research",
            "execution": "single",
            "tools": ["search"],
            "targets": [],
            "query": message,
            "confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# Main reasoning call
# ---------------------------------------------------------------------------

def _build_reasoning_prompt() -> str:
    now = datetime.datetime.utcnow()
    today = now.strftime("%A, %B %d, %Y")
    tomorrow = (now + datetime.timedelta(days=1)).strftime("%A, %B %d, %Y")
    current_time = now.strftime("%H:%M UTC")

    return f"""You are AIROS Research Agent, a professional AI research assistant.

Current date: {today}
Tomorrow: {tomorrow}
Current time: {current_time}

Every message from the user has already been routed through the intent classifier.
You now receive the user's message plus any data collected (search results, browser data, or nothing).
Your job is to produce a clear, accurate, well-organized response.

Rules:
- You always know today's date and tomorrow's date from the context above. Use it.
- If the user asks about time or dates, answer directly from the date context above.
- If search or browser data was collected, base your response on it.
- If no data was collected but you know the answer from general knowledge, answer it.
- Only say you don't know if you genuinely cannot answer even with the date context and your knowledge.
- Distinguish clearly between what you observed in data and what you know from knowledge.
- Keep responses concise and readable in Telegram (plain text, no HTML).
- Use short paragraphs. Use dashes for lists, not markdown bullets.
- For website analysis, follow the report structure below.
- For research questions, synthesize findings into a direct answer.
- For general knowledge questions, answer directly and confidently.

Website Report Structure (use when analyzing a site):
1. Purpose & Overview
2. Target Audience
3. Core Features / Offerings
4. Technologies Detected
5. Navigation & UX Observations
6. Notable Network Activity
7. Strengths
8. Weaknesses / Gaps
9. Overall Assessment

Keep each section brief. Telegram messages have no length limit but users prefer concise."""


async def reason(user_message: str, collected_data: dict, conversation_history: list[dict]) -> str:
    """
    Final reasoning step. Every response passes through the LLM.
    Takes the user's original message, all collected data, and conversation history.
    Returns the formatted response string.
    """
    data_summary = _format_collected_data(collected_data)

    messages = [
        {"role": "system", "content": _build_reasoning_prompt()},
    ]

    # Include short-term conversation history (last 6 exchanges max)
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
        return "The analysis took too long to complete. Please try again."
    except Exception as exc:
        logger.error("LLM reasoning failed: %s", exc)
        return "Something went wrong while generating the response. Please try again."


# ---------------------------------------------------------------------------
# Data formatter
# ---------------------------------------------------------------------------

def _format_collected_data(data: dict) -> str:
    """
    Converts collected data into a compact text representation for the LLM.
    Drops the screenshot (binary) and caps large arrays to avoid token overflow.
    """
    if not data:
        return "No external data collected — answer from knowledge and date context."

    parts = []

    for source, payload in data.items():
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
                parts.append("Search returned no results.")
            else:
                for i, r in enumerate(results[:5], 1):
                    parts.append(f"{i}. {r.get('title')}\n   {r.get('url')}\n   {r.get('body', '')[:300]}")

        elif payload.get("type") == "general":
            parts.append("No external data collected — answer from knowledge and date context.")

    return "\n".join(parts)
