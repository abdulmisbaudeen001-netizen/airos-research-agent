"""
LLM Engine — OpenRouter integration.

Two responsibilities:
  1. classify_intent()  — fast call to determine what the user wants
  2. reason()           — full reasoning call over collected data
"""

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

INTENT_SYSTEM_PROMPT = """You are an intent classifier for an AI research assistant.

Classify the user's message into exactly one intent and return ONLY valid JSON.
No markdown, no explanation, no extra text — just the JSON object.

Possible intents:
- "website_analysis": user wants to analyze, inspect, or learn about a specific website or company's web presence
- "research": user wants current information, news, weather, comparisons, or facts that require searching the web
- "general": user is asking a conceptual or educational question that can be answered from knowledge

For "website_analysis" and "research", extract the primary target or query.
For comparisons, set execution to "parallel" and list multiple targets.
For verification (check a claim against sources), set execution to "sequential".

Response schema:
{
  "intent": "website_analysis" | "research" | "general",
  "execution": "single" | "parallel" | "sequential",
  "tools": ["browser"] | ["search"] | ["browser", "search"] | [],
  "targets": ["<primary target>"],
  "query": "<search query if research intent>",
  "confidence": 0.0–1.0
}

Examples:

User: "What is FastAPI?"
{"intent":"general","execution":"single","tools":[],"targets":[],"query":"","confidence":0.95}

User: "Latest news about AI agents"
{"intent":"research","execution":"single","tools":["search"],"targets":[],"query":"latest AI agents news 2025","confidence":0.97}

User: "Compare Railway and Render"
{"intent":"research","execution":"parallel","tools":["search"],"targets":["Railway hosting","Render hosting"],"query":"","confidence":0.95}

User: "Analyze github.com"
{"intent":"website_analysis","execution":"single","tools":["browser"],"targets":["https://github.com"],"query":"","confidence":0.99}

User: "Go to stripe.com and compare its pricing with PayPal"
{"intent":"research","execution":"parallel","tools":["browser","search"],"targets":["https://stripe.com","PayPal pricing"],"query":"","confidence":0.93}

User: "Is it true that Cloudflare went public in 2019?"
{"intent":"research","execution":"sequential","tools":["search"],"targets":[],"query":"Cloudflare IPO date history","confidence":0.96}"""


async def classify_intent(message: str) -> dict:
    """
    Returns an intent dict. Falls back to general intent on any failure.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
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
        # Strip markdown fences if model ignores the instruction
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

REASONING_SYSTEM_PROMPT = """You are AIROS Research Agent, a professional AI research assistant.

You receive structured data collected from websites and/or web searches.
Your job is to analyze this data and produce a clear, accurate, well-organized response.

Rules:
- Base your response only on the provided data. Do not invent facts.
- Distinguish clearly between what you observed and what you infer.
- If data is missing or incomplete, say so rather than guessing.
- Keep responses concise and readable in Telegram (plain text, no HTML).
- Use short paragraphs. Use dashes for lists, not markdown bullets.
- For website analysis, follow the report structure below.
- For research questions, synthesize findings into a direct answer.

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
    Final reasoning step. Takes the user's original message and all collected data.
    Returns the formatted response string.
    """
    # Build a compact representation of collected data for the prompt
    data_summary = _format_collected_data(collected_data)

    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
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


def _format_collected_data(data: dict) -> str:
    """
    Converts collected data into a compact text representation for the LLM.
    Drops the screenshot (binary) and caps large arrays to avoid token overflow.
    """
    if not data:
        return "No data collected."

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
                parts.append("No search results found.")
            else:
                for i, r in enumerate(results[:5], 1):
                    parts.append(f"{i}. {r.get('title')}\n   {r.get('url')}\n   {r.get('body', '')[:300]}")

        elif payload.get("type") == "general":
            parts.append("No external data collected — answering from knowledge.")

    return "\n".join(parts)
