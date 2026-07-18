# AIROS Research Agent (ARA) v1.0

A lightweight AI research assistant that runs as a Telegram bot.

Send it a natural language message. It figures out whether to analyze a website,
search the web, or answer from knowledge — then returns a structured response.

---

## What it can do

- **Website analysis** — open a real browser, collect all public information, produce a structured report
- **Web research** — search DuckDuckGo, read results, synthesize a researched answer
- **General questions** — answer directly using the LLM when no live data is needed
- **Comparisons** — gather information about multiple subjects in parallel
- **Verification** — check claims against current search results

---

## Architecture

```
Telegram
    │
    ▼
Structural Router
    │
    ├── Bare URL / "analyze <target>"  →  Browser Engine (Playwright)
    │
    └── Everything else  →  Intent Classifier (LLM, one fast call)
                                │
                                ├── website_analysis  →  Browser
                                ├── research          →  Search / Browser
                                └── general           →  LLM directly
                                          │
                                          ▼
                                 Final LLM Reasoning
                                          │
                                          ▼
                                      Telegram
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `LLM_MODEL` | Yes | OpenRouter model string (e.g. `google/gemini-flash-1.5`) |
| `TELEGRAM_BOT_TOKEN` | Yes | From @BotFather on Telegram |
| `HOST` | No | Default: `0.0.0.0` |
| `PORT` | No | Default: `8000` |

Copy `.env.example` to `.env` and fill in the values.

---

## Local setup

**Prerequisites:** Python 3.11+

```bash
git clone <your-repo>
cd airos-research-agent

pip install -r requirements.txt
playwright install chromium
playwright install-deps

cp .env.example .env
# Edit .env with your keys

python app.py
```

The bot starts polling immediately. Send a message to your bot in Telegram.

---

## Changing the LLM model

Change only the `LLM_MODEL` environment variable. No code changes needed.

Free models on OpenRouter that work well:
- `google/gemini-flash-1.5`
- `google/gemini-2.0-flash-exp:free`
- `deepseek/deepseek-chat`
- `meta-llama/llama-3.1-8b-instruct:free`

**Important:** Test your chosen model supports returning plain JSON before deploying.
Some free models ignore formatting instructions. If intent classification breaks,
switch to a more instruction-following model.

---

## Deploying to Render

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Render will detect `render.yaml` automatically.
5. In Render's **Environment** tab, add:
   - `OPENROUTER_API_KEY`
   - `LLM_MODEL`
   - `TELEGRAM_BOT_TOKEN`
6. Deploy.

The bot starts polling Telegram as soon as the service is running.
No webhook configuration is required — the bot connects outward to Telegram.

**Health check:** `GET /health` returns `{"status": "ok"}`.
Use this URL in Render's health check settings.

### Render free tier note

Render's free web services spin down after 15 minutes of no HTTP traffic.
When the service spins down, the bot goes offline until the service wakes up.

To keep it always online for free: use [UptimeRobot](https://uptimerobot.com)
(free tier) to ping `/health` every 10 minutes. This prevents spin-down.

---

## Search provider

The default search implementation uses the `duckduckgo-search` (ddgs) Python library.
This requires no API key and is suitable for personal use and development.

Because it relies on an unofficial integration, occasional maintenance may be required
if the upstream service changes. The search layer is intentionally abstracted behind
a `SearchProvider` interface in `search.py` so a supported API provider such as
Brave Search or Tavily can be adopted later without affecting the rest of the application.

To swap providers: implement `SearchProvider` in `search.py` and update `get_provider()`.

---

## Telegram bot setup

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the prompts.
3. Copy the token you receive into `TELEGRAM_BOT_TOKEN`.

The bot uses long polling — no public URL or HTTPS certificate needed.

---

## Project structure

```
airos-research-agent/
├── app.py            Entry point. FastAPI + polling loop.
├── config.py         Environment variable loading.
├── browser.py        Playwright browser engine. Collect only, no analysis.
├── search.py         Search provider interface + DuckDuckGo implementation.
├── llm.py            OpenRouter integration. Intent classification + reasoning.
├── report.py         Orchestrates tools and calls LLM for final response.
├── telegram_bot.py   Telegram long polling, routing, conversation memory.
├── utils.py          Shared helpers (URL detection, pattern matching).
├── requirements.txt
├── render.yaml       Render deployment config.
├── .env.example
└── README.md
```

---

## Version 1 — out of scope

These are intentionally deferred to future versions:
- Long-term memory / databases
- User authentication
- Multi-agent collaboration
- Scheduled background tasks
- Browser session persistence
- Web dashboard
  
