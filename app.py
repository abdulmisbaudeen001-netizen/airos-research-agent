"""
AIROS Research Agent — Entry Point.

Webhook mode: Render sleeps when idle.
Telegram wakes Render by posting to /webhook when a message arrives.
Render processes it and sleeps again — no idle hours wasted.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from telegram import Update

from config import HOST, PORT, TELEGRAM_BOT_TOKEN
from telegram_bot import build_application

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telegram app
# ---------------------------------------------------------------------------

telegram_app = build_application()

# ---------------------------------------------------------------------------
# FastAPI lifespan — registers webhook on startup, removes on shutdown
# ---------------------------------------------------------------------------

WEBHOOK_PATH = "/webhook"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AIROS Research Agent starting...")

    await telegram_app.initialize()
    await telegram_app.start()

    # Tell Telegram to send all updates to our Render URL
    render_url = f"https://airos-research-agent.onrender.com{WEBHOOK_PATH}"
    await telegram_app.bot.set_webhook(
        url=render_url,
        drop_pending_updates=True,
    )
    logger.info("Webhook registered: %s", render_url)

    yield

    # Shutdown — remove webhook so Telegram stops sending
    logger.info("Shutting down...")
    try:
        await telegram_app.bot.delete_webhook()
        await telegram_app.stop()
        await telegram_app.shutdown()
    except Exception as exc:
        logger.warning("Shutdown warning: %s", exc)
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AIROS Research Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AIROS Research Agent v1.0"}


@app.post(WEBHOOK_PATH)
async def webhook(request: Request) -> Response:
    """
    Telegram calls this endpoint every time a user sends a message.
    This is what wakes Render up on demand.
    """
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
