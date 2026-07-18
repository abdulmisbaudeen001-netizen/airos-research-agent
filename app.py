"""
AIROS Research Agent — Entry Point.

Starts a FastAPI app (for /health and Render compatibility) and runs
the Telegram polling loop as a background task on the same event loop.
"""

import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from config import HOST, PORT
from telegram_bot import build_application, start_polling

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telegram app (module-level so lifespan can reference it)
# ---------------------------------------------------------------------------

telegram_app = build_application()


# ---------------------------------------------------------------------------
# FastAPI lifespan — starts polling on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AIROS Research Agent starting...")
    # Start polling in the background; it runs until the process exits
    asyncio.create_task(start_polling(telegram_app))
    logger.info("Telegram polling task started.")
    yield
    # Shutdown
    logger.info("Shutting down...")
    try:
        await telegram_app.updater.stop()
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
