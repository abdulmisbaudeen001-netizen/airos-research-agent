"""
AIROS Research Agent — Entry Point.

Webhook mode: Render sleeps when idle.
Telegram wakes Render by posting to /webhook when a message arrives.
Render processes it and sleeps again — no idle hours wasted.

Web interface available at / — supports both voice and text.
"""

import logging
import os
import uuid
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from telegram import Update

from config import HOST, PORT, TELEGRAM_BOT_TOKEN
from telegram_bot import build_application
import report
import llm
import voice

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio temp directory
# ---------------------------------------------------------------------------

AUDIO_DIR = "/tmp/airos_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Telegram app
# ---------------------------------------------------------------------------

telegram_app = build_application()

# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

WEBHOOK_PATH = "/webhook"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AIROS Research Agent starting...")

    await telegram_app.initialize()
    await telegram_app.start()

    render_url = f"https://airos-research-agent.onrender.com{WEBHOOK_PATH}"
    await telegram_app.bot.set_webhook(
        url=render_url,
        drop_pending_updates=True,
    )
    logger.info("Webhook registered: %s", render_url)

    yield

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

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AIROS Research Agent v1.0"}


@app.post(WEBHOOK_PATH)
async def webhook(request: Request) -> Response:
    """Telegram webhook — wakes Render on demand."""
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return Response(status_code=200)


@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """
    Web interface endpoint for both voice and text tabs.
    Returns text response + audio URL for voice playback.
    """
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        want_audio = data.get("audio", False)

        if not message:
            return JSONResponse({"response": "No message received."})

        # Run through full LLM pipeline
        intent = await llm.classify_intent(message)
        response_text = await report.generate(message, intent, [])

        result = {"response": response_text}

        # Generate audio if voice tab requested it
        if want_audio:
            audio_id = uuid.uuid4().hex
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.mp3")
            success = await voice.text_to_speech(response_text, audio_path)
            if success:
                result["audio_url"] = f"/audio/{audio_id}"

        return JSONResponse(result)

    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
        return JSONResponse(
            {"response": "Something went wrong. Please try again."},
            status_code=500,
        )


@app.get("/audio/{audio_id}")
async def serve_audio(audio_id: str) -> FileResponse:
    """Serve generated TTS audio file."""
    # Sanitize — only allow hex IDs
    if not audio_id.isalnum() or len(audio_id) != 32:
        return Response(status_code=400)
    path = os.path.join(AUDIO_DIR, f"{audio_id}.mp3")
    if not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


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
    
