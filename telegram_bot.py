"""
Telegram Bot — long polling interface.

Handles incoming messages, sends progress updates, manages short-term
conversation memory, and calls the report orchestrator.
"""

import asyncio
import logging

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.constants import ChatAction

import llm
import report
from utils import is_url, extract_analyze_target, normalize_url
from config import TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)

# In-memory conversation history per chat_id
# Format: { chat_id: [{"role": ..., "content": ...}, ...] }
_history: dict[int, list[dict]] = {}
MAX_HISTORY = 12  # Keep last 6 exchanges (12 messages)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "AIROS Research Agent ready.\n\n"
        "You can:\n"
        "- Analyze a website: analyze stripe.com\n"
        "- Research a topic: latest AI news\n"
        "- Ask a question: what is FastAPI?\n"
        "- Compare things: compare Railway and Render\n\n"
        "Just send a message naturally."
    )


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_message = update.message.text.strip()

    if not user_message:
        return

    logger.info("[%s] Received: %s", chat_id, user_message[:80])

    # --- Structural router ---
    intent = _structural_route(user_message)

    if intent:
        logger.info("[%s] Structural route: %s", chat_id, intent["intent"])
        await _send_typing(update, context)
        await update.message.reply_text("Opening website...")
    else:
        # --- LLM intent classifier ---
        await _send_typing(update, context)
        await update.message.reply_text("Thinking...")
        intent = await llm.classify_intent(user_message)
        logger.info("[%s] LLM intent: %s (%.2f)", chat_id, intent.get("intent"), intent.get("confidence", 0))

        # Inform user what we're about to do
        action_msg = _describe_action(intent)
        if action_msg:
            await update.message.reply_text(action_msg)

    # --- Execute ---
    history = _history.get(chat_id, [])

    try:
        response = await report.generate(user_message, intent, history)
    except Exception as exc:
        logger.error("[%s] Report generation failed: %s", chat_id, exc)
        response = "Something went wrong. Please try again."

    # --- Update conversation history ---
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    _history[chat_id] = history[-MAX_HISTORY:]

    # --- Send response ---
    # Telegram has a 4096 char limit per message
    for chunk in _split_message(response):
        await update.message.reply_text(chunk)

    logger.info("[%s] Response sent (%d chars)", chat_id, len(response))


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Telegram error: %s", context.error)


# ---------------------------------------------------------------------------
# Structural router
# ---------------------------------------------------------------------------

def _structural_route(text: str) -> dict | None:
    """
    Returns an intent dict for unambiguous inputs, or None to fall through
    to the LLM classifier.
    """
    # "analyze <target>" pattern
    target = extract_analyze_target(text)
    if target:
        url = normalize_url(target)
        return {
            "intent": "website_analysis",
            "execution": "single",
            "tools": ["browser"],
            "targets": [url],
            "query": "",
            "confidence": 1.0,
        }

    # Bare URL
    if is_url(text):
        url = normalize_url(text)
        return {
            "intent": "website_analysis",
            "execution": "single",
            "tools": ["browser"],
            "targets": [url],
            "query": "",
            "confidence": 1.0,
        }

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_action(intent: dict) -> str:
    i = intent.get("intent")
    tools = intent.get("tools", [])
    targets = intent.get("targets", [])

    if i == "website_analysis" and targets:
        return f"Analyzing {targets[0]}..."
    if i == "research":
        if "browser" in tools and "search" in tools:
            return "Browsing and searching for information..."
        if "browser" in tools:
            return "Opening website..."
        return "Searching the web..."
    if i == "general":
        return ""
    return "Working on it..."


def _split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


async def _send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Bot lifecycle
# ---------------------------------------------------------------------------

def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_error_handler(error_handler)
    return app


async def start_polling(app: Application) -> None:
    """Start long polling — runs until the event loop is cancelled."""
    logger.info("Starting Telegram long polling...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Bot is online and polling.")
