import os
from dotenv import load_dotenv

load_dotenv()


def require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


OPENROUTER_API_KEY = require("OPENROUTER_API_KEY")
LLM_MODEL = require("LLM_MODEL")
TELEGRAM_BOT_TOKEN = require("TELEGRAM_BOT_TOKEN")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
