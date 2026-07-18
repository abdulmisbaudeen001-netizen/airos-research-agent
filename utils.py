import re
import logging

logger = logging.getLogger(__name__)

# Matches bare URLs or URLs with protocol
URL_PATTERN = re.compile(
    r"^(https?://)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(/\S*)?$",
    re.IGNORECASE,
)

# Matches "analyze <target>" command
ANALYZE_PATTERN = re.compile(
    r"^analyze\s+(.+)$",
    re.IGNORECASE,
)


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.match(text.strip()))


def extract_analyze_target(text: str) -> str | None:
    """Returns the target from 'analyze <target>', or None if no match."""
    match = ANALYZE_PATTERN.match(text.strip())
    return match.group(1).strip() if match else None


def normalize_url(url: str) -> str:
    """Ensures URL has a protocol prefix."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def truncate(text: str, max_chars: int = 3000) -> str:
    """Truncates text to avoid overwhelming the LLM context."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"
