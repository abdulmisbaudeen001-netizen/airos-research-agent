"""
Browser Intelligence Engine.

Responsibilities: open a URL, collect all publicly observable information,
return a structured dict matching the fixed schema. No analysis or formatting.
"""

import asyncio
import logging
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


async def collect(url: str) -> dict:
    start = time.time()
    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            response = await client.get(url)

        redirects = [str(r.url) for r in response.history] + [str(response.url)]
        final_url = str(response.url)
        origin = urlparse(final_url).scheme + "://" + urlparse(final_url).netloc

        soup = BeautifulSoup(response.text, "html.parser")

        meta = _extract_meta(soup)
        content = _extract_content(soup, origin)
        technologies = _detect_technologies(soup, response.text)

        elapsed = round(time.time() - start, 2)

        return {
            "schema_version": SCHEMA_VERSION,
            "error": None,
            "page": {
                "url": url,
                "final_url": final_url,
                "title": soup.title.string.strip() if soup.title else "",
                "description": meta.get("description", ""),
                "language": meta.get("language", ""),
                "canonical": meta.get("canonical", ""),
            },
            "content": {
                "headings": content["headings"],
                "paragraphs": content["paragraphs"],
                "tables": content["tables"],
                "forms": content["forms"],
                "buttons": content["buttons"],
                "images": content["images"],
                "links": content["links"],
                "navigation": content["navigation"],
                "footer": content["footer"],
            },
            "technology": technologies,
            "network": {
                "requests": [],
                "websockets": [],
                "redirects": redirects,
                "failed": [],
            },
            "browser": {
                "console": [],
                "performance": {
                    "load_time_seconds": elapsed,
                    "request_count": 1,
                },
                "screenshot": "",
            },
        }

    except httpx.TimeoutException:
        return _error_schema(url, "Page load timed out after 30 seconds.")
    except Exception as exc:
        logger.error("Browser error for %s: %s", url, exc)
        return _error_schema(url, str(exc))


def _extract_meta(soup: BeautifulSoup) -> dict:
    description = ""
    tag = soup.find("meta", attrs={"name": "description"})
    if tag:
        description = tag.get("content", "")
    if not description:
        tag = soup.find("meta", attrs={"property": "og:description"})
        if tag:
            description = tag.get("content", "")

    language = soup.html.get("lang", "") if soup.html else ""

    canonical = ""
    tag = soup.find("link", attrs={"rel": "canonical"})
    if tag:
        canonical = tag.get("href", "")

    return {"description": description, "language": language, "canonical": canonical}


def _extract_content(soup: BeautifulSoup, origin: str) -> dict:
    headings = []
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = h.get_text(strip=True)
        if text:
            headings.append({"level": h.name.upper(), "text": text})

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 20:
            paragraphs.append(text)

    tables = []
    for table in soup.find_all("table")[:10]:
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    forms = []
    for form in soup.find_all("form"):
        fields = []
        for field in form.find_all(["input", "textarea", "select"]):
            fields.append({
                "type": field.get("type", field.name),
                "name": field.get("name", ""),
                "placeholder": field.get("placeholder", ""),
            })
        forms.append({"action": form.get("action", ""), "fields": fields})

    buttons = list({
        b.get_text(strip=True)
        for b in soup.find_all(["button", "input"])
        if b.get_text(strip=True) or b.get("value")
    })[:30]

    images = [
        {"src": img.get("src", ""), "alt": img.get("alt", "")}
        for img in soup.find_all("img")
    ][:50]

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = origin + href
        links.append({
            "text": a.get_text(strip=True),
            "href": href,
            "internal": href.startswith(origin),
        })

    navigation = []
    for nav in soup.find_all("nav"):
        for a in nav.find_all("a", href=True):
            navigation.append({"text": a.get_text(strip=True), "href": a["href"]})

    footer = []
    for f in soup.find_all("footer"):
        for a in f.find_all("a", href=True):
            footer.append({"text": a.get_text(strip=True), "href": a["href"]})

    return {
        "headings": headings,
        "paragraphs": paragraphs[:50],
        "tables": tables,
        "forms": forms,
        "buttons": buttons,
        "images": images,
        "links": links[:100],
        "navigation": navigation,
        "footer": footer,
    }


def _detect_technologies(soup: BeautifulSoup, html: str) -> dict:
    frameworks = []
    if "__NEXT_DATA__" in html or "next/dist" in html:
        frameworks.append("Next.js")
    if "data-reactroot" in html or "_react" in html.lower():
        frameworks.append("React")
    if "ng-version" in html or "angular" in html.lower():
        frameworks.append("Angular")
    if "__NUXT__" in html:
        frameworks.append("Nuxt")
    if "vue" in html.lower() and ("v-app" in html or "__vue" in html):
        frameworks.append("Vue")

    libraries = []
    scripts = " ".join(s.get("src", "") for s in soup.find_all("script"))
    if "jquery" in scripts.lower() or "jquery" in html.lower():
        libraries.append("jQuery")
    if "tailwind" in html.lower():
        libraries.append("Tailwind CSS")
    if "bootstrap" in html.lower():
        libraries.append("Bootstrap")

    analytics = []
    if "google-analytics" in html or "gtag" in html or "ga.js" in html:
        analytics.append("Google Analytics")
    if "facebook.net" in html or "fbq" in html:
        analytics.append("Meta Pixel")
    if "intercom" in html.lower():
        analytics.append("Intercom")
    if "hs-scripts" in html or "hubspot" in html.lower():
        analytics.append("HubSpot")
    if "hotjar" in html.lower():
        analytics.append("Hotjar")

    cdn = []
    if "cloudflare" in html.lower():
        cdn.append("Cloudflare")
    if "amazonaws.com" in html:
        cdn.append("AWS")
    if "googleapis.com" in html:
        cdn.append("Google APIs")
    if "fastly" in html.lower():
        cdn.append("Fastly")

    return {"frameworks": frameworks, "libraries": libraries, "analytics": analytics, "cdn": cdn}


def _error_schema(url: str, message: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "error": message,
        "page": {"url": url, "final_url": "", "title": "", "description": "", "language": "", "canonical": ""},
        "content": {"headings": [], "paragraphs": [], "tables": [], "forms": [], "buttons": [], "images": [], "links": [], "navigation": [], "footer": []},
        "technology": {"frameworks": [], "libraries": [], "analytics": [], "cdn": []},
        "network": {"requests": [], "websockets": [], "redirects": [], "failed": []},
        "browser": {"console": [], "performance": {}, "screenshot": ""},
    }
