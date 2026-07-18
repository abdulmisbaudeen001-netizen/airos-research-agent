"""
Browser Intelligence Engine.

Responsibilities: open a URL, collect all publicly observable information,
return a structured dict matching the fixed schema. No analysis or formatting.
"""

import asyncio
import base64
import logging
import time

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)

# Fixed output schema version. Bump if structure changes.
SCHEMA_VERSION = "1.0"


async def collect(url: str) -> dict:
    """
    Open url and return structured page data.
    Always returns the schema — errors are captured inside it.
    """
    start = time.time()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        console_messages = []
        network_requests = []
        websockets = []
        redirect_chain = []
        failed_requests = []

        # --- Event listeners ---
        page.on("console", lambda msg: console_messages.append({
            "type": msg.type,
            "text": msg.text,
        }))

        page.on("websocket", lambda ws: websockets.append(ws.url))

        page.on("requestfailed", lambda req: failed_requests.append({
            "url": req.url,
            "method": req.method,
            "failure": req.failure,
        }))

        page.on("response", lambda res: network_requests.append({
            "url": res.url,
            "method": res.request.method,
            "status": res.status,
        }))

        try:
            response = await page.goto(
                url,
                wait_until="networkidle",
                timeout=30_000,
            )

            # Capture redirect chain
            req = response.request if response else None
            while req:
                redirect_chain.append(req.url)
                req = req.redirected_from

            # Scroll to trigger lazy-loaded content
            await _scroll_page(page)

            # Wait for any post-scroll network activity to settle
            await asyncio.sleep(1)

            final_url = page.url
            title = await page.title()

            # --- Meta extraction ---
            meta = await _extract_meta(page)

            # --- Content extraction ---
            content = await _extract_content(page)

            # --- Technology detection ---
            technologies = await _detect_technologies(page)

            # --- Screenshot (full page, base64) ---
            screenshot_bytes = await page.screenshot(full_page=True)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            elapsed = round(time.time() - start, 2)

            result = {
                "schema_version": SCHEMA_VERSION,
                "error": None,
                "page": {
                    "url": url,
                    "final_url": final_url,
                    "title": title,
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
                    "requests": network_requests[:100],  # cap to avoid huge payloads
                    "websockets": websockets,
                    "redirects": list(reversed(redirect_chain)),
                    "failed": failed_requests,
                },
                "browser": {
                    "console": console_messages,
                    "performance": {
                        "load_time_seconds": elapsed,
                        "request_count": len(network_requests),
                    },
                    "screenshot": screenshot_b64,
                },
            }

        except PlaywrightTimeout:
            logger.warning("Timeout loading %s", url)
            result = _error_schema(url, "Page load timed out after 30 seconds.")
        except Exception as exc:
            logger.error("Browser error for %s: %s", url, exc)
            result = _error_schema(url, str(exc))
        finally:
            await browser.close()

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _scroll_page(page) -> None:
    """Scroll to bottom incrementally to trigger lazy content."""
    try:
        await page.evaluate("""
            async () => {
                await new Promise(resolve => {
                    let total = 0;
                    const step = 400;
                    const timer = setInterval(() => {
                        window.scrollBy(0, step);
                        total += step;
                        if (total >= document.body.scrollHeight) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)
    except Exception:
        pass  # Non-fatal


async def _extract_meta(page) -> dict:
    return await page.evaluate("""
        () => ({
            description: (
                document.querySelector('meta[name="description"]')?.content ||
                document.querySelector('meta[property="og:description"]')?.content ||
                ""
            ),
            language: document.documentElement.lang || "",
            canonical: document.querySelector('link[rel="canonical"]')?.href || "",
        })
    """)


async def _extract_content(page) -> dict:
    return await page.evaluate("""
        () => {
            const text = el => el?.innerText?.trim() || "";
            const attr = (el, a) => el?.getAttribute(a) || "";

            const headings = [];
            document.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach(h => {
                headings.push({ level: h.tagName, text: text(h) });
            });

            const paragraphs = [];
            document.querySelectorAll("p").forEach(p => {
                const t = text(p);
                if (t.length > 20) paragraphs.push(t);
            });

            const tables = [];
            document.querySelectorAll("table").forEach(t => {
                const rows = [];
                t.querySelectorAll("tr").forEach(tr => {
                    rows.push([...tr.querySelectorAll("td,th")].map(c => text(c)));
                });
                if (rows.length) tables.push(rows);
            });

            const forms = [];
            document.querySelectorAll("form").forEach(f => {
                const fields = [...f.querySelectorAll("input,textarea,select")]
                    .map(i => ({ type: i.type || i.tagName, name: attr(i, "name"), placeholder: attr(i, "placeholder") }));
                forms.push({ action: attr(f, "action"), fields });
            });

            const buttons = [];
            document.querySelectorAll("button, [role='button'], input[type='submit']").forEach(b => {
                const t = text(b);
                if (t) buttons.push(t);
            });

            const images = [];
            document.querySelectorAll("img").forEach(img => {
                images.push({ src: attr(img, "src"), alt: attr(img, "alt") });
            });

            const links = [];
            const origin = window.location.origin;
            document.querySelectorAll("a[href]").forEach(a => {
                const href = a.href;
                links.push({
                    text: text(a),
                    href,
                    internal: href.startsWith(origin),
                });
            });

            const navItems = [];
            document.querySelectorAll("nav a").forEach(a => {
                navItems.push({ text: text(a), href: a.href });
            });

            const footerItems = [];
            document.querySelectorAll("footer a").forEach(a => {
                footerItems.push({ text: text(a), href: a.href });
            });

            return {
                headings,
                paragraphs: paragraphs.slice(0, 50),
                tables: tables.slice(0, 10),
                forms,
                buttons: [...new Set(buttons)].slice(0, 30),
                images: images.slice(0, 50),
                links: links.slice(0, 100),
                navigation: navItems,
                footer: footerItems,
            };
        }
    """)


async def _detect_technologies(page) -> dict:
    return await page.evaluate("""
        () => {
            const win = window;
            const doc = document;
            const html = doc.documentElement.innerHTML;

            const frameworks = [];
            if (win.React || win.__REACT_DEVTOOLS_GLOBAL_HOOK__) frameworks.push("React");
            if (win.Vue || win.__vue_app__) frameworks.push("Vue");
            if (win.angular || doc.querySelector("[ng-version]")) frameworks.push("Angular");
            if (win.__NEXT_DATA__) frameworks.push("Next.js");
            if (win.__NUXT__) frameworks.push("Nuxt");

            const libraries = [];
            if (win.jQuery || win.$?.fn?.jquery) libraries.push("jQuery");
            if (html.includes("tailwind")) libraries.push("Tailwind CSS");
            if (html.includes("bootstrap")) libraries.push("Bootstrap");

            const analytics = [];
            if (win.gtag || win.ga || html.includes("google-analytics")) analytics.push("Google Analytics");
            if (win.fbq || html.includes("facebook.net")) analytics.push("Meta Pixel");
            if (win.Intercom) analytics.push("Intercom");
            if (win.HubSpotConversations || html.includes("hs-scripts")) analytics.push("HubSpot");
            if (win.Hotjar) analytics.push("Hotjar");

            const cdn = [];
            if (html.includes("cloudflare")) cdn.push("Cloudflare");
            if (html.includes("amazonaws.com")) cdn.push("AWS");
            if (html.includes("googleapis.com")) cdn.push("Google APIs");
            if (html.includes("fastly")) cdn.push("Fastly");

            return { frameworks, libraries, analytics, cdn };
        }
    """)


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
