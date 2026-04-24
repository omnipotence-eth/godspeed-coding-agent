"""Browser automation tool — control a browser for web tasks.

This tool allows the agent to take screenshots, navigate pages,
test applications, and verify visual changes.
"""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Check if playwright is available
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class BrowserTool(Tool):
    """Control a browser for web automation tasks.

    Use cases:
    - Screenshot a web page for visual verification
    - Test that a UI change works in a real browser
    - Scrape dynamic content that requires JavaScript
    - Verify web application functionality
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Control a browser for web automation. Takes screenshots, "
            "navigates pages, interacts with elements, and captures page state. "
            "Requires playwright: pip install playwright && playwright install chromium\n\n"
            "Example: browser(action='screenshot', url='https://example.com')\n"
            "Example: browser(action='navigate', url='https://example.com', action='click', selector='#button')\n"
            "Example: browser(action='get_html', url='https://example.com')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW  # Read-only browser interactions by default

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["screenshot", "navigate", "get_html", "click", "fill", "get_text"],
                    "description": "Browser action to perform",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (required for navigate, screenshot, get_html)",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for click, fill, or get_text actions",
                },
                "text": {
                    "type": "string",
                    "description": "Text to fill into an input field",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Take full page screenshot (default: viewport only)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if not PLAYWRIGHT_AVAILABLE:
            return ToolResult.failure(
                "playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        action = arguments.get("action", "")
        url = arguments.get("url", "")
        selector = arguments.get("selector", "")
        text = arguments.get("text", "")
        full_page = arguments.get("full_page", False)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                if action == "screenshot":
                    if not url:
                        return ToolResult.failure("url required for screenshot action")
                    await page.goto(url, wait_until="networkidle")
                    screenshot = await page.screenshot(full_page=full_page)
                    await browser.close()
                    import base64

                    b64 = base64.b64encode(screenshot).decode()
                    return ToolResult.success(
                        f"Screenshot captured ({len(screenshot)} bytes)",
                        extra={"screenshot_b64": b64[:100] + "...(truncated)"},
                    )

                elif action == "navigate":
                    if not url:
                        return ToolResult.failure("url required for navigate action")
                    await page.goto(url, wait_until="networkidle")
                    title = await page.title()
                    await browser.close()
                    return ToolResult.success(f"Navigated to: {title}")

                elif action == "get_html":
                    if not url:
                        return ToolResult.failure("url required for get_html action")
                    await page.goto(url, wait_until="networkidle")
                    html = await page.content()
                    await browser.close()
                    return ToolResult.success(f"HTML content ({len(html)} chars)", extra={"html": html[:500]})

                elif action == "click":
                    if not selector:
                        return ToolResult.failure("selector required for click action")
                    await page.click(selector)
                    await browser.close()
                    return ToolResult.success(f"Clicked: {selector}")

                elif action == "fill":
                    if not selector or not text:
                        return ToolResult.failure("selector and text required for fill action")
                    await page.fill(selector, text)
                    await browser.close()
                    return ToolResult.success(f"Filled '{text}' into {selector}")

                elif action == "get_text":
                    if not selector:
                        return ToolResult.failure("selector required for get_text action")
                    text = await page.text_content(selector)
                    await browser.close()
                    return ToolResult.success(f"Text: {text}")

                else:
                    return ToolResult.failure(f"Unknown action: {action}")

            except Exception as e:
                await browser.close()
                return ToolResult.failure(f"Browser error: {e}")
