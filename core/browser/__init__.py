"""
Browser automation module for the Sovereign Venture Engine.

Provides AI-driven web scraping and data extraction using browser-use
and Playwright. Includes "Sovereign CCTV" — automatic session recording
for debugging agent behavior.

Components:
- SovereignBrowser: High-level wrapper with session lifecycle management
- BrowserConfig: Pydantic configuration model

Usage:
    from core.browser import SovereignBrowser, BrowserConfig

    async with SovereignBrowser() as browser:
        result = await browser.search_and_extract(
            url="https://example.com",
            instructions="Extract the main heading",
        )
"""

from core.browser.config import BrowserConfig
from core.browser.browser_tool import SovereignBrowser, BrowserSessionError

__all__ = ["SovereignBrowser", "BrowserConfig", "BrowserSessionError"]
