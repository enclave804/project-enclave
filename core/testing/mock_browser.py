"""
Mock browser for testing without a real browser or Playwright.

Provides MockBrowserSession and MockAgentHistory that satisfy the
interfaces used by SovereignBrowser, so agent tests can run fast
without launching Chromium.

Usage:
    from core.testing.mock_browser import MockBrowserSession, create_mock_browser

    # Option 1: Use MockBrowserSession directly
    session = MockBrowserSession()

    # Option 2: Create a fully mocked SovereignBrowser
    browser = create_mock_browser(
        mock_result="Extracted heading: Example Domain",
        mock_urls=["https://example.com"],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

logger = logging.getLogger(__name__)


# ─── Mock Agent History ──────────────────────────────────────────────


@dataclass
class MockAgentHistory:
    """
    Mock for browser-use's AgentHistoryList.

    Provides all the methods that SovereignBrowser._build_result() calls.
    """

    _final_result: Optional[str] = None
    _is_done: bool = True
    _is_successful: bool = True
    _has_errors: bool = False
    _errors: list[str] = field(default_factory=list)
    _urls: list[str] = field(default_factory=list)
    _steps: int = 3

    def final_result(self) -> Optional[str]:
        return self._final_result

    def is_done(self) -> bool:
        return self._is_done

    def is_successful(self) -> Optional[bool]:
        return self._is_successful

    def has_errors(self) -> bool:
        return self._has_errors

    def errors(self) -> list[Optional[str]]:
        return self._errors

    def urls(self) -> list[Optional[str]]:
        return self._urls

    def number_of_steps(self) -> int:
        return self._steps

    def action_names(self) -> list[str]:
        return ["navigate", "extract", "done"]

    def model_actions(self) -> list[dict]:
        return []

    def total_duration_seconds(self) -> float:
        return 2.5


class MockBrowserSession:
    """
    Mock for browser-use's BrowserSession.

    Satisfies the interface used by SovereignBrowser:
    - Can be passed as `browser=` to Agent
    - Has start() and stop() methods
    """

    def __init__(self, **kwargs):
        self._started = False
        self._stopped = False
        self._kwargs = kwargs
        logger.info("MockBrowserSession initialized (test mode)")

    async def start(self):
        self._started = True

    async def stop(self):
        self._stopped = True

    async def kill(self):
        self._stopped = True

    async def get_browser_state_summary(self):
        return "Mock browser state"


# ─── Factory ─────────────────────────────────────────────────────────


def create_mock_history(
    result: Optional[str] = "Mock extracted content",
    success: bool = True,
    urls: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
    steps: int = 3,
) -> MockAgentHistory:
    """Create a MockAgentHistory with the specified behavior."""
    return MockAgentHistory(
        _final_result=result,
        _is_done=success,
        _is_successful=success,
        _has_errors=bool(errors),
        _errors=errors or [],
        _urls=urls or ["https://example.com"],
        _steps=steps,
    )


def create_mock_browser(
    mock_result: Optional[str] = "Mock extracted content",
    mock_urls: Optional[list[str]] = None,
    mock_errors: Optional[list[str]] = None,
    mock_success: bool = True,
    config: Optional[Any] = None,
) -> "SovereignBrowser":
    """
    Create a SovereignBrowser with all external dependencies mocked.

    The returned browser:
    - Will NOT launch a real browser
    - Will return the mock_result when search_and_extract() is called
    - Records no actual video (but reports video_path=None)

    Args:
        mock_result: Text to return as the extraction result.
        mock_urls: URLs to report as visited.
        mock_errors: Errors to include in results.
        mock_success: Whether the task should report success.
        config: Optional BrowserConfig (creates one with record_video=False if None).

    Returns:
        A SovereignBrowser instance ready for testing.
    """
    from core.browser.browser_tool import SovereignBrowser
    from core.browser.config import BrowserConfig

    test_config = config or BrowserConfig(record_video=False)

    browser = SovereignBrowser(
        config=test_config,
        llm=MagicMock(),  # Prevents creating real ChatAnthropic
    )

    # Mock the session so start() doesn't launch a real browser
    mock_session = MockBrowserSession()
    browser._session = mock_session
    browser._started = True

    # Store mock data for the _patch_agent method
    browser._mock_history = create_mock_history(
        result=mock_result,
        success=mock_success,
        urls=mock_urls or ["https://example.com"],
        errors=mock_errors,
    )

    return browser
