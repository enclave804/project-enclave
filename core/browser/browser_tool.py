"""
SovereignBrowser — Browser automation wrapper for the Sovereign Venture Engine.

Wraps the browser-use library to provide a high-level interface for
AI-driven web scraping, keyword research, SERP analysis, and competitor
analysis. Includes "Sovereign CCTV" session recording for debugging.

Used by:
- SEO Content Agent (keyword research, SERP analysis, competitor scraping)
- Future agents requiring web interaction

Architecture:
    SovereignBrowser
    +-- BrowserSession (browser-use/Playwright)
    +-- Agent (browser-use AI agent)
    +-- BrowserConfig (Pydantic config)
    +-- Session recordings → storage/recordings/{session_id}/

Usage:
    from core.browser.browser_tool import SovereignBrowser

    async with SovereignBrowser() as browser:
        result = await browser.search_and_extract(
            url="https://google.com",
            instructions="Search for 'cybersecurity assessment' and extract the top 5 organic results",
        )

    # Or manual lifecycle:
    browser = SovereignBrowser()
    await browser.start()
    result = await browser.scrape_page("https://example.com", "Extract the main heading")
    await browser.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from core.browser.config import BrowserConfig

if TYPE_CHECKING:
    from browser_use import BrowserSession, Agent
    from browser_use.agent.views import AgentHistoryList

logger = logging.getLogger(__name__)


class BrowserSessionError(Exception):
    """Raised when a browser session fails to start or encounters a fatal error."""
    pass


class SovereignBrowser:
    """
    High-level browser automation tool with session recording.

    Wraps browser-use's BrowserSession and Agent to provide:
    - Automatic session lifecycle management (start/close)
    - Video recording of every session ("Sovereign CCTV")
    - Structured logging of all browser operations
    - Clean error handling with guaranteed resource cleanup
    - Async context manager support

    Args:
        config: BrowserConfig instance. If None, uses defaults.
        llm: LangChain BaseChatModel for the browser-use Agent.
            If None, creates a default ChatAnthropic instance.
        session_id: Unique session identifier. Auto-generated if None.
    """

    def __init__(
        self,
        config: Optional[BrowserConfig] = None,
        llm: Any = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config or BrowserConfig()
        self.session_id = session_id or f"bs-{uuid.uuid4().hex[:12]}"
        self._llm = llm
        self._session: Optional[BrowserSession] = None
        self._started = False
        self._closed = False

        # Session video directory
        self._video_dir: Optional[Path] = None
        if self.config.record_video:
            self._video_dir = self.config.recordings_dir / self.session_id
            self._video_dir.mkdir(parents=True, exist_ok=True)

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Initialize the browser session.

        Creates a BrowserSession with the configured settings.
        Must be called before any browser operations (or use as context manager).

        Raises:
            BrowserSessionError: If the session fails to start.
        """
        if self._started:
            logger.warning(
                "browser_session_already_started",
                extra={"session_id": self.session_id},
            )
            return

        if self._closed:
            raise BrowserSessionError(
                f"Session {self.session_id} has been closed and cannot be restarted"
            )

        try:
            from browser_use import BrowserSession as BUSession

            session_kwargs = self.config.to_session_kwargs(
                session_video_dir=self._video_dir,
            )

            self._session = BUSession(**session_kwargs)
            self._started = True

            logger.info(
                "browser_session_started",
                extra={
                    "session_id": self.session_id,
                    "headless": self.config.headless,
                    "record_video": self.config.record_video,
                    "video_dir": str(self._video_dir) if self._video_dir else None,
                },
            )

        except ImportError as e:
            raise BrowserSessionError(
                "browser-use is not installed. "
                "Run: pip install browser-use && playwright install chromium"
            ) from e
        except Exception as e:
            self._started = False
            logger.error(
                "browser_session_start_failed",
                extra={
                    "session_id": self.session_id,
                    "error": str(e),
                },
            )
            raise BrowserSessionError(f"Failed to start browser session: {e}") from e

    async def close(self) -> None:
        """
        Close the browser session and ensure videos are saved.

        Always call this when done (or use as context manager).
        Safe to call multiple times.
        """
        if self._closed:
            return

        self._closed = True
        video_path = None

        if self._session is not None:
            try:
                await self._session.stop()

                # Check for recorded video
                if self._video_dir and self._video_dir.exists():
                    videos = list(self._video_dir.glob("*.webm")) + \
                             list(self._video_dir.glob("*.mp4"))
                    if videos:
                        video_path = str(videos[0])

            except Exception as e:
                logger.warning(
                    "browser_session_close_warning",
                    extra={
                        "session_id": self.session_id,
                        "error": str(e),
                    },
                )
            finally:
                self._session = None
                self._started = False

        logger.info(
            "browser_session_closed",
            extra={
                "session_id": self.session_id,
                "video_path": video_path,
            },
        )

    async def __aenter__(self) -> "SovereignBrowser":
        """Async context manager entry — starts the browser."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit — always closes the browser."""
        await self.close()

    # ─── Core Operations ─────────────────────────────────────────────

    async def search_and_extract(
        self,
        url: str,
        instructions: str,
        *,
        max_steps: Optional[int] = None,
        output_model: Optional[type] = None,
    ) -> dict[str, Any]:
        """
        Navigate to a URL and execute AI-driven extraction.

        Uses browser-use's Agent to perform complex multi-step browser
        interactions guided by natural language instructions.

        Args:
            url: Starting URL to navigate to.
            instructions: Natural language instructions for the AI agent.
                Example: "Extract the top 5 organic search results including
                title, URL, and meta description."
            max_steps: Maximum agent steps (default: config.max_steps).
            output_model: Optional Pydantic model for structured output.

        Returns:
            Dict with keys:
                - result: Extracted text/data (str or None)
                - success: Whether the task completed successfully
                - steps: Number of steps taken
                - duration_ms: Execution time in milliseconds
                - urls_visited: List of URLs the agent visited
                - errors: List of any errors encountered
                - video_path: Path to session recording (if enabled)
                - session_id: Session identifier

        Raises:
            BrowserSessionError: If the browser session is not started.
        """
        self._ensure_started()
        start_time = time.monotonic()
        max_steps = max_steps or self.config.max_steps

        logger.info(
            "browser_task_started",
            extra={
                "session_id": self.session_id,
                "url": url,
                "instructions": instructions[:200],
                "max_steps": max_steps,
            },
        )

        try:
            from browser_use import Agent

            llm = self._get_llm()

            # Build the full task: navigate + execute instructions
            task = f"Go to {url} and then: {instructions}"

            agent_kwargs: dict[str, Any] = {
                "task": task,
                "llm": llm,
                "browser": self._session,
                "max_actions_per_step": self.config.max_actions_per_step,
                "max_failures": self.config.max_failures,
                "use_vision": self.config.use_vision,
                "step_timeout": self.config.step_timeout,
            }

            if output_model is not None:
                agent_kwargs["output_model_schema"] = output_model

            agent = Agent(**agent_kwargs)

            history: AgentHistoryList = await asyncio.wait_for(
                agent.run(max_steps=max_steps),
                timeout=self.config.step_timeout * max_steps,
            )

            duration_ms = int((time.monotonic() - start_time) * 1000)

            result = self._build_result(history, duration_ms)

            logger.info(
                "browser_task_completed",
                extra={
                    "session_id": self.session_id,
                    "success": result["success"],
                    "steps": result["steps"],
                    "duration_ms": duration_ms,
                    "urls_visited_count": len(result["urls_visited"]),
                },
            )

            return result

        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "browser_task_timeout",
                extra={
                    "session_id": self.session_id,
                    "duration_ms": duration_ms,
                    "max_steps": max_steps,
                },
            )
            return {
                "result": None,
                "success": False,
                "steps": 0,
                "duration_ms": duration_ms,
                "urls_visited": [],
                "errors": ["Task timed out"],
                "video_path": self._get_video_path(),
                "session_id": self.session_id,
            }

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "browser_task_failed",
                extra={
                    "session_id": self.session_id,
                    "error": str(e),
                    "duration_ms": duration_ms,
                },
            )
            return {
                "result": None,
                "success": False,
                "steps": 0,
                "duration_ms": duration_ms,
                "urls_visited": [],
                "errors": [str(e)],
                "video_path": self._get_video_path(),
                "session_id": self.session_id,
            }

    async def scrape_page(
        self,
        url: str,
        instructions: str,
        *,
        max_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Simple page scraping — navigates to URL and extracts data.

        Convenience wrapper around search_and_extract for single-page scraping.
        Uses fewer max_steps since scraping is typically simpler than multi-page tasks.

        Args:
            url: URL to scrape.
            instructions: What to extract from the page.
            max_steps: Max steps (default: min(20, config.max_steps)).

        Returns:
            Same dict format as search_and_extract().
        """
        max_steps = max_steps or min(20, self.config.max_steps)
        return await self.search_and_extract(
            url=url,
            instructions=instructions,
            max_steps=max_steps,
        )

    # ─── Properties ──────────────────────────────────────────────────

    @property
    def is_started(self) -> bool:
        """Whether the browser session has been started."""
        return self._started and not self._closed

    @property
    def is_closed(self) -> bool:
        """Whether the browser session has been closed."""
        return self._closed

    @property
    def video_dir(self) -> Optional[Path]:
        """Path to this session's video recording directory."""
        return self._video_dir

    @property
    def session(self) -> Optional["BrowserSession"]:
        """The underlying browser-use BrowserSession (or None)."""
        return self._session

    # ─── Private Helpers ─────────────────────────────────────────────

    def _ensure_started(self) -> None:
        """Raise if the session is not active."""
        if not self._started or self._closed:
            raise BrowserSessionError(
                f"Browser session {self.session_id} is not active. "
                "Call start() or use as async context manager."
            )

    def _get_llm(self) -> Any:
        """Get or create the LLM for the browser-use Agent."""
        if self._llm is not None:
            return self._llm

        try:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
            )
            return self._llm
        except ImportError:
            raise BrowserSessionError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            )

    def _build_result(
        self,
        history: "AgentHistoryList",
        duration_ms: int,
    ) -> dict[str, Any]:
        """Build a standardized result dict from AgentHistoryList."""
        errors = [e for e in history.errors() if e is not None] if history.has_errors() else []
        urls = [u for u in history.urls() if u is not None]

        return {
            "result": history.final_result(),
            "success": bool(history.is_done() and not history.has_errors()),
            "steps": history.number_of_steps(),
            "duration_ms": duration_ms,
            "urls_visited": urls,
            "errors": errors,
            "video_path": self._get_video_path(),
            "session_id": self.session_id,
        }

    def _get_video_path(self) -> Optional[str]:
        """Get the path to the first recorded video file, if any."""
        if not self._video_dir or not self._video_dir.exists():
            return None

        videos = list(self._video_dir.glob("*.webm")) + \
                 list(self._video_dir.glob("*.mp4"))
        return str(videos[0]) if videos else None

    def __repr__(self) -> str:
        status = "active" if self.is_started else ("closed" if self.is_closed else "idle")
        return (
            f"SovereignBrowser(session_id={self.session_id!r}, "
            f"status={status!r}, "
            f"headless={self.config.headless}, "
            f"record_video={self.config.record_video})"
        )
