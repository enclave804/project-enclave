"""
Browser configuration for the Sovereign Venture Engine.

Defines BrowserConfig (Pydantic model) that maps to browser-use's
BrowserSession / BrowserProfile parameters. Provides sensible defaults
for headless scraping with video recording ("Sovereign CCTV").

Usage:
    from core.browser.config import BrowserConfig

    config = BrowserConfig()                   # headless + recording
    config = BrowserConfig(headless=False)      # visible browser
    config = BrowserConfig(record_video=False)  # no recording
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Default Storage Paths ───────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RECORDINGS_DIR = _PROJECT_ROOT / "storage" / "recordings"
DEFAULT_DOWNLOADS_DIR = _PROJECT_ROOT / "storage" / "downloads"


# ─── Default Headers ────────────────────────────────────────────────

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class BrowserConfig(BaseModel):
    """
    Configuration for the SovereignBrowser.

    Controls headless mode, video recording, viewport, timeouts,
    and other browser session parameters.

    Attributes:
        headless: Run the browser without a visible window.
        record_video: Enable "Sovereign CCTV" session recording.
        recordings_dir: Directory for video files. Each session gets
            a subdirectory: recordings/{session_id}/
        viewport_width: Browser content area width in pixels.
        viewport_height: Browser content area height in pixels.
        video_width: Recording frame width (defaults to viewport_width).
        video_height: Recording frame height (defaults to viewport_height).
        video_framerate: Recording FPS.
        user_agent: Browser User-Agent string.
        downloads_dir: Directory for downloaded files.
        wait_page_load: Min seconds to wait after page load.
        wait_network_idle: Seconds to wait for network settlement.
        wait_between_actions: Delay between agent actions.
        step_timeout: Max seconds per agent step.
        max_actions_per_step: Max browser actions per LLM call.
        max_failures: Consecutive failures before stopping.
        max_steps: Default max steps for agent.run().
        use_vision: Send screenshots to the LLM for visual understanding.
        allowed_domains: Whitelist of domains the browser can navigate to.
        prohibited_domains: Blacklist of domains to block.
    """

    # ─── Display & Mode ─────────────────────────────────────────
    headless: bool = Field(
        True,
        description="Run browser without visible window",
    )

    # ─── Video Recording ("Sovereign CCTV") ─────────────────────
    record_video: bool = Field(
        True,
        description="Enable session video recording",
    )
    recordings_dir: Path = Field(
        default_factory=lambda: DEFAULT_RECORDINGS_DIR,
        description="Root directory for video recordings",
    )
    video_width: Optional[int] = Field(
        None,
        description="Recording frame width (defaults to viewport_width)",
    )
    video_height: Optional[int] = Field(
        None,
        description="Recording frame height (defaults to viewport_height)",
    )
    video_framerate: int = Field(
        30,
        ge=1,
        le=60,
        description="Recording frames per second",
    )

    # ─── Viewport ───────────────────────────────────────────────
    viewport_width: int = Field(1280, ge=320, description="Content area width")
    viewport_height: int = Field(720, ge=240, description="Content area height")

    # ─── Identity ───────────────────────────────────────────────
    user_agent: str = Field(
        DEFAULT_USER_AGENT,
        description="Browser User-Agent string",
    )

    # ─── Downloads ──────────────────────────────────────────────
    downloads_dir: Path = Field(
        default_factory=lambda: DEFAULT_DOWNLOADS_DIR,
        description="Directory for downloaded files",
    )

    # ─── Timing ─────────────────────────────────────────────────
    wait_page_load: float = Field(
        0.5,
        ge=0.0,
        description="Min seconds to wait after page load",
    )
    wait_network_idle: float = Field(
        1.0,
        ge=0.0,
        description="Seconds to wait for network settlement",
    )
    wait_between_actions: float = Field(
        0.5,
        ge=0.0,
        description="Delay between agent actions (seconds)",
    )
    step_timeout: int = Field(
        180,
        ge=10,
        description="Max seconds per agent step",
    )

    # ─── Agent Behavior ─────────────────────────────────────────
    max_actions_per_step: int = Field(
        5,
        ge=1,
        description="Max browser actions per LLM call",
    )
    max_failures: int = Field(
        3,
        ge=1,
        description="Consecutive failures before stopping",
    )
    max_steps: int = Field(
        50,
        ge=1,
        description="Default max steps for agent.run()",
    )
    use_vision: bool = Field(
        True,
        description="Send screenshots to LLM for visual understanding",
    )

    # ─── Stealth & Proxy ("Invisible Wall") ─────────────────────
    proxy_url: Optional[str] = Field(
        None,
        description=(
            "Proxy server URL (e.g., 'http://user:pass@proxy:8080'). "
            "Routes all browser traffic through the proxy for anonymity."
        ),
    )
    stealth_mode: bool = Field(
        True,
        description=(
            "Enable anti-bot evasion: disables automation detection flags, "
            "randomizes user agent if user_agent_rotate is True."
        ),
    )
    user_agent_rotate: bool = Field(
        True,
        description=(
            "Rotate User-Agent on each session to avoid fingerprinting. "
            "Only effective when stealth_mode is True."
        ),
    )

    # ─── Domain Restrictions ────────────────────────────────────
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Whitelist of allowed domains (empty = all allowed)",
    )
    prohibited_domains: list[str] = Field(
        default_factory=list,
        description="Blacklist of blocked domains",
    )

    def get_video_size(self) -> dict[str, int]:
        """Get video recording dimensions (defaults to viewport if not set)."""
        return {
            "width": self.video_width or self.viewport_width,
            "height": self.video_height or self.viewport_height,
        }

    def get_viewport(self) -> dict[str, int]:
        """Get viewport dimensions as dict for browser-use."""
        return {
            "width": self.viewport_width,
            "height": self.viewport_height,
        }

    def get_stealth_args(self) -> list[str]:
        """Get Chromium launch args for anti-bot evasion."""
        if not self.stealth_mode:
            return []
        return [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-infobars",
        ]

    def to_session_kwargs(self, session_video_dir: Optional[Path] = None) -> dict[str, Any]:
        """
        Convert config to keyword arguments for BrowserSession().

        Args:
            session_video_dir: Specific video directory for this session.
                If None and record_video is True, uses recordings_dir directly.

        Returns:
            Dict of kwargs to pass to BrowserSession().
        """
        kwargs: dict[str, Any] = {
            "headless": self.headless,
            "viewport": self.get_viewport(),
            "user_agent": self.user_agent,
            "minimum_wait_page_load_time": self.wait_page_load,
            "wait_for_network_idle_page_load_time": self.wait_network_idle,
            "wait_between_actions": self.wait_between_actions,
            "accept_downloads": True,
            "keep_alive": False,
        }

        # Video recording
        if self.record_video:
            video_dir = session_video_dir or self.recordings_dir
            kwargs["record_video_dir"] = str(video_dir)
            kwargs["record_video_size"] = self.get_video_size()
            kwargs["record_video_framerate"] = self.video_framerate

        # Stealth & Proxy
        if self.proxy_url:
            kwargs["proxy"] = {"server": self.proxy_url}
        stealth_args = self.get_stealth_args()
        if stealth_args:
            kwargs["extra_chromium_args"] = stealth_args

        # Domain restrictions
        if self.allowed_domains:
            kwargs["allowed_domains"] = self.allowed_domains
        if self.prohibited_domains:
            kwargs["prohibited_domains"] = self.prohibited_domains

        return kwargs

    @classmethod
    def from_env(cls) -> "BrowserConfig":
        """
        Create config from environment variables.

        Reads:
            BROWSER_HEADLESS: "true"/"false" (default: "true")
            BROWSER_RECORD_VIDEO: "true"/"false" (default: "true")
            BROWSER_RECORDINGS_DIR: path (default: storage/recordings)
            BROWSER_PROXY_URL: proxy URL (default: None)
            BROWSER_STEALTH_MODE: "true"/"false" (default: "true")
        """
        kwargs: dict[str, Any] = {
            "headless": os.environ.get("BROWSER_HEADLESS", "true").lower() == "true",
            "record_video": os.environ.get("BROWSER_RECORD_VIDEO", "true").lower() == "true",
            "recordings_dir": Path(
                os.environ.get("BROWSER_RECORDINGS_DIR", str(DEFAULT_RECORDINGS_DIR))
            ),
            "stealth_mode": os.environ.get("BROWSER_STEALTH_MODE", "true").lower() == "true",
        }
        proxy = os.environ.get("BROWSER_PROXY_URL")
        if proxy:
            kwargs["proxy_url"] = proxy
        return cls(**kwargs)
