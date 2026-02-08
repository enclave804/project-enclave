"""
Tests for browser integration module (Phase 1C).

Validates:
- BrowserConfig Pydantic model and defaults
- BrowserConfig.to_session_kwargs() conversion
- BrowserConfig.from_env() environment loading
- SovereignBrowser lifecycle (start/close/context manager)
- SovereignBrowser error handling
- SovereignBrowser result building
- MockBrowserSession and MockAgentHistory interfaces
- Video recording directory creation ("Sovereign CCTV")
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.browser.config import (
    BrowserConfig,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_DOWNLOADS_DIR,
    DEFAULT_USER_AGENT,
)
from core.browser.browser_tool import SovereignBrowser, BrowserSessionError


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── BrowserConfig Tests ─────────────────────────────────────────────


class TestBrowserConfig:
    """Tests for the BrowserConfig Pydantic model."""

    def test_default_values(self):
        """Config has sensible defaults for headless scraping."""
        config = BrowserConfig()
        assert config.headless is True
        assert config.record_video is True
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.video_framerate == 30
        assert config.max_steps == 50
        assert config.max_failures == 3
        assert config.use_vision is True
        assert config.wait_page_load == 0.5
        assert config.wait_network_idle == 1.0
        assert config.wait_between_actions == 0.5

    def test_custom_values(self):
        """Custom values override defaults."""
        config = BrowserConfig(
            headless=False,
            record_video=False,
            viewport_width=1920,
            viewport_height=1080,
            max_steps=100,
            use_vision=False,
        )
        assert config.headless is False
        assert config.record_video is False
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.max_steps == 100
        assert config.use_vision is False

    def test_default_user_agent(self):
        """Default User-Agent looks like a real browser."""
        config = BrowserConfig()
        assert "Chrome" in config.user_agent
        assert "Mozilla" in config.user_agent

    def test_custom_user_agent(self):
        """Custom User-Agent is preserved."""
        config = BrowserConfig(user_agent="MyBot/1.0")
        assert config.user_agent == "MyBot/1.0"

    def test_default_recordings_dir(self):
        """Default recordings directory is under storage/."""
        config = BrowserConfig()
        assert "storage" in str(config.recordings_dir)
        assert "recordings" in str(config.recordings_dir)

    def test_custom_recordings_dir(self):
        """Custom recordings directory is preserved."""
        config = BrowserConfig(recordings_dir=Path("/tmp/my-recordings"))
        assert config.recordings_dir == Path("/tmp/my-recordings")

    def test_video_framerate_bounds(self):
        """Video framerate must be between 1 and 60."""
        with pytest.raises(Exception):
            BrowserConfig(video_framerate=0)
        with pytest.raises(Exception):
            BrowserConfig(video_framerate=61)

    def test_viewport_width_bounds(self):
        """Viewport width must be >= 320."""
        with pytest.raises(Exception):
            BrowserConfig(viewport_width=100)

    def test_allowed_domains_default_empty(self):
        """No domain restrictions by default."""
        config = BrowserConfig()
        assert config.allowed_domains == []
        assert config.prohibited_domains == []

    def test_domain_restrictions(self):
        """Domain restrictions are preserved."""
        config = BrowserConfig(
            allowed_domains=["google.com", "*.wikipedia.org"],
            prohibited_domains=["malware.com"],
        )
        assert len(config.allowed_domains) == 2
        assert "malware.com" in config.prohibited_domains


class TestBrowserConfigViewport:
    """Tests for viewport and video size helpers."""

    def test_get_viewport(self):
        """get_viewport() returns dict with width and height."""
        config = BrowserConfig(viewport_width=1920, viewport_height=1080)
        vp = config.get_viewport()
        assert vp == {"width": 1920, "height": 1080}

    def test_get_video_size_defaults_to_viewport(self):
        """Video size defaults to viewport dimensions."""
        config = BrowserConfig(viewport_width=1280, viewport_height=720)
        vs = config.get_video_size()
        assert vs == {"width": 1280, "height": 720}

    def test_get_video_size_custom(self):
        """Custom video size overrides viewport."""
        config = BrowserConfig(
            viewport_width=1920,
            viewport_height=1080,
            video_width=1280,
            video_height=720,
        )
        vs = config.get_video_size()
        assert vs == {"width": 1280, "height": 720}


class TestBrowserConfigSessionKwargs:
    """Tests for to_session_kwargs() conversion."""

    def test_basic_kwargs(self):
        """Basic kwargs include headless, viewport, user_agent."""
        config = BrowserConfig(record_video=False)
        kwargs = config.to_session_kwargs()

        assert kwargs["headless"] is True
        assert kwargs["viewport"] == {"width": 1280, "height": 720}
        assert "Chrome" in kwargs["user_agent"]
        assert kwargs["accept_downloads"] is True
        assert kwargs["keep_alive"] is False

    def test_video_recording_kwargs(self):
        """With record_video=True, video params are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            video_dir = Path(tmpdir) / "test-session"
            kwargs = config.to_session_kwargs(session_video_dir=video_dir)

            assert kwargs["record_video_dir"] == str(video_dir)
            assert kwargs["record_video_size"] == {"width": 1280, "height": 720}
            assert kwargs["record_video_framerate"] == 30

    def test_no_video_kwargs_when_disabled(self):
        """With record_video=False, no video params in kwargs."""
        config = BrowserConfig(record_video=False)
        kwargs = config.to_session_kwargs()

        assert "record_video_dir" not in kwargs
        assert "record_video_size" not in kwargs
        assert "record_video_framerate" not in kwargs

    def test_domain_restrictions_in_kwargs(self):
        """Domain restrictions are passed through."""
        config = BrowserConfig(
            record_video=False,
            allowed_domains=["google.com"],
            prohibited_domains=["malware.com"],
        )
        kwargs = config.to_session_kwargs()

        assert kwargs["allowed_domains"] == ["google.com"]
        assert kwargs["prohibited_domains"] == ["malware.com"]

    def test_no_domain_kwargs_when_empty(self):
        """Empty domain lists are not included in kwargs."""
        config = BrowserConfig(record_video=False)
        kwargs = config.to_session_kwargs()

        assert "allowed_domains" not in kwargs
        assert "prohibited_domains" not in kwargs

    def test_timing_kwargs(self):
        """Timing parameters are converted correctly."""
        config = BrowserConfig(
            record_video=False,
            wait_page_load=1.0,
            wait_network_idle=2.0,
            wait_between_actions=0.75,
        )
        kwargs = config.to_session_kwargs()

        assert kwargs["minimum_wait_page_load_time"] == 1.0
        assert kwargs["wait_for_network_idle_page_load_time"] == 2.0
        assert kwargs["wait_between_actions"] == 0.75


class TestBrowserConfigFromEnv:
    """Tests for from_env() class method."""

    def test_default_env(self):
        """from_env() uses defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove browser env vars if present
            for key in ["BROWSER_HEADLESS", "BROWSER_RECORD_VIDEO", "BROWSER_RECORDINGS_DIR"]:
                os.environ.pop(key, None)

            config = BrowserConfig.from_env()
            assert config.headless is True
            assert config.record_video is True

    def test_headless_false_from_env(self):
        """BROWSER_HEADLESS=false disables headless."""
        with patch.dict(os.environ, {"BROWSER_HEADLESS": "false"}):
            config = BrowserConfig.from_env()
            assert config.headless is False

    def test_record_video_false_from_env(self):
        """BROWSER_RECORD_VIDEO=false disables recording."""
        with patch.dict(os.environ, {"BROWSER_RECORD_VIDEO": "false"}):
            config = BrowserConfig.from_env()
            assert config.record_video is False

    def test_custom_recordings_dir_from_env(self):
        """BROWSER_RECORDINGS_DIR sets the recordings path."""
        with patch.dict(os.environ, {"BROWSER_RECORDINGS_DIR": "/tmp/custom-recordings"}):
            config = BrowserConfig.from_env()
            assert config.recordings_dir == Path("/tmp/custom-recordings")


# ─── SovereignBrowser Tests ──────────────────────────────────────────


class TestSovereignBrowserInit:
    """Tests for SovereignBrowser initialization."""

    def test_default_config(self):
        """Default SovereignBrowser uses default BrowserConfig."""
        browser = SovereignBrowser()
        assert browser.config.headless is True
        assert browser.config.record_video is True
        assert browser.session_id.startswith("bs-")

    def test_custom_config(self):
        """Custom config is preserved."""
        config = BrowserConfig(headless=False, record_video=False)
        browser = SovereignBrowser(config=config)
        assert browser.config.headless is False
        assert browser.config.record_video is False

    def test_custom_session_id(self):
        """Custom session ID is preserved."""
        browser = SovereignBrowser(session_id="test-session-123")
        assert browser.session_id == "test-session-123"

    def test_auto_session_id(self):
        """Auto-generated session IDs are unique."""
        b1 = SovereignBrowser()
        b2 = SovereignBrowser()
        assert b1.session_id != b2.session_id

    def test_video_dir_created_when_recording(self):
        """Video directory is created for session when recording enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            browser = SovereignBrowser(config=config, session_id="test-vid")
            assert browser.video_dir is not None
            assert browser.video_dir.exists()
            assert "test-vid" in str(browser.video_dir)

    def test_no_video_dir_when_disabled(self):
        """No video directory when recording is disabled."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)
        assert browser.video_dir is None

    def test_initial_state(self):
        """Browser starts in idle state."""
        browser = SovereignBrowser()
        assert browser.is_started is False
        assert browser.is_closed is False
        assert browser.session is None


class TestSovereignBrowserLifecycle:
    """Tests for browser lifecycle (start/close)."""

    def test_start_requires_browser_use(self):
        """start() raises BrowserSessionError if browser-use not installed."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        with patch.dict("sys.modules", {"browser_use": None}):
            with pytest.raises(BrowserSessionError, match="browser-use"):
                _run(browser.start())

    def test_start_sets_started_flag(self):
        """After successful start, is_started is True."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        # Mock the BrowserSession import
        mock_session_class = MagicMock()
        mock_module = MagicMock()
        mock_module.BrowserSession = mock_session_class

        with patch.dict("sys.modules", {"browser_use": mock_module}):
            _run(browser.start())

        assert browser.is_started is True
        assert browser.session is not None

    def test_double_start_is_safe(self):
        """Calling start() twice doesn't error."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        mock_session_class = MagicMock()
        mock_module = MagicMock()
        mock_module.BrowserSession = mock_session_class

        with patch.dict("sys.modules", {"browser_use": mock_module}):
            _run(browser.start())
            _run(browser.start())  # Should log warning but not error

        assert browser.is_started is True

    def test_close_stops_session(self):
        """close() calls stop() on the session."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        mock_session = AsyncMock()
        browser._session = mock_session
        browser._started = True

        _run(browser.close())

        mock_session.stop.assert_called_once()
        assert browser.is_closed is True
        assert browser.is_started is False

    def test_double_close_is_safe(self):
        """Calling close() twice doesn't error."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        mock_session = AsyncMock()
        browser._session = mock_session
        browser._started = True

        _run(browser.close())
        _run(browser.close())  # Should be safe

        # stop() should only be called once
        mock_session.stop.assert_called_once()

    def test_close_handles_session_error(self):
        """close() handles errors from session.stop() gracefully."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        mock_session = AsyncMock()
        mock_session.stop.side_effect = RuntimeError("Browser crashed")
        browser._session = mock_session
        browser._started = True

        # Should not raise
        _run(browser.close())
        assert browser.is_closed is True

    def test_restart_after_close_raises(self):
        """Starting a closed session raises BrowserSessionError."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)
        browser._closed = True

        with pytest.raises(BrowserSessionError, match="closed"):
            _run(browser.start())


class TestSovereignBrowserContextManager:
    """Tests for async context manager protocol."""

    def test_context_manager_starts_and_closes(self):
        """Context manager calls start() and close()."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        mock_session_class = MagicMock()
        mock_module = MagicMock()
        mock_module.BrowserSession = mock_session_class

        async def run():
            with patch.dict("sys.modules", {"browser_use": mock_module}):
                async with browser:
                    assert browser.is_started is True
            assert browser.is_closed is True

        _run(run())

    def test_context_manager_closes_on_exception(self):
        """Context manager closes browser even if body raises."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        mock_session = AsyncMock()
        browser._session = mock_session
        browser._started = True

        async def run():
            with pytest.raises(ValueError, match="test error"):
                async with browser:
                    raise ValueError("test error")

        _run(run())
        assert browser.is_closed is True
        mock_session.stop.assert_called_once()


class TestSovereignBrowserOperations:
    """Tests for search_and_extract and scrape_page."""

    def test_search_and_extract_requires_started_session(self):
        """search_and_extract raises if session not started."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        with pytest.raises(BrowserSessionError, match="not active"):
            _run(browser.search_and_extract(
                url="https://example.com",
                instructions="Extract heading",
            ))

    def test_scrape_page_requires_started_session(self):
        """scrape_page raises if session not started."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        with pytest.raises(BrowserSessionError, match="not active"):
            _run(browser.scrape_page(
                url="https://example.com",
                instructions="Extract heading",
            ))

    def test_search_and_extract_returns_result_dict(self):
        """search_and_extract returns a standardized result dict."""
        from core.testing.mock_browser import MockAgentHistory

        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)
        browser._started = True

        mock_session = AsyncMock()
        browser._session = mock_session
        browser._llm = MagicMock()

        mock_history = MockAgentHistory(
            _final_result="Example Domain",
            _is_done=True,
            _is_successful=True,
            _has_errors=False,
            _urls=["https://example.com"],
            _steps=3,
        )

        # Mock the Agent class
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = mock_history

        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        mock_module = MagicMock()
        mock_module.Agent = mock_agent_class

        with patch.dict("sys.modules", {"browser_use": mock_module}):
            result = _run(browser.search_and_extract(
                url="https://example.com",
                instructions="Extract the main heading",
                max_steps=10,
            ))

        assert result["result"] == "Example Domain"
        assert result["success"] is True
        assert result["steps"] == 3
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0
        assert "https://example.com" in result["urls_visited"]
        assert result["errors"] == []
        assert result["session_id"] == browser.session_id

    def test_search_and_extract_handles_agent_error(self):
        """search_and_extract returns error result on exception."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)
        browser._started = True

        mock_session = AsyncMock()
        browser._session = mock_session
        browser._llm = MagicMock()

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.side_effect = RuntimeError("Agent crashed")

        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        mock_module = MagicMock()
        mock_module.Agent = mock_agent_class

        with patch.dict("sys.modules", {"browser_use": mock_module}):
            result = _run(browser.search_and_extract(
                url="https://example.com",
                instructions="Extract heading",
            ))

        assert result["success"] is False
        assert result["result"] is None
        assert "Agent crashed" in result["errors"]

    def test_search_and_extract_with_errors_in_history(self):
        """search_and_extract reports errors from agent history."""
        from core.testing.mock_browser import MockAgentHistory

        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)
        browser._started = True
        browser._session = AsyncMock()
        browser._llm = MagicMock()

        mock_history = MockAgentHistory(
            _final_result="Partial result",
            _is_done=True,
            _is_successful=False,
            _has_errors=True,
            _errors=["Element not found", "Click failed"],
            _urls=["https://example.com"],
            _steps=5,
        )

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = mock_history
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        mock_module = MagicMock()
        mock_module.Agent = mock_agent_class

        with patch.dict("sys.modules", {"browser_use": mock_module}):
            result = _run(browser.search_and_extract(
                url="https://example.com",
                instructions="Extract data",
            ))

        assert result["success"] is False  # has_errors() = True
        assert len(result["errors"]) == 2
        assert "Element not found" in result["errors"]

    def test_scrape_page_uses_fewer_steps(self):
        """scrape_page defaults to min(20, config.max_steps)."""
        from core.testing.mock_browser import MockAgentHistory

        config = BrowserConfig(record_video=False, max_steps=50)
        browser = SovereignBrowser(config=config)
        browser._started = True
        browser._session = AsyncMock()
        browser._llm = MagicMock()

        mock_history = MockAgentHistory(_final_result="Scraped content")
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = mock_history
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        mock_module = MagicMock()
        mock_module.Agent = mock_agent_class

        with patch.dict("sys.modules", {"browser_use": mock_module}):
            _run(browser.scrape_page(
                url="https://example.com",
                instructions="Get heading",
            ))

        # Check that agent.run was called with max_steps=20 (not 50)
        mock_agent_instance.run.assert_called_once_with(max_steps=20)


class TestSovereignBrowserVideoRecording:
    """Tests for Sovereign CCTV video recording infrastructure."""

    def test_video_dir_created_per_session(self):
        """Each session gets its own video directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )

            b1 = SovereignBrowser(config=config, session_id="session-1")
            b2 = SovereignBrowser(config=config, session_id="session-2")

            assert b1.video_dir != b2.video_dir
            assert b1.video_dir.exists()
            assert b2.video_dir.exists()
            assert b1.video_dir.name == "session-1"
            assert b2.video_dir.name == "session-2"

    def test_video_dir_under_recordings(self):
        """Video directory is under the configured recordings dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            browser = SovereignBrowser(config=config, session_id="my-session")
            assert str(browser.video_dir).startswith(tmpdir)

    def test_get_video_path_finds_webm(self):
        """_get_video_path() finds .webm files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            browser = SovereignBrowser(config=config, session_id="vid-test")

            # Create a fake video file
            fake_video = browser.video_dir / "recording.webm"
            fake_video.touch()

            path = browser._get_video_path()
            assert path is not None
            assert path.endswith(".webm")

    def test_get_video_path_finds_mp4(self):
        """_get_video_path() finds .mp4 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            browser = SovereignBrowser(config=config, session_id="vid-mp4")

            fake_video = browser.video_dir / "recording.mp4"
            fake_video.touch()

            path = browser._get_video_path()
            assert path is not None
            assert path.endswith(".mp4")

    def test_get_video_path_returns_none_when_empty(self):
        """_get_video_path() returns None when no video files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            browser = SovereignBrowser(config=config, session_id="vid-empty")

            path = browser._get_video_path()
            assert path is None

    def test_get_video_path_returns_none_when_no_video_dir(self):
        """_get_video_path() returns None when recording is disabled."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config)

        path = browser._get_video_path()
        assert path is None

    def test_session_kwargs_include_video_dir(self):
        """to_session_kwargs() passes video directory for the session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BrowserConfig(
                record_video=True,
                recordings_dir=Path(tmpdir),
            )
            browser = SovereignBrowser(config=config, session_id="kwargs-test")

            kwargs = config.to_session_kwargs(
                session_video_dir=browser.video_dir,
            )

            assert kwargs["record_video_dir"] == str(browser.video_dir)


class TestSovereignBrowserRepr:
    """Tests for string representation."""

    def test_repr_idle(self):
        """Idle browser shows status='idle'."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config, session_id="repr-test")
        r = repr(browser)
        assert "repr-test" in r
        assert "idle" in r
        assert "headless=True" in r

    def test_repr_active(self):
        """Active browser shows status='active'."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config, session_id="repr-active")
        browser._started = True
        r = repr(browser)
        assert "active" in r

    def test_repr_closed(self):
        """Closed browser shows status='closed'."""
        config = BrowserConfig(record_video=False)
        browser = SovereignBrowser(config=config, session_id="repr-closed")
        browser._closed = True
        r = repr(browser)
        assert "closed" in r


# ─── Mock Browser Tests ──────────────────────────────────────────────


class TestMockBrowserSession:
    """Tests for the MockBrowserSession."""

    def test_mock_session_interface(self):
        """MockBrowserSession has the required interface."""
        from core.testing.mock_browser import MockBrowserSession

        session = MockBrowserSession()
        _run(session.start())
        assert session._started is True

        _run(session.stop())
        assert session._stopped is True

    def test_mock_session_kill(self):
        """MockBrowserSession.kill() sets stopped flag."""
        from core.testing.mock_browser import MockBrowserSession

        session = MockBrowserSession()
        _run(session.kill())
        assert session._stopped is True


class TestMockAgentHistory:
    """Tests for the MockAgentHistory."""

    def test_default_success(self):
        """Default MockAgentHistory reports success."""
        from core.testing.mock_browser import MockAgentHistory

        history = MockAgentHistory()
        assert history.is_done() is True
        assert history.is_successful() is True
        assert history.has_errors() is False

    def test_custom_result(self):
        """Custom result is returned by final_result()."""
        from core.testing.mock_browser import MockAgentHistory

        history = MockAgentHistory(_final_result="Custom result")
        assert history.final_result() == "Custom result"

    def test_error_history(self):
        """Error history reports errors correctly."""
        from core.testing.mock_browser import MockAgentHistory

        history = MockAgentHistory(
            _has_errors=True,
            _errors=["Error 1", "Error 2"],
            _is_successful=False,
        )
        assert history.has_errors() is True
        assert len(history.errors()) == 2

    def test_urls_tracking(self):
        """URLs are tracked correctly."""
        from core.testing.mock_browser import MockAgentHistory

        history = MockAgentHistory(
            _urls=["https://google.com", "https://example.com"],
        )
        assert len(history.urls()) == 2

    def test_steps_count(self):
        """Step count is reported correctly."""
        from core.testing.mock_browser import MockAgentHistory

        history = MockAgentHistory(_steps=7)
        assert history.number_of_steps() == 7


class TestCreateMockHistory:
    """Tests for the create_mock_history factory."""

    def test_default_mock(self):
        """Default mock history is successful."""
        from core.testing.mock_browser import create_mock_history

        h = create_mock_history()
        assert h.is_done() is True
        assert h.final_result() == "Mock extracted content"

    def test_failure_mock(self):
        """Failure mock reports errors."""
        from core.testing.mock_browser import create_mock_history

        h = create_mock_history(
            success=False,
            errors=["Navigation failed"],
        )
        assert h.is_done() is False
        assert h.has_errors() is True


class TestCreateMockBrowser:
    """Tests for the create_mock_browser factory."""

    def test_creates_started_browser(self):
        """create_mock_browser returns an active browser."""
        from core.testing.mock_browser import create_mock_browser

        browser = create_mock_browser()
        assert browser.is_started is True
        assert browser.config.record_video is False

    def test_custom_config(self):
        """Custom config is passed through."""
        from core.testing.mock_browser import create_mock_browser

        config = BrowserConfig(
            record_video=False,
            headless=False,
            max_steps=10,
        )
        browser = create_mock_browser(config=config)
        assert browser.config.headless is False
        assert browser.config.max_steps == 10

    def test_mock_browser_has_mock_history(self):
        """Mock browser stores mock history for testing."""
        from core.testing.mock_browser import create_mock_browser

        browser = create_mock_browser(mock_result="Test result")
        assert hasattr(browser, "_mock_history")
        assert browser._mock_history.final_result() == "Test result"
