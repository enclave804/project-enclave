"""
Tests for Feature Freeze upgrades:
- Kill Switch: AgentModelConfig provider/base_url/context_window
- Invisible Wall: BrowserConfig proxy_url/stealth_mode

These are the final architectural config changes before Phase 1D execution.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from core.config.agent_schema import AgentModelConfig, AgentInstanceConfig
from core.browser.config import BrowserConfig


# ─── Kill Switch: AgentModelConfig Provider Support ──────────────────

class TestAgentModelConfigProvider:
    """Tests for the Kill Switch — LLM provider configuration."""

    def test_default_provider_is_anthropic(self):
        """Default provider should be anthropic."""
        config = AgentModelConfig()
        assert config.provider == "anthropic"

    def test_provider_openai(self):
        """Should accept openai provider."""
        config = AgentModelConfig(provider="openai", model="gpt-4o")
        assert config.provider == "openai"
        assert config.model == "gpt-4o"

    def test_provider_ollama(self):
        """Should accept ollama provider with base_url."""
        config = AgentModelConfig(
            provider="ollama",
            model="llama3.1:70b",
            base_url="http://localhost:11434",
        )
        assert config.provider == "ollama"
        assert config.model == "llama3.1:70b"
        assert config.base_url == "http://localhost:11434"

    def test_invalid_provider_rejected(self):
        """Should reject invalid provider names."""
        with pytest.raises(Exception):
            AgentModelConfig(provider="invalid_provider")

    def test_base_url_defaults_none(self):
        """base_url should default to None for cloud providers."""
        config = AgentModelConfig()
        assert config.base_url is None

    def test_context_window_default(self):
        """context_window should default to 128000."""
        config = AgentModelConfig()
        assert config.context_window == 128000

    def test_context_window_custom(self):
        """Should accept custom context_window."""
        config = AgentModelConfig(context_window=32000)
        assert config.context_window == 32000

    def test_context_window_bounds(self):
        """context_window should enforce min/max bounds."""
        with pytest.raises(Exception):
            AgentModelConfig(context_window=500)  # Below 1024 min

    def test_full_ollama_config_in_agent(self):
        """Kill Switch scenario: full Ollama config in agent YAML."""
        config = AgentInstanceConfig(
            agent_id="local_agent",
            agent_type="outreach",
            name="Local Agent",
            model={
                "provider": "ollama",
                "model": "llama3.1:70b",
                "base_url": "http://localhost:11434",
                "context_window": 32000,
                "temperature": 0.3,
                "max_tokens": 4096,
            },
        )
        assert config.model.provider == "ollama"
        assert config.model.base_url == "http://localhost:11434"
        assert config.model.context_window == 32000

    def test_serialization_includes_provider_fields(self):
        """model_dump should include all new fields."""
        config = AgentModelConfig(
            provider="openai",
            base_url="https://api.openai.com/v1",
            context_window=128000,
        )
        data = config.model_dump()
        assert data["provider"] == "openai"
        assert data["base_url"] == "https://api.openai.com/v1"
        assert data["context_window"] == 128000

    def test_backward_compat_no_provider_specified(self):
        """Existing configs without provider should still work (anthropic default)."""
        config = AgentModelConfig(
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            max_tokens=4096,
        )
        assert config.provider == "anthropic"
        assert config.base_url is None


# ─── Invisible Wall: BrowserConfig Stealth & Proxy ───────────────────

class TestBrowserConfigStealth:
    """Tests for anti-bot stealth and proxy configuration."""

    def test_stealth_mode_defaults_true(self):
        """stealth_mode should default to True."""
        config = BrowserConfig()
        assert config.stealth_mode is True

    def test_user_agent_rotate_defaults_true(self):
        """user_agent_rotate should default to True."""
        config = BrowserConfig()
        assert config.user_agent_rotate is True

    def test_proxy_url_defaults_none(self):
        """proxy_url should default to None."""
        config = BrowserConfig()
        assert config.proxy_url is None

    def test_proxy_url_custom(self):
        """Should accept a custom proxy URL."""
        config = BrowserConfig(proxy_url="http://user:pass@proxy.io:8080")
        assert config.proxy_url == "http://user:pass@proxy.io:8080"

    def test_stealth_mode_disabled(self):
        """Should accept stealth_mode=False."""
        config = BrowserConfig(stealth_mode=False)
        assert config.stealth_mode is False

    def test_get_stealth_args_when_enabled(self):
        """get_stealth_args should return anti-detection flags."""
        config = BrowserConfig(stealth_mode=True)
        args = config.get_stealth_args()
        assert "--disable-blink-features=AutomationControlled" in args
        assert len(args) >= 1

    def test_get_stealth_args_when_disabled(self):
        """get_stealth_args should return empty list when disabled."""
        config = BrowserConfig(stealth_mode=False)
        args = config.get_stealth_args()
        assert args == []

    def test_session_kwargs_includes_proxy(self):
        """to_session_kwargs should include proxy when configured."""
        config = BrowserConfig(proxy_url="http://proxy:8080")
        kwargs = config.to_session_kwargs()
        assert "proxy" in kwargs
        assert kwargs["proxy"]["server"] == "http://proxy:8080"

    def test_session_kwargs_no_proxy_when_none(self):
        """to_session_kwargs should NOT include proxy when None."""
        config = BrowserConfig(proxy_url=None)
        kwargs = config.to_session_kwargs()
        assert "proxy" not in kwargs

    def test_session_kwargs_includes_stealth_args(self):
        """to_session_kwargs should include stealth chromium args."""
        config = BrowserConfig(stealth_mode=True)
        kwargs = config.to_session_kwargs()
        assert "extra_chromium_args" in kwargs
        assert "--disable-blink-features=AutomationControlled" in kwargs["extra_chromium_args"]

    def test_session_kwargs_no_stealth_when_disabled(self):
        """to_session_kwargs should NOT include stealth args when disabled."""
        config = BrowserConfig(stealth_mode=False)
        kwargs = config.to_session_kwargs()
        assert "extra_chromium_args" not in kwargs

    def test_from_env_reads_proxy(self):
        """from_env should read BROWSER_PROXY_URL."""
        with patch.dict(os.environ, {
            "BROWSER_PROXY_URL": "http://proxy:9090",
        }):
            config = BrowserConfig.from_env()
            assert config.proxy_url == "http://proxy:9090"

    def test_from_env_reads_stealth(self):
        """from_env should read BROWSER_STEALTH_MODE."""
        with patch.dict(os.environ, {
            "BROWSER_STEALTH_MODE": "false",
        }):
            config = BrowserConfig.from_env()
            assert config.stealth_mode is False

    def test_from_env_stealth_defaults_true(self):
        """from_env stealth_mode should default to True when not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Need required vars
            config = BrowserConfig.from_env()
            assert config.stealth_mode is True

    def test_from_env_proxy_defaults_none(self):
        """from_env proxy_url should default to None when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = BrowserConfig.from_env()
            assert config.proxy_url is None

    def test_full_stealth_config(self):
        """Complete stealth setup: proxy + stealth + user agent rotation."""
        config = BrowserConfig(
            proxy_url="socks5://user:pass@proxy.io:1080",
            stealth_mode=True,
            user_agent_rotate=True,
            headless=True,
        )
        kwargs = config.to_session_kwargs()
        assert kwargs["proxy"]["server"] == "socks5://user:pass@proxy.io:1080"
        assert "--disable-blink-features=AutomationControlled" in kwargs["extra_chromium_args"]
        assert kwargs["headless"] is True
