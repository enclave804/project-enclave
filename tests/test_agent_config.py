"""
Unit tests for agent config schema (Pydantic models).

Validates YAML config parsing, validation, and defaults.
"""

import pytest

from core.config.agent_schema import (
    AgentInstanceConfig,
    AgentModelConfig,
    AgentToolConfig,
    HumanGateConfig,
    AgentScheduleConfig,
)


class TestAgentModelConfig:
    """Tests for AgentModelConfig defaults and validation."""

    def test_defaults(self):
        config = AgentModelConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096

    def test_custom_values(self):
        config = AgentModelConfig(model="claude-3-haiku-20240307", temperature=0.2, max_tokens=2048)
        assert config.model == "claude-3-haiku-20240307"
        assert config.temperature == 0.2

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            AgentModelConfig(temperature=3.0)
        with pytest.raises(Exception):
            AgentModelConfig(temperature=-1.0)


class TestAgentToolConfig:
    """Tests for AgentToolConfig."""

    def test_minimal(self):
        tool = AgentToolConfig(name="apollo_search", type="builtin")
        assert tool.name == "apollo_search"
        assert tool.type == "builtin"
        assert tool.mcp_server is None
        assert tool.config == {}

    def test_mcp_tool(self):
        tool = AgentToolConfig(
            name="rag_search",
            type="mcp",
            mcp_server="enclave-tools",
            config={"timeout": 30},
        )
        assert tool.type == "mcp"
        assert tool.mcp_server == "enclave-tools"
        assert tool.config["timeout"] == 30


class TestHumanGateConfig:
    """Tests for HumanGateConfig."""

    def test_defaults(self):
        gate = HumanGateConfig()
        assert gate.enabled is True
        assert gate.gate_before == []
        assert gate.auto_approve_threshold is None

    def test_configured_gates(self):
        gate = HumanGateConfig(
            gate_before=["send_outreach", "publish_content"],
            auto_approve_threshold=0.95,
        )
        assert len(gate.gate_before) == 2
        assert gate.auto_approve_threshold == 0.95


class TestAgentScheduleConfig:
    """Tests for AgentScheduleConfig."""

    def test_defaults(self):
        sched = AgentScheduleConfig()
        assert sched.trigger == "manual"
        assert sched.cron is None

    def test_cron_schedule(self):
        sched = AgentScheduleConfig(trigger="scheduled", cron="0 9 * * 1-5")
        assert sched.trigger == "scheduled"
        assert sched.cron == "0 9 * * 1-5"

    def test_event_trigger(self):
        sched = AgentScheduleConfig(trigger="event", event_source="email_reply_received")
        assert sched.trigger == "event"
        assert sched.event_source == "email_reply_received"


class TestAgentInstanceConfig:
    """Tests for the full AgentInstanceConfig model."""

    def test_minimal_valid_config(self):
        config = AgentInstanceConfig(
            agent_id="outreach",
            agent_type="outreach",
            name="Outreach Agent",
        )
        assert config.agent_id == "outreach"
        assert config.agent_type == "outreach"
        assert config.enabled is True
        assert config.browser_enabled is False
        assert config.max_consecutive_errors == 5
        assert config.rag_write_confidence_threshold == 0.7

    def test_full_config(self):
        config = AgentInstanceConfig(
            agent_id="seo_content",
            agent_type="seo_content",
            name="SEO Content Agent",
            description="Generates blog posts for cybersecurity topics",
            enabled=True,
            model=AgentModelConfig(temperature=0.7, max_tokens=8192),
            tools=[
                AgentToolConfig(name="rag_search", type="mcp", mcp_server="enclave-tools"),
                AgentToolConfig(name="browser", type="browser"),
            ],
            browser_enabled=True,
            human_gates=HumanGateConfig(gate_before=["store_content"]),
            schedule=AgentScheduleConfig(trigger="scheduled", cron="0 6 * * MON"),
            system_prompt_path="prompts/agent_prompts/seo_system.md",
            max_consecutive_errors=3,
            rag_write_confidence_threshold=0.8,
            params={"target_word_count": 1500, "content_types": ["blog_post"]},
        )
        assert config.browser_enabled is True
        assert len(config.tools) == 2
        assert config.model.temperature == 0.7
        assert config.params["target_word_count"] == 1500
        assert config.max_consecutive_errors == 3

    def test_agent_id_validation(self):
        """agent_id must be lowercase alphanumeric with underscores."""
        # Valid
        AgentInstanceConfig(agent_id="my_agent_2", agent_type="test", name="Test")
        # Invalid: starts with number
        with pytest.raises(Exception):
            AgentInstanceConfig(agent_id="2bad", agent_type="test", name="Test")
        # Invalid: uppercase
        with pytest.raises(Exception):
            AgentInstanceConfig(agent_id="MyAgent", agent_type="test", name="Test")
        # Invalid: contains dash
        with pytest.raises(Exception):
            AgentInstanceConfig(agent_id="my-agent", agent_type="test", name="Test")

    def test_vertical_id_defaults_empty(self):
        """vertical_id is injected by AgentRegistry, defaults empty."""
        config = AgentInstanceConfig(
            agent_id="test", agent_type="test", name="Test"
        )
        assert config.vertical_id == ""

    def test_confidence_threshold_bounds(self):
        """rag_write_confidence_threshold must be 0.0 - 1.0."""
        with pytest.raises(Exception):
            AgentInstanceConfig(
                agent_id="test", agent_type="test", name="Test",
                rag_write_confidence_threshold=1.5,
            )
