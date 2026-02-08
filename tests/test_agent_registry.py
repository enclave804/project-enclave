"""
Unit tests for the agent registry and @register_agent_type decorator.

Tests YAML discovery, decorator registration, and agent instantiation.
"""

import tempfile
from pathlib import Path
from typing import Any, Type
from unittest.mock import MagicMock

import pytest
import yaml

from core.agents.base import BaseAgent
from core.agents.state import BaseAgentState
from core.agents.registry import (
    AgentRegistry,
    register_agent_type,
    AGENT_IMPLEMENTATIONS,
    get_registered_types,
)
from core.config.agent_schema import AgentInstanceConfig


# ─── Test Agent Implementation ─────────────────────────────────────────

@register_agent_type("test_agent")
class TestAgentImpl(BaseAgent):
    """Minimal agent implementation for testing."""

    def build_graph(self) -> Any:
        return MagicMock()

    def get_tools(self) -> list[Any]:
        return []

    def get_state_class(self) -> Type[BaseAgentState]:
        return BaseAgentState


# ─── Decorator Tests ───────────────────────────────────────────────────

class TestRegisterAgentType:
    """Tests for the @register_agent_type decorator."""

    def test_registers_implementation(self):
        assert "test_agent" in AGENT_IMPLEMENTATIONS
        assert AGENT_IMPLEMENTATIONS["test_agent"] is TestAgentImpl

    def test_sets_agent_type_on_class(self):
        assert TestAgentImpl.agent_type == "test_agent"

    def test_get_registered_types(self):
        types = get_registered_types()
        assert "test_agent" in types
        assert types == sorted(types)  # Should be sorted

    def test_duplicate_registration_warns(self, caplog):
        """Re-registering should warn but succeed."""
        # Use a separate key to avoid polluting the global registry
        @register_agent_type("_dup_test_only")
        class FirstDupAgent(BaseAgent):
            def build_graph(self): return MagicMock()
            def get_tools(self): return []
            def get_state_class(self): return BaseAgentState

        @register_agent_type("_dup_test_only")
        class SecondDupAgent(BaseAgent):
            def build_graph(self): return MagicMock()
            def get_tools(self): return []
            def get_state_class(self): return BaseAgentState

        assert "Overwriting" in caplog.text
        # Clean up so other tests are not affected
        AGENT_IMPLEMENTATIONS.pop("_dup_test_only", None)


# ─── Registry Discovery Tests ─────────────────────────────────────────

class TestAgentRegistry:
    """Tests for AgentRegistry YAML discovery and instantiation."""

    @pytest.fixture
    def temp_verticals_dir(self, tmp_path):
        """Create a temporary verticals directory with agent YAML configs."""
        agents_dir = tmp_path / "test_vertical" / "agents"
        agents_dir.mkdir(parents=True)

        # Write a valid agent config
        config = {
            "agent_id": "my_test_agent",
            "agent_type": "test_agent",
            "name": "My Test Agent",
            "description": "A test agent for unit tests",
            "enabled": True,
            "params": {"key": "value"},
        }
        with open(agents_dir / "my_test_agent.yaml", "w") as f:
            yaml.dump(config, f)

        # Write a disabled agent config
        disabled_config = {
            "agent_id": "disabled_agent",
            "agent_type": "test_agent",
            "name": "Disabled Agent",
            "enabled": False,
        }
        with open(agents_dir / "disabled_agent.yaml", "w") as f:
            yaml.dump(disabled_config, f)

        # Write an empty file (should be skipped)
        (agents_dir / "empty.yaml").touch()

        return tmp_path

    def test_discover_agents(self, temp_verticals_dir):
        """Should discover enabled agents from YAML files."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        configs = registry.discover_agents()

        assert len(configs) == 1
        assert configs[0].agent_id == "my_test_agent"
        assert configs[0].agent_type == "test_agent"
        assert configs[0].vertical_id == "test_vertical"

    def test_disabled_agents_skipped(self, temp_verticals_dir):
        """Disabled agents should not appear in discovered configs."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        configs = registry.discover_agents()
        ids = [c.agent_id for c in configs]
        assert "disabled_agent" not in ids

    def test_empty_yaml_skipped(self, temp_verticals_dir):
        """Empty YAML files should be silently skipped."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        configs = registry.discover_agents()
        # Should not raise, and only valid configs returned
        assert len(configs) == 1

    def test_missing_agents_dir(self, tmp_path):
        """Should return empty list if agents/ dir doesn't exist."""
        registry = AgentRegistry("nonexistent", tmp_path)
        configs = registry.discover_agents()
        assert configs == []

    def test_instantiate_agent(self, temp_verticals_dir):
        """Should create a proper agent instance from config."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        registry.discover_agents()

        mock_db = MagicMock()
        mock_embedder = MagicMock()
        mock_client = MagicMock()

        agent = registry.instantiate_agent(
            "my_test_agent", mock_db, mock_embedder, mock_client
        )

        assert isinstance(agent, BaseAgent)
        assert agent.agent_type == "test_agent"
        assert agent.agent_id == "my_test_agent"
        assert agent.vertical_id == "test_vertical"
        assert agent.config.params["key"] == "value"

    def test_instantiate_unknown_agent_raises(self, temp_verticals_dir):
        """Should raise ValueError for unknown agent_id."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        registry.discover_agents()

        with pytest.raises(ValueError, match="Agent config not found"):
            registry.instantiate_agent(
                "nonexistent", MagicMock(), MagicMock(), MagicMock()
            )

    def test_instantiate_all(self, temp_verticals_dir):
        """Should instantiate all discovered agents."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        registry.discover_agents()

        agents = registry.instantiate_all(
            MagicMock(), MagicMock(), MagicMock()
        )

        assert len(agents) == 1
        assert "my_test_agent" in agents

    def test_list_agent_ids(self, temp_verticals_dir):
        """Should return sorted list of discovered agent IDs."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        registry.discover_agents()
        ids = registry.list_agent_ids()
        assert ids == ["my_test_agent"]

    def test_get_config(self, temp_verticals_dir):
        """Should return config for a specific agent."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        registry.discover_agents()
        config = registry.get_config("my_test_agent")
        assert config is not None
        assert config.name == "My Test Agent"

    def test_get_agent_before_instantiation(self, temp_verticals_dir):
        """Should return None if agent hasn't been instantiated yet."""
        registry = AgentRegistry("test_vertical", temp_verticals_dir)
        registry.discover_agents()
        assert registry.get_agent("my_test_agent") is None
