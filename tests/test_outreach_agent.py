"""
Unit tests for the OutreachAgent strangler-fig adapter.

Tests registration, graph building, state preparation, and the
adapter's ability to wrap the legacy pipeline.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agents.base import BaseAgent
from core.agents.registry import AGENT_IMPLEMENTATIONS
from core.agents.implementations.outreach_agent import OutreachAgent
from core.config.agent_schema import AgentInstanceConfig
from core.exceptions import AgentConfigurationError


# ─── Registration Tests ───────────────────────────────────────────


class TestOutreachAgentRegistration:
    """Tests that the @register_agent_type("outreach") decorator works."""

    def test_registered_in_implementations(self):
        assert "outreach" in AGENT_IMPLEMENTATIONS

    def test_registered_class_is_outreach_agent(self):
        assert AGENT_IMPLEMENTATIONS["outreach"] is OutreachAgent

    def test_agent_type_set(self):
        assert OutreachAgent.agent_type == "outreach"

    def test_inherits_base_agent(self):
        assert issubclass(OutreachAgent, BaseAgent)


# ─── Construction Tests ───────────────────────────────────────────


class TestOutreachAgentConstruction:
    """Tests for OutreachAgent instantiation."""

    @pytest.fixture
    def config(self):
        return AgentInstanceConfig(
            agent_id="outreach",
            agent_type="outreach",
            name="Outreach Agent",
            vertical_id="enclave_guard",
            params={"daily_lead_limit": 25},
        )

    @pytest.fixture
    def agent(self, config):
        return OutreachAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_basic_properties(self, agent):
        assert agent.agent_id == "outreach"
        assert agent.vertical_id == "enclave_guard"
        assert agent.agent_type == "outreach"
        assert agent.name == "Outreach Agent"

    def test_config_params_accessible(self, agent):
        assert agent.config.params["daily_lead_limit"] == 25

    def test_legacy_deps_default_none(self, agent):
        assert agent._apollo is None
        assert agent._vertical_config is None

    def test_constructor_with_legacy_deps(self, config):
        mock_apollo = MagicMock()
        mock_vc = MagicMock()

        agent = OutreachAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
            apollo=mock_apollo,
            vertical_config=mock_vc,
        )
        assert agent._apollo is mock_apollo
        assert agent._vertical_config is mock_vc

    def test_repr(self, agent):
        r = repr(agent)
        assert "OutreachAgent" in r
        assert "outreach" in r
        assert "enclave_guard" in r
        assert "legacy_adapter=True" in r

    def test_get_tools_returns_empty(self, agent):
        """Legacy pipeline doesn't use tool-calling; tools are baked in."""
        assert agent.get_tools() == []

    def test_get_state_class(self, agent):
        from core.agents.state import OutreachAgentState
        assert agent.get_state_class() is OutreachAgentState


# ─── Graph Building Tests ─────────────────────────────────────────


class TestOutreachAgentGraph:
    """Tests for graph construction via the adapter."""

    @pytest.fixture
    def config(self):
        return AgentInstanceConfig(
            agent_id="outreach",
            agent_type="outreach",
            name="Outreach Agent",
            vertical_id="enclave_guard",
        )

    def test_build_graph_without_vertical_config_raises(self, config):
        """Should raise AgentConfigurationError if vertical_config is missing."""
        agent = OutreachAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )
        with pytest.raises(AgentConfigurationError, match="vertical_config"):
            agent.build_graph()

    @patch("core.graph.workflow_engine.build_pipeline_graph")
    def test_build_graph_delegates_to_legacy(self, mock_build, config):
        """Should delegate to build_pipeline_graph from workflow_engine."""
        mock_graph = MagicMock()
        mock_build.return_value = mock_graph

        mock_vc = MagicMock()
        mock_db = MagicMock()
        mock_embedder = MagicMock()
        mock_llm = MagicMock()
        mock_apollo = MagicMock()

        agent = OutreachAgent(
            config=config,
            db=mock_db,
            embedder=mock_embedder,
            anthropic_client=mock_llm,
            apollo=mock_apollo,
            vertical_config=mock_vc,
        )

        result = agent.build_graph()

        assert result is mock_graph
        mock_build.assert_called_once_with(
            config=mock_vc,
            db=mock_db,
            apollo=mock_apollo,
            embedder=mock_embedder,
            anthropic_client=mock_llm,
            checkpointer=None,
            test_mode=False,
        )

    def test_set_legacy_deps_invalidates_cache(self, config):
        """Setting legacy deps should clear the cached graph."""
        agent = OutreachAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )
        # Pretend a graph was built
        agent._graph = MagicMock()

        agent.set_legacy_deps(
            apollo=MagicMock(),
            vertical_config=MagicMock(),
        )

        # Cached graph should be cleared
        assert agent._graph is None


# ─── State Preparation Tests ──────────────────────────────────────


class TestOutreachAgentState:
    """Tests for state preparation."""

    @pytest.fixture
    def agent(self):
        config = AgentInstanceConfig(
            agent_id="outreach",
            agent_type="outreach",
            name="Outreach Agent",
            vertical_id="enclave_guard",
        )
        return OutreachAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_prepare_initial_state_from_lead(self, agent):
        """Should translate task dict to LeadState format."""
        task = {
            "lead": {
                "contact": {
                    "name": "Jane Doe",
                    "email": "jane@acme.com",
                    "title": "CTO",
                },
                "company": {
                    "name": "Acme Corp",
                    "domain": "acme.com",
                    "industry": "Technology",
                    "employee_count": 150,
                },
            }
        }

        state = agent._prepare_initial_state(task, "run-123")

        assert state["contact_name"] == "Jane Doe"
        assert state["contact_email"] == "jane@acme.com"
        assert state["company_name"] == "Acme Corp"
        assert state["company_domain"] == "acme.com"
        assert state["vertical_id"] == "enclave_guard"
        assert state["pipeline_run_id"] == "run-123"

    def test_prepare_initial_state_defaults(self, agent):
        """Should handle missing fields gracefully."""
        task = {"lead": {"contact": {}, "company": {}}}
        state = agent._prepare_initial_state(task, "run-456")

        assert state["contact_name"] == ""
        assert state["company_domain"] == ""
        assert state["is_duplicate"] is False
        assert state["qualification_score"] == 0.0


# ─── Knowledge Writing Tests ──────────────────────────────────────


class TestOutreachAgentKnowledge:
    """Tests for knowledge writing (no-op in adapter)."""

    @pytest.fixture
    def agent(self):
        config = AgentInstanceConfig(
            agent_id="outreach",
            agent_type="outreach",
            name="Outreach Agent",
            vertical_id="enclave_guard",
        )
        return OutreachAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_write_knowledge_is_noop(self, agent):
        """Legacy pipeline handles its own RAG writing."""
        # Should not raise, should not call db
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(agent.write_knowledge({"some": "result"}))
        finally:
            loop.close()
        # No calls to db.store_insight expected
        agent.db.store_insight.assert_not_called()


# ─── YAML Discovery Tests ────────────────────────────────────────


class TestOutreachAgentYAMLDiscovery:
    """Tests that the outreach.yaml is discovered by AgentRegistry."""

    def test_discover_from_real_verticals_dir(self):
        """Should find the outreach.yaml in the enclave_guard vertical."""
        from pathlib import Path
        from core.agents.registry import AgentRegistry

        verticals_dir = Path(__file__).parent.parent / "verticals"
        registry = AgentRegistry("enclave_guard", verticals_dir)
        configs = registry.discover_agents()

        agent_ids = [c.agent_id for c in configs]
        assert "outreach" in agent_ids

    def test_discovered_config_matches_yaml(self):
        """The discovered config should match what's in the YAML file."""
        from pathlib import Path
        from core.agents.registry import AgentRegistry

        verticals_dir = Path(__file__).parent.parent / "verticals"
        registry = AgentRegistry("enclave_guard", verticals_dir)
        registry.discover_agents()

        config = registry.get_config("outreach")
        assert config is not None
        assert config.agent_type == "outreach"
        assert config.name == "Outreach Agent"
        assert config.enabled is True
        assert config.browser_enabled is False
        assert "send_outreach" in config.human_gates.gate_before
        assert config.params["daily_lead_limit"] == 25
        assert config.params["duplicate_cooldown_days"] == 90
        assert config.model.model == "claude-sonnet-4-20250514"

    def test_instantiate_outreach_from_registry(self):
        """Should instantiate OutreachAgent from discovered config."""
        from pathlib import Path
        from core.agents.registry import AgentRegistry

        verticals_dir = Path(__file__).parent.parent / "verticals"
        registry = AgentRegistry("enclave_guard", verticals_dir)
        registry.discover_agents()

        agent = registry.instantiate_agent(
            "outreach",
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )
        assert isinstance(agent, OutreachAgent)
        assert agent.agent_id == "outreach"
        assert agent.vertical_id == "enclave_guard"
