"""
Unit tests for the Neural Architecture upgrade.

Tests routing config, refinement config, and their integration
into BaseAgent's run lifecycle.
"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock

from core.config.agent_schema import (
    AgentInstanceConfig,
    AgentModelConfig,
    ModelType,
    RoutingConfig,
    RefinementConfig,
)
from core.agents.base import BaseAgent
from core.agents.state import BaseAgentState


# ─── Test Agent Implementation ────────────────────────────────────


class _StubAgent(BaseAgent):
    """Minimal agent for testing base class behavior."""

    agent_type = "_stub"

    def build_graph(self):
        return MagicMock()

    def get_tools(self):
        return []

    def get_state_class(self):
        return BaseAgentState


def _make_agent(
    routing: RoutingConfig | None = None,
    refinement: RefinementConfig | None = None,
    **kwargs,
) -> _StubAgent:
    """Helper to create a stub agent with optional routing/refinement config."""
    config = AgentInstanceConfig(
        agent_id="test_stub",
        agent_type="_stub",
        name="Test Stub",
        vertical_id="test_vertical",
        routing=routing or RoutingConfig(),
        refinement=refinement or RefinementConfig(),
        **kwargs,
    )
    return _StubAgent(
        config=config,
        db=MagicMock(),
        embedder=MagicMock(),
        anthropic_client=MagicMock(),
    )


# ─── RoutingConfig Schema Tests ──────────────────────────────────


class TestRoutingConfig:
    """Tests for the RoutingConfig Pydantic model."""

    def test_defaults(self):
        rc = RoutingConfig()
        assert rc.enabled is False
        assert rc.model_type == ModelType.LOCAL_CLASSIFICATION
        assert rc.intent_actions == {}
        assert rc.fallback_action == "proceed"
        assert rc.confidence_threshold == 0.8

    def test_custom_config(self):
        rc = RoutingConfig(
            enabled=True,
            intent_actions={"out_of_office": "sleep", "spam": "discard"},
            fallback_action="proceed",
            confidence_threshold=0.9,
        )
        assert rc.enabled is True
        assert rc.intent_actions["out_of_office"] == "sleep"
        assert rc.intent_actions["spam"] == "discard"
        assert rc.confidence_threshold == 0.9

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            RoutingConfig(confidence_threshold=1.5)
        with pytest.raises(Exception):
            RoutingConfig(confidence_threshold=-0.1)

    def test_model_type_enum(self):
        assert ModelType.LLM.value == "llm"
        assert ModelType.LOCAL_CLASSIFICATION.value == "local_classification"

    def test_in_agent_config(self):
        """RoutingConfig should be embeddable in AgentInstanceConfig."""
        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="test",
            name="Test",
            routing=RoutingConfig(
                enabled=True,
                intent_actions={"spam": "discard"},
            ),
        )
        assert config.routing.enabled is True
        assert config.routing.intent_actions["spam"] == "discard"

    def test_from_yaml_dict(self):
        """Should parse from raw dict (as YAML loader would produce)."""
        raw = {
            "enabled": True,
            "model_type": "local_classification",
            "intent_actions": {"ooo": "sleep"},
            "confidence_threshold": 0.85,
        }
        rc = RoutingConfig(**raw)
        assert rc.intent_actions["ooo"] == "sleep"


# ─── RefinementConfig Schema Tests ───────────────────────────────


class TestRefinementConfig:
    """Tests for the RefinementConfig Pydantic model."""

    def test_defaults(self):
        rf = RefinementConfig()
        assert rf.enabled is False
        assert rf.critic_prompt == ""
        assert rf.max_iterations == 1
        assert rf.quality_threshold == 0.7

    def test_custom_config(self):
        rf = RefinementConfig(
            enabled=True,
            critic_prompt="Rate this email 1-10 for professionalism.",
            max_iterations=3,
            quality_threshold=0.8,
        )
        assert rf.enabled is True
        assert "professionalism" in rf.critic_prompt
        assert rf.max_iterations == 3
        assert rf.quality_threshold == 0.8

    def test_max_iterations_bounds(self):
        with pytest.raises(Exception):
            RefinementConfig(max_iterations=0)
        with pytest.raises(Exception):
            RefinementConfig(max_iterations=6)

    def test_quality_threshold_bounds(self):
        with pytest.raises(Exception):
            RefinementConfig(quality_threshold=1.5)
        with pytest.raises(Exception):
            RefinementConfig(quality_threshold=-0.1)

    def test_in_agent_config(self):
        """RefinementConfig should be embeddable in AgentInstanceConfig."""
        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="test",
            name="Test",
            refinement=RefinementConfig(
                enabled=True,
                critic_prompt="Is this professional?",
                max_iterations=2,
            ),
        )
        assert config.refinement.enabled is True
        assert config.refinement.max_iterations == 2

    def test_from_yaml_dict(self):
        """Should parse from raw dict (as YAML loader would produce)."""
        raw = {
            "enabled": True,
            "critic_prompt": "Check for compliance issues.",
            "max_iterations": 2,
            "quality_threshold": 0.9,
        }
        rf = RefinementConfig(**raw)
        assert rf.quality_threshold == 0.9


# ─── ModelType in AgentModelConfig Tests ──────────────────────────


class TestModelTypeInConfig:
    """Tests for model_type field on AgentModelConfig."""

    def test_default_model_type(self):
        mc = AgentModelConfig()
        assert mc.model_type == ModelType.LLM

    def test_local_classification_type(self):
        mc = AgentModelConfig(model_type="local_classification")
        assert mc.model_type == ModelType.LOCAL_CLASSIFICATION

    def test_invalid_model_type(self):
        with pytest.raises(Exception):
            AgentModelConfig(model_type="quantum")


# ─── BaseAgent._route_task() Tests ───────────────────────────────


class TestRouteTask:
    """Tests for the neural router in BaseAgent."""

    def test_disabled_routing_returns_proceed(self):
        agent = _make_agent(routing=RoutingConfig(enabled=False))
        action = agent._route_task({"some": "task"})
        assert action == "proceed"

    def test_no_intent_returns_fallback(self):
        agent = _make_agent(
            routing=RoutingConfig(
                enabled=True,
                intent_actions={"spam": "discard"},
                fallback_action="proceed",
            )
        )
        action = agent._route_task({"data": "no intent here"})
        assert action == "proceed"

    def test_matched_intent_with_high_confidence(self):
        agent = _make_agent(
            routing=RoutingConfig(
                enabled=True,
                intent_actions={"out_of_office": "sleep", "spam": "discard"},
                confidence_threshold=0.8,
            )
        )
        action = agent._route_task({
            "classified_intent": "out_of_office",
            "classification_confidence": 0.95,
        })
        assert action == "sleep"

    def test_matched_intent_low_confidence_falls_back(self):
        agent = _make_agent(
            routing=RoutingConfig(
                enabled=True,
                intent_actions={"spam": "discard"},
                confidence_threshold=0.8,
                fallback_action="proceed",
            )
        )
        action = agent._route_task({
            "classified_intent": "spam",
            "classification_confidence": 0.5,  # Below threshold
        })
        assert action == "proceed"

    def test_unknown_intent_returns_fallback(self):
        agent = _make_agent(
            routing=RoutingConfig(
                enabled=True,
                intent_actions={"spam": "discard"},
                fallback_action="proceed",
            )
        )
        action = agent._route_task({
            "classified_intent": "unknown_intent",
            "classification_confidence": 0.99,
        })
        assert action == "proceed"

    def test_custom_fallback_action(self):
        agent = _make_agent(
            routing=RoutingConfig(
                enabled=True,
                intent_actions={},
                fallback_action="queue_for_review",
            )
        )
        action = agent._route_task({
            "classified_intent": "something",
            "classification_confidence": 0.99,
        })
        assert action == "queue_for_review"


# ─── BaseAgent._run_refinement_loop() Tests ──────────────────────


class TestRefinementLoop:
    """Tests for the refinement loop placeholder in BaseAgent."""

    def test_disabled_returns_result_unchanged(self):
        agent = _make_agent(refinement=RefinementConfig(enabled=False))
        result = {"output": "hello"}

        loop = asyncio.new_event_loop()
        try:
            refined = loop.run_until_complete(
                agent._run_refinement_loop(result)
            )
        finally:
            loop.close()

        assert refined is result  # Same object, not copied

    def test_enabled_without_prompt_returns_unchanged(self):
        agent = _make_agent(
            refinement=RefinementConfig(
                enabled=True,
                critic_prompt="",  # No prompt
            )
        )
        result = {"output": "hello"}

        loop = asyncio.new_event_loop()
        try:
            refined = loop.run_until_complete(
                agent._run_refinement_loop(result)
            )
        finally:
            loop.close()

        assert refined is result

    def test_enabled_with_prompt_placeholder(self):
        """Enabled with prompt should pass through (placeholder)."""
        agent = _make_agent(
            refinement=RefinementConfig(
                enabled=True,
                critic_prompt="Is this professional?",
                max_iterations=2,
            )
        )
        result = {"draft": "Hello there!"}

        loop = asyncio.new_event_loop()
        try:
            refined = loop.run_until_complete(
                agent._run_refinement_loop(result)
            )
        finally:
            loop.close()

        # Placeholder returns result unchanged
        assert refined is result


# ─── Config Backward Compatibility Tests ──────────────────────────


class TestBackwardCompatibility:
    """Ensure existing configs without routing/refinement still work."""

    def test_minimal_config_still_works(self):
        """Config without routing/refinement should get defaults."""
        config = AgentInstanceConfig(
            agent_id="outreach",
            agent_type="outreach",
            name="Outreach Agent",
        )
        assert config.routing.enabled is False
        assert config.refinement.enabled is False

    def test_existing_outreach_yaml_compatible(self):
        """The existing outreach.yaml should still load correctly."""
        from pathlib import Path

        import yaml

        yaml_path = (
            Path(__file__).parent.parent
            / "verticals"
            / "enclave_guard"
            / "agents"
            / "outreach.yaml"
        )
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        raw["vertical_id"] = "enclave_guard"
        config = AgentInstanceConfig(**raw)

        assert config.agent_id == "outreach"
        # routing/refinement should get defaults
        assert config.routing.enabled is False
        assert config.refinement.enabled is False
