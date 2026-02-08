"""
Unit tests for the SEO Content Agent (Phase 1D).

Tests cover:
- Agent construction and configuration
- State preparation
- Node logic (research, draft, review, finalize)
- RLHF data capture on human edit
- Graph structure
- YAML config loading
"""

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from core.agents.state import SEOContentAgentState
from core.config.agent_schema import AgentInstanceConfig


def _run(coro):
    """Helper to run async code in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_seo_config(**overrides):
    """Create a minimal SEOContentAgent config."""
    defaults = {
        "agent_id": "seo_v1",
        "agent_type": "seo_content",
        "name": "SEO Content Agent",
        "vertical_id": "enclave_guard",
        "browser_enabled": True,
        "params": {
            "target_word_count": 1500,
            "tone": "Authoritative, Technical",
            "content_type": "blog_post",
        },
    }
    defaults.update(overrides)
    return AgentInstanceConfig(**defaults)


def _make_mock_db():
    """Create a mock DB with all required methods."""
    db = MagicMock()
    db.log_agent_run = MagicMock()
    db.reset_agent_errors = MagicMock()
    db.record_agent_error = MagicMock()
    db.store_content.return_value = {"id": "content-001", "status": "draft"}
    db.store_training_example.return_value = {"id": "te-001"}
    db.store_insight.return_value = {"id": "ins-001"}
    db.search_insights.return_value = [
        {"content": "CTOs respond well to compliance-focused messaging"},
        {"content": "Avoid generic security buzzwords — be specific"},
    ]
    return db


def _make_mock_llm():
    """Create a mock Anthropic client."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = (
        "# The Complete Guide to AI-Powered Penetration Testing\n\n"
        "Artificial intelligence is transforming how security teams "
        "identify vulnerabilities in enterprise networks. This guide "
        "covers the latest approaches, tools, and best practices for "
        "incorporating AI into your penetration testing workflow.\n\n"
        "## Why AI Changes Everything\n\n"
        "Traditional penetration testing relies on human expertise to "
        "identify attack vectors. AI augments this by analyzing patterns "
        "across thousands of previous assessments.\n\n"
        "## Getting Started\n\n"
        "Contact our team to learn how AI-powered testing can improve "
        "your security posture."
    )
    mock_response.content = [mock_content]
    llm.messages.create.return_value = mock_response
    return llm


def _make_mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1536
    return embedder


# ─── SEOContentAgent Construction ─────────────────────────────────────

class TestSEOContentAgentInit:
    """Tests for agent construction and properties."""

    def test_agent_type_registered(self):
        """seo_content agent type should be registered."""
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        # Import triggers registration
        from core.agents.implementations.seo_content_agent import SEOContentAgent
        assert "seo_content" in AGENT_IMPLEMENTATIONS

    def test_construction(self):
        """Should construct with required dependencies."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        config = _make_seo_config()
        agent = SEOContentAgent(
            config=config,
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )
        assert agent.agent_id == "seo_v1"
        assert agent.vertical_id == "enclave_guard"

    def test_construction_with_browser(self):
        """Should accept browser_tool in constructor."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        browser = MagicMock()
        config = _make_seo_config()
        agent = SEOContentAgent(
            config=config,
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
            browser_tool=browser,
        )
        assert agent.browser_tool is browser

    def test_get_tools_returns_empty(self):
        """SEO agent uses browser, not MCP tools directly."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        agent = SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )
        assert agent.get_tools() == []

    def test_get_state_class(self):
        """Should return SEOContentAgentState."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        agent = SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )
        assert agent.get_state_class() is SEOContentAgentState

    def test_repr(self):
        """Should have a useful repr."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        agent = SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )
        r = repr(agent)
        assert "SEOContentAgent" in r
        assert "seo_v1" in r


# ─── State Preparation ────────────────────────────────────────────────

class TestSEOStatePreparation:
    """Tests for initial state preparation from task input."""

    def _make_agent(self):
        from core.agents.implementations.seo_content_agent import SEOContentAgent
        return SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )

    def test_prepare_state_from_task(self):
        """Should initialize state with topic and keywords."""
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={
                "topic": "AI penetration testing",
                "keywords": ["ai pentest", "automated security"],
                "content_type": "blog_post",
            },
            run_id="run-001",
        )
        assert state["topic"] == "AI penetration testing"
        assert state["target_keywords"] == ["ai pentest", "automated security"]
        assert state["content_type"] == "blog_post"
        assert state["competitor_analysis"] == []
        assert state["draft_content"] == ""

    def test_prepare_state_defaults(self):
        """Should use param defaults when task doesn't specify."""
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"topic": "SOC 2 compliance"},
            run_id="run-002",
        )
        assert state["content_type"] == "blog_post"  # From config.params
        assert state["target_keywords"] == []

    def test_prepare_state_includes_base_fields(self):
        """Should include BaseAgent fields (agent_id, run_id, etc.)."""
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"topic": "test"},
            run_id="run-003",
        )
        assert state["agent_id"] == "seo_v1"
        assert state["vertical_id"] == "enclave_guard"
        assert state["run_id"] == "run-003"


# ─── Node Logic ───────────────────────────────────────────────────────

class TestSEONodes:
    """Tests for individual graph node logic."""

    def _make_agent(self, browser=None):
        from core.agents.implementations.seo_content_agent import SEOContentAgent
        return SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
            browser_tool=browser,
        )

    # ── research_competitors ─────────────────────────────────────

    def test_research_without_browser(self):
        """Should still work without a browser (LLM-only analysis)."""
        agent = self._make_agent(browser=None)
        state: SEOContentAgentState = {
            "topic": "penetration testing",
            "target_keywords": ["pentest", "security assessment"],
        }

        result = _run(agent._node_research_competitors(state))

        assert result["current_node"] == "research_competitors"
        assert isinstance(result["competitor_analysis"], list)
        assert isinstance(result["outreach_insights"], list)

    def test_research_with_browser(self):
        """Should use browser for SERP research when available."""
        browser = MagicMock()
        browser.search_and_extract = AsyncMock(return_value={
            "success": True,
            "result": "1. CompetitorA.com - Best Pentest Guide\n2. CompetitorB.com - Pentest 101",
            "steps": 5,
        })

        agent = self._make_agent(browser=browser)
        state: SEOContentAgentState = {
            "topic": "penetration testing",
            "target_keywords": [],
        }

        result = _run(agent._node_research_competitors(state))

        browser.search_and_extract.assert_called_once()
        assert len(result["competitor_analysis"]) >= 1
        assert result["competitor_analysis"][0]["source"] == "serp"

    def test_research_handles_browser_failure(self):
        """Should gracefully handle browser errors."""
        browser = MagicMock()
        browser.search_and_extract = AsyncMock(side_effect=Exception("Browser crashed"))

        agent = self._make_agent(browser=browser)
        state: SEOContentAgentState = {
            "topic": "penetration testing",
            "target_keywords": [],
        }

        result = _run(agent._node_research_competitors(state))
        # Should not crash — returns what it can
        assert result["current_node"] == "research_competitors"

    def test_research_pulls_outreach_insights(self):
        """Should fetch shared brain insights from outreach agent."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "topic": "cybersecurity",
            "target_keywords": [],
        }

        result = _run(agent._node_research_competitors(state))

        # DB.search_insights should have been called
        agent.db.search_insights.assert_called_once()
        assert len(result["outreach_insights"]) == 2

    def test_research_handles_insight_failure(self):
        """Should work even when insights fetch fails."""
        agent = self._make_agent()
        agent.db.search_insights.side_effect = Exception("Vector search down")

        state: SEOContentAgentState = {
            "topic": "cybersecurity",
            "target_keywords": [],
        }

        result = _run(agent._node_research_competitors(state))
        assert result["outreach_insights"] == []

    # ── draft_content ────────────────────────────────────────────

    def test_draft_content_generates_text(self):
        """Should generate a draft with title and body."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "topic": "AI penetration testing",
            "content_type": "blog_post",
            "competitor_analysis": [],
            "outreach_insights": [],
            "target_keywords": ["ai pentest"],
        }

        result = _run(agent._node_draft_content(state))

        assert result["current_node"] == "draft_content"
        assert result["draft_title"] != ""
        assert result["draft_content"] != ""
        assert result["word_count"] > 0
        assert result["meta_title"] != ""

    def test_draft_uses_correct_model(self):
        """Should call LLM with the configured model."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "topic": "test",
            "content_type": "blog_post",
            "competitor_analysis": [],
            "outreach_insights": [],
            "target_keywords": [],
        }

        _run(agent._node_draft_content(state))

        call_kwargs = agent.llm.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-20250514"

    def test_draft_handles_llm_failure(self):
        """Should return error state when LLM fails."""
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("API rate limited")

        state: SEOContentAgentState = {
            "topic": "test",
            "content_type": "blog_post",
            "competitor_analysis": [],
            "outreach_insights": [],
            "target_keywords": [],
        }

        result = _run(agent._node_draft_content(state))
        assert "error" in result
        assert result["draft_content"] == ""

    # ── human_review ─────────────────────────────────────────────

    def test_human_review_sets_approval_flag(self):
        """Human review node should set requires_human_approval."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "draft_title": "Test Title",
            "word_count": 1500,
        }

        result = _run(agent._node_human_review(state))
        assert result["requires_human_approval"] is True

    # ── finalize_and_learn ───────────────────────────────────────

    def test_finalize_saves_content(self):
        """Should save content to agent_content table."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "draft_title": "Test Blog Post",
            "draft_content": "This is the draft content...",
            "content_type": "blog_post",
            "topic": "penetration testing",
            "target_keywords": ["pentest"],
            "meta_title": "Test",
            "meta_description": "Test desc",
            "competitor_analysis": [],
        }

        result = _run(agent._node_finalize_and_learn(state))

        agent.db.store_content.assert_called_once()
        assert result["content_approved"] is True

    def test_finalize_captures_rlhf_on_edit(self):
        """Should capture RLHF data when human edits the draft."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "draft_title": "Test",
            "draft_content": "Original draft...",
            "human_edited_content": "Improved version by human...",
            "content_type": "blog_post",
            "topic": "pentest",
            "target_keywords": [],
            "meta_title": "",
            "meta_description": "",
            "competitor_analysis": [],
        }

        result = _run(agent._node_finalize_and_learn(state))

        agent.db.store_training_example.assert_called_once()
        call_kwargs = agent.db.store_training_example.call_args.kwargs
        assert call_kwargs["model_output"] == "Original draft..."
        assert call_kwargs["human_correction"] == "Improved version by human..."
        assert result["rlhf_captured"] is True

    def test_finalize_no_rlhf_without_edit(self):
        """Should NOT capture RLHF when draft wasn't edited."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "draft_title": "Test",
            "draft_content": "Draft content...",
            "content_type": "blog_post",
            "topic": "test",
            "target_keywords": [],
            "meta_title": "",
            "meta_description": "",
            "competitor_analysis": [],
        }

        result = _run(agent._node_finalize_and_learn(state))

        agent.db.store_training_example.assert_not_called()
        assert result["rlhf_captured"] is False

    def test_finalize_writes_insight(self):
        """Should write keyword performance insight to shared brain."""
        agent = self._make_agent()
        state: SEOContentAgentState = {
            "draft_title": "Test",
            "draft_content": "Content here...",
            "content_type": "blog_post",
            "topic": "cybersecurity",
            "target_keywords": ["security"],
            "meta_title": "",
            "meta_description": "",
            "competitor_analysis": [],
        }

        _run(agent._node_finalize_and_learn(state))

        agent.db.store_insight.assert_called_once()


# ─── Graph Structure ──────────────────────────────────────────────────

class TestSEOGraphStructure:
    """Tests for the LangGraph compilation."""

    def test_build_graph_succeeds(self):
        """Should build a valid LangGraph."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        agent = SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )
        graph = agent.build_graph()
        assert graph is not None

    def test_graph_cached_on_second_call(self):
        """get_graph() should return cached graph."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        agent = SEOContentAgent(
            config=_make_seo_config(),
            db=_make_mock_db(),
            embedder=_make_mock_embedder(),
            anthropic_client=_make_mock_llm(),
        )
        g1 = agent.get_graph()
        g2 = agent.get_graph()
        assert g1 is g2


# ─── Routing ──────────────────────────────────────────────────────────

class TestSEORouting:
    """Tests for the review routing logic."""

    def test_route_approved(self):
        """Approved review should route to finalize."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        result = SEOContentAgent._route_after_review(
            {"human_approval_status": "approved"}
        )
        assert result == "approved"

    def test_route_rejected(self):
        """Rejected review should route to END."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        result = SEOContentAgent._route_after_review(
            {"human_approval_status": "rejected"}
        )
        assert result == "rejected"

    def test_route_edited_treated_as_approved(self):
        """Edited review should route to finalize (approved)."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        result = SEOContentAgent._route_after_review(
            {"human_approval_status": "edited"}
        )
        assert result == "approved"

    def test_route_default_is_approved(self):
        """Missing status should default to approved."""
        from core.agents.implementations.seo_content_agent import SEOContentAgent

        result = SEOContentAgent._route_after_review({})
        assert result == "approved"


# ─── YAML Config ──────────────────────────────────────────────────────

class TestSEOYamlConfig:
    """Tests for the YAML config file."""

    @pytest.fixture
    def yaml_path(self):
        return Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "seo_content.yaml"

    def test_yaml_exists(self, yaml_path):
        """SEO agent YAML config should exist."""
        assert yaml_path.exists()

    def test_yaml_loads_valid_config(self, yaml_path):
        """YAML should parse into a valid AgentInstanceConfig."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        config = AgentInstanceConfig(**data)
        assert config.agent_id == "seo_v1"
        assert config.agent_type == "seo_content"
        assert config.browser_enabled is True

    def test_yaml_has_correct_params(self, yaml_path):
        """YAML should have the expected agent params."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        config = AgentInstanceConfig(**data)
        assert config.params["target_word_count"] == 1500
        assert "tone" in config.params
        assert "target_topics" in config.params

    def test_yaml_has_human_gate(self, yaml_path):
        """YAML should configure human_review gate."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        config = AgentInstanceConfig(**data)
        assert config.human_gates.enabled is True
        assert "human_review" in config.human_gates.gate_before

    def test_yaml_model_config(self, yaml_path):
        """YAML should configure model with provider."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        config = AgentInstanceConfig(**data)
        assert config.model.provider == "anthropic"
        assert config.model.max_tokens == 8192


# ─── System Prompt ────────────────────────────────────────────────────

class TestSEOSystemPrompt:
    """Tests for the system prompt file."""

    @pytest.fixture
    def prompt_path(self):
        return Path(__file__).parent.parent / "verticals" / "enclave_guard" / "prompts" / "agent_prompts" / "seo_system.md"

    def test_prompt_file_exists(self, prompt_path):
        """System prompt file should exist."""
        assert prompt_path.exists()

    def test_prompt_contains_key_sections(self, prompt_path):
        """System prompt should cover identity, standards, and audience."""
        content = prompt_path.read_text()
        assert "Enclave Guard" in content
        assert "cybersecurity" in content.lower()
        assert "SEO" in content or "seo" in content.lower()


# ─── SEOContentAgentState ─────────────────────────────────────────────

class TestSEOContentAgentState:
    """Tests for the enhanced state schema."""

    def test_state_has_topic_field(self):
        """State should have topic field."""
        state: SEOContentAgentState = {"topic": "test topic"}
        assert state["topic"] == "test topic"

    def test_state_has_competitor_analysis(self):
        """State should have competitor_analysis field."""
        state: SEOContentAgentState = {
            "competitor_analysis": [
                {"url": "example.com", "design_score": 8, "summary": "Good site"},
            ],
        }
        assert len(state["competitor_analysis"]) == 1

    def test_state_has_rlhf_captured(self):
        """State should have rlhf_captured field."""
        state: SEOContentAgentState = {"rlhf_captured": True}
        assert state["rlhf_captured"] is True

    def test_state_has_outreach_insights(self):
        """State should have outreach_insights for shared brain."""
        state: SEOContentAgentState = {
            "outreach_insights": [{"content": "CTOs hate fluff"}],
        }
        assert len(state["outreach_insights"]) == 1

    def test_state_has_human_edited_content(self):
        """State should have human_edited_content for RLHF."""
        state: SEOContentAgentState = {
            "human_edited_content": "Improved by human",
        }
        assert state["human_edited_content"] == "Improved by human"
